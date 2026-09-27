"""v5 stage-2 interventions on the principal subspace (docs/unified_paper_document_v5.md part 3 §2).

Object alignment: the opponents' "principal" object is the top-k singular SUBSPACE of W
(2511.08567 off-principal updates; 2509.12235 rotation of top singular vectors), not the
diagonal of U^T H V that v2-v4 manipulated (0.05% of the update energy -> vacuous arms).

For every 2D decoder matrix, with the realized optimizer update H = W_after - W_before and an
orthonormal pair (A: m x k, B: n x k) = top-k left/right singular vectors of W_before
(principal, refreshed every step) or a fixed random orthonormal pair (control):
    P_A H P_B  (PP)   P_A H Q_B (PQ)   Q_A H P_B (QP)   Q_A H Q_B (QQ),  Q = I - P
  *_norot : keep PP + QQ      -> block-diagonal in the frame: the top-k subspaces do not rotate
  *_avoid : keep QQ only      -> update never touches the top-k subspaces
  *_only  : keep PP + PQ + QP -> update lives only where it touches the top-k subspaces
  prefix p_ = principal pair, r_ = random pair of the same k.

Step matching (--match):
  none : apply the projected update as is (records how much the projection shrinks the step)
  norm : rescale each matrix to ||H||_F (v2-v4 protocol; kept only to show the trap)
  kl   : one global factor s for all matrices, s = sqrt(KL_dense / KL_arm) measured on a fixed
         probe batch every `kl_every` steps (single-step KL is quadratic in the step, A5), held
         in between. Requires frozen non-matrix params so that H covers the whole update.
Logged every step: cos(H_arm, H) and kept-energy fraction (non-vacuity check, part 4 rule 1).
"""

import math
import re

import torch

MODES = ("p_norot", "p_avoid", "p_only", "r_norot", "r_avoid", "r_only")


def _orth(m, k, g):
    Q, _ = torch.linalg.qr(torch.randn(m, k, generator=g, dtype=torch.float64))
    return Q.float()


def project(H, A, B, keep):
    """Return the part of H made of the requested blocks of the (A, B) frame."""
    AH = A.T @ H                 # k x n
    HB = H @ B                   # m x k
    PP = A @ (AH @ B) @ B.T
    if keep == "norot":          # PP + QQ = H - PQ - QP ; PQ + QP = A AH + HB B^T - 2 PP
        return H - (A @ AH) - (HB @ B.T) + 2 * PP
    if keep == "avoid":          # QQ = H - A AH - HB B^T + PP
        return H - (A @ AH) - (HB @ B.T) + PP
    if keep == "only":           # H - QQ
        return (A @ AH) + (HB @ B.T) - PP
    raise ValueError(keep)


class PrincipalEngine:
    def __init__(self, model, mode, frac=0.05, match="kl", kl_every=10, seed=0,
                 refresh_every=1, s_clip=(0.1, 30.0)):
        assert mode in MODES, mode
        self.mode, self.frac, self.match = mode, frac, match
        self.kind, self.keep = mode.split("_")
        self.kl_every, self.refresh_every, self.s_clip = kl_every, refresh_every, s_clip
        pat = re.compile(r"layers\.\d+\.(?:self_attn|mlp|linear_attn)\.\w+\.weight$")
        self.params = {n: p for n, p in model.named_parameters()
                       if p.dim() == 2 and pat.search(n)
                       and "visual" not in n and not n.startswith("mtp.")}
        self.k = {n: max(1, math.ceil(frac * min(p.shape))) for n, p in self.params.items()}
        self.frame = {}
        if self.kind == "r":
            g = torch.Generator().manual_seed(seed)
            for n, p in self.params.items():
                self.frame[n] = (_orth(p.shape[0], self.k[n], g).to(p.device),
                                 _orth(p.shape[1], self.k[n], g).to(p.device))
        self.s = 1.0
        self._step = 0
        self._w_before = None
        self.log = {}

    @torch.no_grad()
    def _refresh(self):
        for n, p in self.params.items():
            kw = {"driver": "gesvda"} if p.is_cuda else {}
            U, _, Vh = torch.linalg.svd(p.detach().float(), full_matrices=False, **kw)
            k = self.k[n]
            self.frame[n] = (U[:, :k].contiguous(), Vh[:k].T.contiguous())

    @torch.no_grad()
    def pre_step(self):
        if self.kind == "p" and (self._step % self.refresh_every == 0 or not self.frame):
            self._refresh()      # frame of W_before (current weights)
        self._w_before = {n: p.detach().clone() for n, p in self.params.items()}

    @torch.no_grad()
    def post_step(self, kl_fn=None):
        """kl_fn(set_weights) -> KL of the current params vs the W_before params on the probe
        batch; supplied by train.py on calibration steps when match == 'kl'."""
        self._step += 1
        self.log = {}
        Hp = {}
        e_h = e_p = dot = 0.0
        for n, p in self.params.items():
            h = (p.detach() - self._w_before[n]).float()
            A, B = self.frame[n]
            hp = project(h, A, B, self.keep)
            if self.match == "norm":
                hp = hp * (h.norm() / hp.norm().clamp_min(1e-30))
            Hp[n] = hp
            e_h += float((h * h).sum())
            e_p += float((hp * hp).sum())
            dot += float((h * hp).sum())
        calibrate = (self.match == "kl" and kl_fn is not None
                     and (self._step == 1 or self._step % self.kl_every == 0))
        if calibrate:
            kl_dense = kl_fn()                                  # params currently = W_before + H
            for n, p in self.params.items():
                p.copy_(self._w_before[n] + Hp[n].to(p.dtype))
            kl_arm = kl_fn()
            if kl_arm > 0 and kl_dense > 0:
                self.s = min(max(math.sqrt(kl_dense / kl_arm), self.s_clip[0]), self.s_clip[1])
            self.log.update(kl_dense=kl_dense, kl_arm_unscaled=kl_arm)
        s = self.s if self.match == "kl" else 1.0
        for n, p in self.params.items():
            p.copy_(self._w_before[n] + (s * Hp[n]).to(p.dtype))
        self.log.update(s=s, cos=dot / math.sqrt(max(e_h * e_p, 1e-60)),
                        kept_energy=e_p / max(e_h, 1e-60), step_norm_rel=s * math.sqrt(e_p / max(e_h, 1e-60)))
        self._w_before = None
        return self.log
