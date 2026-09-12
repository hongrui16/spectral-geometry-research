"""M1: Spectral-SNR Descent (SSD). Draft v1 — refined after Phase 1 results.

Per 2D parameter:
  buf   momentum EMA of gradients (like Muon)
  P, Q  top-k singular basis of buf, aligned to previous step's basis
  m, v  per-mode EMAs of c_l = p_l^T G q_l  ->  SNR_l = m_l^2 / v_l
Update:
  wiener : H = P diag(f(SNR) * c) Q^T + tail_coef * (buf - P P^T buf Q Q^T)
  muon   : H = P diag(f(SNR) * sign(c)) Q^T + tail (Muon-flat but SNR-gated)
with f(s) = s / (1 + s). f == 1 recovers Muon on the top-k subspace.
"""

import torch

from .muon import newton_schulz


@torch.no_grad()
def _align(P_new, Q_new, c_new, P_old):
    """Greedy mode matching on |P_old^T P_new| overlap; returns permuted bases.

    Keeps EMA statistics attached to the physically-same mode across steps.
    """
    if P_old is None:
        return P_new, Q_new, c_new, torch.arange(P_new.shape[1])
    M = (P_old.T @ P_new).abs()          # (k_old, k_new)
    k = min(M.shape)
    perm = torch.full((M.shape[1],), -1, dtype=torch.long)
    used_r, used_c = set(), set()
    vals, idx = M.flatten().sort(descending=True)
    for f in idx.tolist():
        r, c = divmod(f, M.shape[1])
        if r in used_r or c in used_c:
            continue
        perm[c] = r
        used_r.add(r), used_c.add(c)
        if len(used_r) == k:
            break
    order = perm.argsort()
    order = order[perm[order] >= 0]
    return P_new[:, order], Q_new[:, order], c_new[order], perm


class SSD(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, momentum=0.95, beta1=0.9, beta2=0.99,
                 k=256, variant="wiener", tail_coef=0.1, eps=1e-12,
                 no_align=False):
        defaults = dict(lr=lr, momentum=momentum, beta1=beta1, beta2=beta2,
                        k=k, variant=variant, tail_coef=tail_coef, eps=eps,
                        no_align=no_align)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad.float()
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(G)
                    st["m"] = None
                    st["v"] = None
                    st["P"] = None
                buf = st["buf"]
                buf.mul_(group["momentum"]).add_(G)

                k = min(group["k"], min(G.shape))
                P, S, Q = torch.svd_lowrank(buf, q=min(2 * k, min(G.shape)),
                                            niter=4)
                P, Q = P[:, :k], Q[:, :k]
                c = torch.einsum("mk,mn,nk->k", P, G, Q)
                if not group["no_align"]:
                    P, Q, c, perm = _align(P, Q, c, st["P"])

                if st["m"] is None or st["m"].shape[0] != c.shape[0]:
                    st["m"] = torch.zeros_like(c)
                    st["v"] = torch.full_like(c, group["eps"])
                st["m"].mul_(group["beta1"]).add_(c, alpha=1 - group["beta1"])
                st["v"].mul_(group["beta2"]).add_(c * c, alpha=1 - group["beta2"])
                snr = st["m"].pow(2) / st["v"].clamp_min(group["eps"])
                f = snr / (1.0 + snr)

                if group["variant"] == "wiener":
                    diag = f * c / c.abs().max().clamp_min(group["eps"])
                else:  # 'muon'
                    diag = f * torch.sign(c)
                core = P @ torch.diag(diag) @ Q.T
                tail = buf - P @ (P.T @ buf @ Q) @ Q.T
                tail = newton_schulz(tail) * group["tail_coef"]
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_((core + tail).to(p.dtype),
                       alpha=-group["lr"] * scale)
                st["P"] = P
        return loss
