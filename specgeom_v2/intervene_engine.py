"""Intervention engine: project every decoder matrix's realized update in the
(cached) weight SVD basis, every step.

Basis (U, S, V) per matrix is refreshed every `refresh_every` steps (updates are
small, so the frozen-basis approximation between refreshes is first-order
accurate — same approximation as doc M2). Per-mode SNR is an EMA over steps of
the projected raw gradient: m_ij ~ EMA(C_ij), v_ij ~ EMA(C_ij^2),
SNR = m^2 / v  (online version of the B-microbatch estimator).

v2 additions (doc §11 E4a, equal-Frobenius-step comparison):
  spectrum_matched / frame_matched / random_ext / exact_iso
  - matched modes rescale the projected part to ||H||_F; the per-matrix scale
    factor is logged in `scale_log` (train.py writes its mean as scale_mean).
  - random_ext uses a fixed random orthonormal dictionary per matrix, drawn
    once at construction from `seed` (same dimension r as the spectral one).
  - exact_iso resets the singular values to the anchor spectrum captured at
    construction (the pretrained checkpoint's spectrum), every step. This
    costs one SVD per matrix per step.
"""

import re

import torch

from .interventions import match_norm, random_basis


class InterventionEngine:
    def __init__(self, model, mode, q=0.1, refresh_every=10,
                 beta1=0.9, beta2=0.99, eps=1e-12, seed=0):
        self.mode = mode
        self.q = q
        self.refresh_every = refresh_every
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        pat = re.compile(r"layers\.\d+\.(?:self_attn|mlp|linear_attn)\.\w+\.weight$")
        self.params = {n: p for n, p in model.named_parameters()
                       if p.dim() == 2 and pat.search(n)
                       and "visual" not in n and not n.startswith("mtp.")}
        self.basis = {}      # name -> (U, V)
        self.S_anchor = {}   # name -> singular values at construction (exact_iso)
        self.rand = {}       # name -> (Ur, Vr) fixed random dictionary (random_ext)
        self.m = {}
        self.v = {}
        self.alpha_log = {}  # adaptive_alpha: latest alpha per matrix (Fig.10)
        self.scale_log = {}  # matched modes: latest ||H||/||part|| per matrix
        self._w_before = None
        self._step = 0
        if mode == "exact_iso":
            for n, p in self.params.items():
                self.S_anchor[n] = torch.linalg.svdvals(p.detach().float())
        if mode == "random_ext":
            g = torch.Generator().manual_seed(seed)
            for n, p in self.params.items():
                m_, n_ = p.shape
                self.rand[n] = random_basis(m_, n_, min(m_, n_), generator=g,
                                            device=p.device)

    @torch.no_grad()
    def _refresh_basis(self):
        for n, p in self.params.items():
            U, S, Vh = torch.linalg.svd(p.detach().float(), full_matrices=False)
            self.basis[n] = (U, Vh.T)

    @torch.no_grad()
    def observe_grad(self):
        """Call after backward, before optimizer.step (needs p.grad)."""
        if self.mode not in ("snr_topq", "adaptive_alpha"):
            return
        for n, p in self.params.items():
            U, V = self.basis[n]
            C = U.T @ p.grad.detach().float() @ V
            if n not in self.m:
                self.m[n] = torch.zeros_like(C)
                self.v[n] = torch.full_like(C, self.eps)
            self.m[n].mul_(self.beta1).add_(C, alpha=1 - self.beta1)
            self.v[n].mul_(self.beta2).add_(C * C, alpha=1 - self.beta2)

    @torch.no_grad()
    def pre_step(self):
        if self._step % self.refresh_every == 0 and self.mode != "exact_iso":
            self._refresh_basis()
        self._w_before = {n: p.detach().clone() for n, p in self.params.items()}

    @torch.no_grad()
    def post_step(self):
        self._step += 1
        for n, p in self.params.items():
            H = (p.detach() - self._w_before[n]).float()
            if self.mode == "exact_iso":
                W_after = self._w_before[n].float() + H
                Ua, _, Vah = torch.linalg.svd(W_after, full_matrices=False)
                W_new = Ua @ torch.diag(self.S_anchor[n]) @ Vah
                p.copy_(W_new.to(p.dtype))
                continue
            U, V = self.basis[n]
            C = U.T @ H @ V
            spec = U @ torch.diag(C.diagonal()) @ V.T
            if self.mode == "spectrum_only":
                Hp = spec
            elif self.mode == "frame_only":
                Hp = H - spec
            elif self.mode == "spectrum_matched":
                Hp, self.scale_log[n] = match_norm(spec, H)
            elif self.mode == "frame_matched":
                Hp, self.scale_log[n] = match_norm(H - spec, H)
            elif self.mode == "random_ext":
                Ur, Vr = self.rand[n]
                Cr = Ur.T @ H @ Vr
                Hr = Ur @ torch.diag(Cr.diagonal()) @ Vr.T
                Hp, self.scale_log[n] = match_norm(Hr, H)
            elif self.mode == "mag_topq":
                k = max(1, int(self.q * C.numel()))
                th = C.abs().flatten().kthvalue(C.numel() - k + 1).values
                Hp = U @ (C * (C.abs() >= th)) @ V.T
            elif self.mode == "snr_topq":
                snr = self.m[n].pow(2) / self.v[n].clamp_min(self.eps)
                k = max(1, int(self.q * C.numel()))
                th = snr.flatten().kthvalue(snr.numel() - k + 1).values
                Hp = U @ (C * (snr >= th)) @ V.T
            elif self.mode == "adaptive_alpha":
                # M2: H' = H - (1 - alpha) * spectral part; alpha = Wiener gate
                # on the mean diagonal SNR (doc Cor. 13.1). alpha=1 -> full
                # update, alpha=0 -> ISO-like (first-order isospectral).
                snr_diag = (self.m[n].diagonal().pow(2)
                            / self.v[n].diagonal().clamp_min(self.eps))
                s = snr_diag.mean()
                alpha = (s / (1.0 + s)).item()
                self.alpha_log[n] = alpha
                Hp = H - (1.0 - alpha) * spec
            else:
                raise ValueError(self.mode)
            p.copy_(self._w_before[n] + Hp.to(p.dtype))
        self._w_before = None
