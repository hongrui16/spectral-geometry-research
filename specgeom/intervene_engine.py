"""Phase 2 intervention engine: project every decoder matrix's realized update
in the (cached) weight SVD basis, every step.

Basis (U, V) per matrix is refreshed every `refresh_every` steps (updates are
small, so the frozen-basis approximation between refreshes is first-order
accurate — same approximation as doc M2). Per-mode SNR is an EMA over steps of
the projected raw gradient: m_ij ~ EMA(C_ij), v_ij ~ EMA(C_ij^2),
SNR = m^2 / v  (online version of the B-microbatch estimator).
"""

import re

import torch


class InterventionEngine:
    def __init__(self, model, mode, q=0.1, refresh_every=10,
                 beta1=0.9, beta2=0.99, eps=1e-12):
        self.mode = mode
        self.q = q
        self.refresh_every = refresh_every
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        pat = re.compile(r"layers\.\d+\.(?:self_attn|mlp|linear_attn)\.\w+\.weight$")
        self.params = {n: p for n, p in model.named_parameters()
                       if p.dim() == 2 and pat.search(n)
                       and "visual" not in n and not n.startswith("mtp.")}
        self.basis = {}      # name -> (U, V)
        self.m = {}
        self.v = {}
        self.alpha_log = {}  # adaptive_alpha: latest alpha per matrix (Fig.10)
        self._w_before = None
        self._step = 0

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
        if self._step % self.refresh_every == 0:
            self._refresh_basis()
        self._w_before = {n: p.detach().clone() for n, p in self.params.items()}

    @torch.no_grad()
    def post_step(self):
        self._step += 1
        for n, p in self.params.items():
            U, V = self.basis[n]
            H = (p.detach() - self._w_before[n]).float()
            C = U.T @ H @ V
            if self.mode == "spectrum_only":
                Hp = U @ torch.diag(C.diagonal()) @ V.T
            elif self.mode == "frame_only":
                Hp = H - U @ torch.diag(C.diagonal()) @ V.T
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
                spec = U @ torch.diag(C.diagonal()) @ V.T
                Hp = H - (1.0 - alpha) * spec
            else:
                raise ValueError(self.mode)
            p.copy_(self._w_before[n] + Hp.to(p.dtype))
        self._w_before = None
