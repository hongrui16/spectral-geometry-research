"""Update-projection interventions (v1: doc §6.4; v2: doc §11 E4a).

Applied to the realized update H of tracked-type matrices AFTER the optimizer
step: W <- W_before + P(H). All operate in the weight SVD basis of W_before.

v1 modes (unmatched: the projected part keeps its own norm):
  spectrum_only : U diag(diag C) V^T                  (H5)
  frame_only    : H - spectrum_only                   (H5)
  mag_topq      : keep largest-|C| q-fraction entries (H6 baseline)
  snr_topq      : keep highest-SNR q-fraction entries (H6, needs snr matrix)

v2 modes (E4a, equal-Frobenius-step comparison):
  spectrum_matched : spectrum_only rescaled to ||H||_F
  frame_matched    : frame_only    rescaled to ||H||_F
  random_ext       : projection of H onto r random rank-1 directions
                     u'_i v'_i^T (fixed random orthonormal bases, same
                     dimension as the spectral dictionary), rescaled to ||H||_F
  exact_iso        : after the step, reset the singular values of W to an
                     anchor spectrum (ISO-style exact isospectral baseline);
                     returns W_new - W_before, which is NOT a projection of H

The v1 unmatched modes are kept for reproducibility of results/v1; new runs
must use the matched modes (v2 doc §9.4 caveat 1).
"""

import torch

from .metrics import weight_svd

UNMATCHED = ("spectrum_only", "frame_only", "mag_topq", "snr_topq")
MATCHED = ("spectrum_matched", "frame_matched", "random_ext", "exact_iso")
ALL_MODES = UNMATCHED + MATCHED + ("adaptive_alpha",)


@torch.no_grad()
def random_basis(m: int, n: int, r: int, generator: torch.Generator = None,
                 device=None, dtype=torch.float32):
    """Fixed random orthonormal bases Ur (m,r), Vr (n,r) for random_ext.

    Same dimension r as the thin-SVD spectral dictionary of an (m,n) matrix.
    """
    g = generator
    A = torch.randn(m, r, generator=g, dtype=dtype).to(device)
    B = torch.randn(n, r, generator=g, dtype=dtype).to(device)
    Ur, _ = torch.linalg.qr(A)
    Vr, _ = torch.linalg.qr(B)
    return Ur, Vr


@torch.no_grad()
def match_norm(Hp: torch.Tensor, H: torch.Tensor, eps: float = 1e-30):
    """Rescale Hp to ||H||_F. Returns (Hp_scaled, scale)."""
    s = H.norm() / Hp.norm().clamp_min(eps)
    return Hp * s, s.item()


@torch.no_grad()
def project_update(W_before: torch.Tensor, H: torch.Tensor, mode: str,
                   q: float = 0.1, snr: torch.Tensor = None,
                   rand_basis=None, S_anchor: torch.Tensor = None,
                   basis=None) -> torch.Tensor:
    """Return the projected (or, for exact_iso, re-spectralized) update.

    basis: optional precomputed (U, S, V) of W_before to avoid a re-SVD.
    rand_basis: (Ur, Vr) from random_basis(), required for random_ext.
    S_anchor: (r,) anchor singular values, required for exact_iso.
    """
    Hf = H.float()
    if mode == "exact_iso":
        assert S_anchor is not None, "exact_iso needs the anchor spectrum"
        W_after = W_before.float() + Hf
        Ua, _, Vah = torch.linalg.svd(W_after, full_matrices=False)
        W_new = Ua @ torch.diag(S_anchor.to(Ua.dtype)) @ Vah
        return (W_new - W_before.float()).to(H.dtype)

    if basis is None:
        U, S, V = weight_svd(W_before)
    else:
        U, S, V = basis
    C = U.T @ Hf @ V
    spec = U @ torch.diag(torch.diagonal(C)) @ V.T

    if mode == "spectrum_only":
        return spec.to(H.dtype)
    if mode == "frame_only":
        # exact complement: includes the part of H outside the thin-SVD span
        # (directions with sigma=0), which is frame motion by definition
        return (Hf - spec).to(H.dtype)
    if mode == "spectrum_matched":
        return match_norm(spec, Hf)[0].to(H.dtype)
    if mode == "frame_matched":
        return match_norm(Hf - spec, Hf)[0].to(H.dtype)
    if mode == "random_ext":
        assert rand_basis is not None, "random_ext needs rand_basis"
        Ur, Vr = rand_basis
        Cr = Ur.T @ Hf @ Vr
        Hr = Ur @ torch.diag(torch.diagonal(Cr)) @ Vr.T
        return match_norm(Hr, Hf)[0].to(H.dtype)

    if mode == "mag_topq":
        k = max(1, int(q * C.numel()))
        thresh = C.abs().flatten().kthvalue(C.numel() - k + 1).values
        Cp = C * (C.abs() >= thresh)
    elif mode == "snr_topq":
        assert snr is not None, "snr_topq needs the SNR matrix"
        k = max(1, int(q * C.numel()))
        thresh = snr.flatten().kthvalue(snr.numel() - k + 1).values
        Cp = C * (snr >= thresh)
    else:
        raise ValueError(mode)
    # top-q modes: both variants operate on the r x r block and drop the
    # complement equally, so the mag-vs-snr comparison stays apples-to-apples
    return (U @ Cp @ V.T).to(H.dtype)
