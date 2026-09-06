"""Phase 2 update-projection interventions (doc section 6.4).

Applied to the realized update H of tracked-type matrices AFTER the optimizer
step: W <- W_before + P(H). All operate in the weight SVD basis of W_before.
"""

import torch

from .metrics import weight_svd


@torch.no_grad()
def project_update(W_before: torch.Tensor, H: torch.Tensor, mode: str,
                   q: float = 0.1, snr: torch.Tensor = None) -> torch.Tensor:
    """Return the projected update.

    mode:
      spectrum_only : U diag(diag C) V^T                  (H5)
      frame_only    : H - spectrum_only                   (H5)
      mag_topq      : keep largest-|C| q-fraction entries (H6 baseline)
      snr_topq      : keep highest-SNR q-fraction entries (H6, needs snr matrix)
    """
    U, S, V = weight_svd(W_before)
    Hf = H.float()
    C = U.T @ Hf @ V
    spec = U @ torch.diag(torch.diagonal(C)) @ V.T
    if mode == "spectrum_only":
        return spec.to(H.dtype)
    if mode == "frame_only":
        # exact complement: includes the part of H outside the thin-SVD span
        # (directions with sigma=0), which is frame motion by definition
        return (Hf - spec).to(H.dtype)
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
