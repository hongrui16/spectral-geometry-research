"""Spectral-frame metrics for gradients/updates relative to weight SVD basis.

Notation follows the paper document:
  W = U Sigma V^T,  C = U^T G V,
  R_spectrum(G) = ||diag(C)||^2 / ||G||_F^2      (Cor. 1.1)
  T^U_ij = sigma_j C_ij - sigma_i C_ji           (Prop. 5 frame generator coeff)

All functions expect float32 tensors on any device; SVD is done in float32.
"""

import torch


@torch.no_grad()
def weight_svd(W: torch.Tensor):
    """Full SVD of a weight matrix. Returns U (m,r), S (r,), V (n,r), r=min(m,n)."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return U, S, Vh.T


@torch.no_grad()
def project(G: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """C = U^T G V, shape (r, r)."""
    return U.T @ G.float() @ V


@torch.no_grad()
def r_spectrum(G: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> float:
    """||diag(U^T G V)||^2 / ||G||_F^2. With full (non-truncated) U,V the
    denominator equals ||C||_F^2."""
    C = project(G, U, V)
    return (torch.diagonal(C).pow(2).sum() / G.float().pow(2).sum().clamp_min(1e-30)).item()


@torch.no_grad()
def subspace_energy(G: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                    bounds=(0.01, 0.5)) -> dict:
    """Energy fraction of G in W's top/mid/tail singular subspaces.

    bounds are fractions of r: top = [0, b0*r), mid = [b0*r, b1*r), tail = rest.
    Energy in block (I,J) uses the projector U_I U_I^T G V_J V_J^T; we report
    the diagonal blocks (top->top etc.) which is the standard usage.
    """
    r = U.shape[1]
    i0, i1 = max(1, int(bounds[0] * r)), int(bounds[1] * r)
    C = project(G, U, V)
    tot = C.pow(2).sum().clamp_min(1e-30)
    return {
        "top": (C[:i0, :i0].pow(2).sum() / tot).item(),
        "mid": (C[i0:i1, i0:i1].pow(2).sum() / tot).item(),
        "tail": (C[i1:, i1:].pow(2).sum() / tot).item(),
        "top_rows": (C[:i0, :].pow(2).sum() / tot).item(),
        "tail_rows": (C[i1:, :].pow(2).sum() / tot).item(),
    }


@torch.no_grad()
def principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians) between column spaces of A (m,k) and B (m,l)."""
    Qa, _ = torch.linalg.qr(A.float())
    Qb, _ = torch.linalg.qr(B.float())
    s = torch.linalg.svdvals(Qa.T @ Qb).clamp(-1, 1)
    return torch.acos(s)


@torch.no_grad()
def stable_rank(G: torch.Tensor) -> float:
    s = torch.linalg.svdvals(G.float())
    return (s.pow(2).sum() / s[0].pow(2).clamp_min(1e-30)).item()


@torch.no_grad()
def alignment(G: torch.Tensor, W: torch.Tensor) -> float:
    """<G,W> / (||G|| ||W||) — Prop. 6 scale-invariance check."""
    G, W = G.float(), W.float()
    return ((G * W).sum() / (G.norm() * W.norm()).clamp_min(1e-30)).item()


@torch.no_grad()
def frame_coeffs(C: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """T^U_ij = sigma_j * C_ij - sigma_i * C_ji  (antisymmetric generator coeff).

    C: (r,r) projection of per-rollout grad of log-pi; S: (r,) singular values.
    """
    return C * S.unsqueeze(0) - C.T * S.unsqueeze(1)


@torch.no_grad()
def snr_from_samples(Cs: torch.Tensor) -> dict:
    """Given B independent minibatch projections Cs (B, r, r), compute per-entry
    mean, variance and SNR_hat = mean^2 / var (H4 statistic)."""
    mean = Cs.mean(dim=0)
    var = Cs.var(dim=0, unbiased=True).clamp_min(1e-30)
    return {"mean": mean, "var": var, "snr": mean.pow(2) / var}


@torch.no_grad()
def rollout_correlations(Cks: torch.Tensor, A: torch.Tensor, S: torch.Tensor) -> dict:
    """rho^Sigma_i = Corr_k(A_k, diag(C_k)_i); rho^frame_ij = Corr_k(A_k, T_ij(C_k)).

    Cks: (K, r, r) per-rollout projections of grad log-pi (NO advantage weighting).
    A:   (K,) group-normalized advantages.
    S:   (r,) singular values of W.
    Returns flattened correlation tensors (frame off-diagonal upper triangle).
    """
    K = Cks.shape[0]
    A = A.float()
    Ac = A - A.mean()
    denomA = Ac.pow(2).sum().sqrt().clamp_min(1e-12)

    diag = torch.diagonal(Cks, dim1=1, dim2=2)          # (K, r)  = S_i per rollout
    T = Cks * S.unsqueeze(0).unsqueeze(0) - Cks.transpose(1, 2) * S.unsqueeze(0).unsqueeze(-1)

    def corr(x):  # x: (K, ...)
        xc = x - x.mean(dim=0, keepdim=True)
        num = (xc * Ac.view(K, *[1] * (x.dim() - 1))).sum(dim=0)
        den = xc.pow(2).sum(dim=0).sqrt().clamp_min(1e-12) * denomA
        return num / den

    rho_sigma = corr(diag)                               # (r,)
    rho_frame_full = corr(T)                             # (r, r) antisymmetric
    iu = torch.triu_indices(T.shape[1], T.shape[2], offset=1)
    rho_frame = rho_frame_full[iu[0], iu[1]]
    return {"rho_sigma": rho_sigma, "rho_frame": rho_frame}


@torch.no_grad()
def full_report(G: torch.Tensor, W: torch.Tensor, U=None, S=None, V=None) -> dict:
    """All scalar geometry metrics for one matrix at one step."""
    if U is None:
        U, S, V = weight_svd(W)
    rep = {
        "r_spectrum": r_spectrum(G, U, V),
        "stable_rank_G": stable_rank(G),
        "alignment_GW": alignment(G, W),
        "G_fro": G.float().norm().item(),
    }
    rep.update({f"energy_{k}": v for k, v in subspace_energy(G, U, V).items()})
    k = min(16, U.shape[1])
    Ug, Sg, Vg = torch.svd_lowrank(G.float(), q=min(2 * k, min(G.shape)), niter=4)
    ang = principal_angles(Ug[:, :k], U[:, :max(1, int(0.01 * U.shape[1]))])
    rep["pangle_Gtop_Wtop_mean"] = ang.mean().item()
    return rep
