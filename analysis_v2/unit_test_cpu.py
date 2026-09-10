"""CPU unit tests for the torch-only parts of specgeom (no transformers needed).

Checks the math against the propositions in the paper doc:
  P1: sigma-dot = diag(C)  (finite-difference check)
  P2: spectrum-only update changes outputs only along existing (u_i, v_i)
  P3: R_spectrum of polar(G) invariant to joint rotations (Muon equivariance)
  rollout_correlations: planted-signal recovery
  SSD/Muon/InterventionEngine: run without error, sane outputs
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom_v2 import metrics
from specgeom_v2.interventions import project_update
from specgeom_v2.muon import Muon, newton_schulz
from specgeom_v2.ssd import SSD

torch.manual_seed(0)
m, n = 64, 48


def test_prop1_sigma_dot():
    W = torch.randn(m, n)
    H = 1e-5 * torch.randn(m, n)
    U, S, V = metrics.weight_svd(W)
    C = metrics.project(H, U, V)
    S2 = torch.linalg.svdvals(W + H)
    pred = S + torch.diagonal(C)
    err = (S2 - pred).abs().max() / S.max()
    assert err < 1e-6, f"P1 violated: {err}"
    print(f"P1 ok (err {err:.2e})")


def test_prop2_spectrum_only():
    W = torch.randn(m, n)
    U, S, V = metrics.weight_svd(W)
    H = torch.randn(m, n)
    Hs = project_update(W, H, "spectrum_only")
    # output change for random x must lie in span of U columns weighted by v_i^T x
    x = torch.randn(n)
    dy = Hs @ x
    coeff = U.T @ dy
    recon = U @ coeff
    assert (dy - recon).norm() / dy.norm() < 1e-5
    # and C' must be diagonal
    Cp = U.T @ Hs @ V
    off = Cp - torch.diag(torch.diagonal(Cp))
    assert off.norm() / Cp.norm() < 1e-5, "spectrum_only not diagonal"
    print("P2 ok")


def test_prop3_muon_equivariance():
    W = torch.randn(m, n)
    G = torch.randn(m, n)
    Qm, _ = torch.linalg.qr(torch.randn(m, m))
    Qn, _ = torch.linalg.qr(torch.randn(n, n))
    U, S, V = metrics.weight_svd(W)
    r1 = metrics.r_spectrum(newton_schulz(G, steps=15).float(), U, V)
    U2, S2, V2 = metrics.weight_svd(Qm @ W @ Qn.T)
    r2 = metrics.r_spectrum(newton_schulz(Qm @ G @ Qn.T, steps=15).float(), U2, V2)
    # sign ambiguity of SVD can permute degenerate modes; tolerance loose
    assert abs(r1 - r2) < 0.02, f"P3 violated: {r1} vs {r2}"
    print(f"P3 ok ({r1:.4f} vs {r2:.4f})")


def test_rollout_corr_planted():
    K, r = 64, 16
    S = torch.linspace(2, 0.1, r)
    A = torch.randn(K)
    # plant: diag coeff of mode 3 correlates with A; others noise
    Cks = 0.1 * torch.randn(K, r, r)
    Cks[:, 3, 3] += 2.0 * A
    out = metrics.rollout_correlations(Cks, A, S)
    assert out["rho_sigma"][3] > 0.9, out["rho_sigma"][3]
    assert out["rho_sigma"].abs().median() < 0.3
    print(f"rollout corr ok (planted {out['rho_sigma'][3]:.3f})")


def test_snr_planted():
    B, r = 8, 16
    mean = torch.zeros(r, r)
    mean[2, 5] = 1.0
    Cs = mean.unsqueeze(0) + 0.05 * torch.randn(B, r, r)
    st = metrics.snr_from_samples(Cs)
    assert st["snr"][2, 5] > 50, st["snr"][2, 5]
    print(f"SNR planted ok ({st['snr'][2, 5]:.1f})")


def test_optimizers_run():
    for opt_cls, kw in ((Muon, {}), (SSD, {"k": 16, "variant": "wiener"}),
                        (SSD, {"k": 16, "variant": "muon"})):
        p = torch.nn.Parameter(torch.randn(m, n))
        opt = opt_cls([p], lr=1e-3, **kw)
        for _ in range(3):
            opt.zero_grad()
            (p.sum() ** 2).backward()
            w0 = p.detach().clone()
            opt.step()
            assert (p.detach() - w0).norm() > 0
            assert torch.isfinite(p).all()
    print("optimizers ok")


def test_interventions():
    W = torch.randn(m, n)
    H = 1e-3 * torch.randn(m, n)
    spec = project_update(W, H, "spectrum_only")
    frame = project_update(W, H, "frame_only")
    # exact decomposition: spectrum + frame == H, and they are orthogonal
    assert (spec + frame - H).norm() / H.norm() < 1e-5, "decomposition broken"
    assert (spec * frame).sum().abs() / H.norm() ** 2 < 1e-5, "not orthogonal"
    for mode in ("mag_topq",):
        Hp = project_update(W, H, mode, q=0.1)
        assert torch.isfinite(Hp).all()
        assert Hp.norm() <= H.norm() * (1 + 1e-5), "projection grew the update"
    snr = torch.rand(min(m, n), min(m, n))
    Hp = project_update(W, H, "snr_topq", q=0.1, snr=snr)
    assert torch.isfinite(Hp).all()
    print("interventions ok")


if __name__ == "__main__":
    test_prop1_sigma_dot()
    test_prop2_spectrum_only()
    test_prop3_muon_equivariance()
    test_rollout_corr_planted()
    test_snr_planted()
    test_optimizers_run()
    test_interventions()
    print("ALL_CPU_TESTS_PASS")
