"""Sanity-check capture files: shapes, dtypes, non-degenerate values, and a
first R_spectrum computation. Usage: check_capture.py <capture_dir> [...]"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom_v2 import metrics


def check(capture_dir):
    files = sorted(glob.glob(os.path.join(capture_dir, "step_*.pt")))
    assert files, f"no capture files in {capture_dir}"
    rec = torch.load(files[-1], map_location="cpu", weights_only=False)
    print(f"\n{capture_dir}: {len(files)} files, last step {rec['step']}, "
          f"{len(rec['mats'])} matrices")
    name = list(rec["mats"].keys())[0]
    m = rec["mats"][name]
    G, H, W = m["G"].float(), m["H"].float(), m["W"].float()
    print(f"  [{name}] shape {tuple(W.shape)}")
    print(f"  |G|={G.norm():.3e} |H|={H.norm():.3e} |W|={W.norm():.3e}")
    assert G.norm() > 0 and H.norm() > 0, "degenerate G or H"
    U, S, V = metrics.weight_svd(W)
    print(f"  R_spectrum(G)={metrics.r_spectrum(G, U, V):.4f} "
          f"R_spectrum(H)={metrics.r_spectrum(H, U, V):.4f}")
    if "G_b" in m:
        Cs = torch.stack([metrics.project(g.float(), U, V)
                          for g in m["G_b"].values()])
        snr = metrics.snr_from_samples(Cs)
        print(f"  G_b: {len(m['G_b'])} samples, "
              f"median SNR diag={snr['snr'].diagonal().median():.3f} "
              f"offdiag={snr['snr'].median():.3f}")
    if "rollout_G" in m:
        Cks = torch.stack([metrics.project(g.float(), U, V)
                           for g in m["rollout_G"].values()])
        A = torch.tensor(rec["rollout_adv"])
        rho = metrics.rollout_correlations(Cks, A, S)
        print(f"  rollouts: {Cks.shape[0]}, "
              f"|rho_sigma| median={rho['rho_sigma'].abs().median():.3f} "
              f"|rho_frame| median={rho['rho_frame'].abs().median():.3f}")
    print("  OK")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        check(d)
