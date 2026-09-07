"""Offline metric computation over capture dirs -> one CSV per run.

For every capture file (save step) and every tracked matrix computes:
  geometry of G and H (R_spectrum, subspace energies, stable rank, alignment)
  SNR stats from G_b (H4), incl. rank correlation of |C| vs SNR
  rollout correlations rho_sigma / rho_frame if present (H3)

Usage: python analysis/compute_metrics.py runs/phase1_sft_adamw [...more runs]
Writes <run>/metrics.csv and <run>/rho_hist.pt
"""

import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom import metrics


def spearman(x, y):
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = (rx - rx.mean()) / rx.std().clamp_min(1e-12)
    ry = (ry - ry.mean()) / ry.std().clamp_min(1e-12)
    return (rx * ry).mean().item()


def process_run(run_dir, device="cpu"):
    files = sorted(glob.glob(os.path.join(run_dir, "capture", "step_*.pt")))
    rows, rho_store = [], []
    for f in files:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        step = rec["step"]
        for name, m in rec["mats"].items():
            W = m["W"].float().to(device)
            G = m["G"].float().to(device)
            H = m["H"].float().to(device)
            U, S, V = metrics.weight_svd(W)
            # null baseline for R_spectrum of an isotropic random G is
            # r/(m*n) = 1/max(m,n); report shapes so plots can normalize
            row = {"step": step, "matrix": name,
                   "m": W.shape[0], "n": W.shape[1],
                   "r_null": 1.0 / max(W.shape)}
            for pref, X in (("G", G), ("H", H)):
                rep = metrics.full_report(X, W, U, S, V)
                row.update({f"{pref}_{k}": v for k, v in rep.items()})
            if "G_b" in m:
                Cs = torch.stack([metrics.project(g.float().to(device), U, V)
                                  for g in m["G_b"].values()])
                st = metrics.snr_from_samples(Cs)
                snr, mean = st["snr"], st["mean"]
                row["snr_diag_median"] = snr.diagonal().median().item()
                row["snr_offdiag_median"] = snr.median().item()
                # H4: rank correlation between |C_ij| and SNR_ij (sampled entries)
                n = min(20000, mean.numel())
                idx = torch.randperm(mean.numel())[:n]
                row["spearman_absC_snr"] = spearman(
                    mean.flatten()[idx].abs(), snr.flatten()[idx])
            if "rollout_G" in m:
                keys = sorted(m["rollout_G"].keys())
                Cks_all = torch.stack(
                    [metrics.project(m["rollout_G"][k].float().to(device), U, V)
                     for k in keys])
                A_all = torch.tensor(rec["rollout_adv"])
                sizes = rec.get("rollout_group_sizes", [len(keys)])
                off = 0
                meds_s, meds_f = [], []
                for gi, sz in enumerate(sizes):
                    Cks, A = Cks_all[off:off + sz], A_all[off:off + sz]
                    off += sz
                    if A.std() <= 1e-6:
                        continue
                    rho = metrics.rollout_correlations(Cks, A, S)
                    meds_s.append(rho["rho_sigma"].abs().median().item())
                    meds_f.append(rho["rho_frame"].abs().median().item())
                    rho_store.append({"step": step, "matrix": name, "group": gi,
                                      "rho_sigma": rho["rho_sigma"].cpu(),
                                      "rho_frame": rho["rho_frame"].cpu()})
                if meds_s:
                    row["rho_sigma_absmed"] = sum(meds_s) / len(meds_s)
                    row["rho_frame_absmed"] = sum(meds_f) / len(meds_f)
            rows.append(row)
        print(f"{run_dir}: step {step} done", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(run_dir, "metrics.csv")
    df.to_csv(out, index=False)
    if rho_store:
        torch.save(rho_store, os.path.join(run_dir, "rho_hist.pt"))
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for run in sys.argv[1:]:
        process_run(run, device=dev)
