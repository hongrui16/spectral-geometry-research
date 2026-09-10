"""Offline metric computation over capture dirs -> one CSV per run.

For every capture file (save step) and every tracked matrix computes:
  geometry of G and H (R_spectrum, subspace energies, stable rank, alignment)
  pooled SNR stats from G_b (v1 H4 statistic; kept for results/v1 comparability)
  cross-sample (split-half) statistics from G_b (v2 doc §5.3-5.4):
    R_sigma_cross            noise-corrected diagonal energy fraction
    sig_energy_diag/total    cross-sample signal energy Σ_i mA_ii mB_ii, Σ_ij mA_ij mB_ij
    noise_energy             Σ_ij pooled per-microbatch variance
    snr_cross_{diag,offdiag}_median   mA*mB / var_pool (can be negative)
    spearman_absC_snr_cross  rank corr of |mA| (half A) vs SNR_B (half B)
    spearman_absC_split      rank corr of |mA| vs |mB| (split-half reliability)
  rollout correlations rho_sigma / rho_frame if present (H3)
  H self-check for spectrum-type interventions (v2 doc §12.4)

Split-half convention: G_b keys are in microbatch order. For RLVR (2 seqs per
microbatch, K=8) the first 4 microbatches are prompt-group 0 and the next 4
are prompt-group 1, so first-half / second-half are independent groups. For
SFT/OPD every microbatch is one independent sequence.

Usage: python analysis/compute_metrics_v2.py runs/phase1_sft_adamw [...more runs]
Writes <run>/metrics.csv, <run>/rho_hist.pt and <run>/manifest.json
"""

import glob
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom import metrics

SPECTRUM_MODES = ("spectrum_only", "spectrum_matched")


def spearman(x, y):
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = (rx - rx.mean()) / rx.std().clamp_min(1e-12)
    ry = (ry - ry.mean()) / ry.std().clamp_min(1e-12)
    return (rx * ry).mean().item()


@torch.no_grad()
def cross_sample_stats(Cs: torch.Tensor, n_sub: int = 20000) -> dict:
    """Split-half statistics of B projected microbatch gradients Cs (B, r, r)."""
    B = Cs.shape[0]
    if B < 4:
        return {}
    A_, B_ = Cs[: B // 2], Cs[B // 2:]
    mA, mB = A_.mean(0), B_.mean(0)
    vA = A_.var(0, unbiased=True)
    vB = B_.var(0, unbiased=True)
    var_pool = 0.5 * (vA + vB)
    cross = mA * mB                                   # unbiased for mu_ij^2
    sig_diag = cross.diagonal().sum().item()
    sig_tot = cross.sum().item()
    noise = var_pool.sum().item()
    snr_cross = cross / var_pool.clamp_min(1e-30)
    snr_B = mB.pow(2) / vB.clamp_min(1e-30)
    out = {
        "sig_energy_diag": sig_diag,
        "sig_energy_total": sig_tot,
        "noise_energy": noise,
        # ratio left undefined (NaN) when total signal energy is not resolved
        "R_sigma_cross": sig_diag / sig_tot if sig_tot > 0 else float("nan"),
        "snr_cross_diag_median": snr_cross.diagonal().median().item(),
        "snr_cross_offdiag_median": snr_cross.median().item(),
    }
    n = min(n_sub, mA.numel())
    idx = torch.randperm(mA.numel(), generator=torch.Generator().manual_seed(0))[:n]
    a = mA.flatten()[idx].abs()
    out["spearman_absC_snr_cross"] = spearman(a, snr_B.flatten()[idx])
    out["spearman_absC_split"] = spearman(a, mB.flatten()[idx].abs())
    return out


def git_commit():
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def process_run(run_dir, device="cpu", out_dir=None):
    """out_dir: where metrics.csv / rho_hist.pt / manifest.json go
    (default: run_dir itself)."""
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(run_dir, "capture", "step_*.pt")))
    args_path = os.path.join(run_dir, "args.json")
    args = json.load(open(args_path)) if os.path.exists(args_path) else {}
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
                row["n_microbatch"] = Cs.shape[0]
                row["snr_diag_median"] = snr.diagonal().median().item()
                row["snr_offdiag_median"] = snr.median().item()
                # v1 H4 statistic: |C| and SNR from the SAME samples (coupled;
                # kept only for comparability with results/v1)
                n = min(20000, mean.numel())
                idx = torch.randperm(mean.numel())[:n]
                row["spearman_absC_snr"] = spearman(
                    mean.flatten()[idx].abs(), snr.flatten()[idx])
                row.update(cross_sample_stats(Cs))
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
    out = os.path.join(out_dir, "metrics.csv")
    df.to_csv(out, index=False)
    if rho_store:
        torch.save(rho_store, os.path.join(out_dir, "rho_hist.pt"))
    print(f"wrote {out} ({len(df)} rows)")

    # --- H self-check (v2 doc §12.4): the captured H of a spectrum-type
    # intervention must be ~diagonal in the captured W_before basis.
    check = {"intervention": args.get("intervention", "none")}
    if len(df) and "H_r_spectrum" in df:
        check["H_r_spectrum_median"] = float(df.H_r_spectrum.median())
        check["H_enrich_median"] = float((df.H_r_spectrum / df.r_null).median())
    if check["intervention"] in SPECTRUM_MODES and len(df):
        ok = check["H_r_spectrum_median"] > 0.5
        check["passed"] = bool(ok)
        msg = ("OK" if ok else
               "FAILED: captured H is not spectral -> check that engine.post_step "
               "runs before instr.post_optimizer and that tracked params are the "
               "engine's params; do NOT use H-side metrics of this run")
        print(f"H self-check [{check['intervention']}]: "
              f"median R_sigma(H)={check['H_r_spectrum_median']:.3g} -> {msg}")

    manifest = {
        "run": os.path.basename(run_dir.rstrip("/")),
        "args": args,
        "n_capture_files": len(files),
        "steps": [int(x) for x in sorted(df.step.unique())] if len(df) else [],
        "n_matrices": int(df.matrix.nunique()) if len(df) else 0,
        "columns": list(df.columns),
        "h_check": check,
        "analysis_commit": git_commit(),
        "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out-root", default=None,
                    help="write <out-root>/<run name>/ instead of into the run dir "
                         "(e.g. results/v2/result_A)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if (torch.cuda.is_available() and not a.cpu) else "cpu"
    for run in a.runs:
        od = (os.path.join(a.out_root, os.path.basename(run.rstrip("/")))
              if a.out_root else None)
        process_run(run, device=dev, out_dir=od)
