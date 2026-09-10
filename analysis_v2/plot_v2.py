"""v2 figures (docs/unified_paper_document_v2.md §13.1) from small result files.

  python analysis_v2/plot_v2.py --out figs/v2

Fig.A  equal-norm H5 main figure         <- results/v2/result_B/e4a_*  (P3; drawn when present)
Fig.B  adaptive-alpha convergence        <- results/v1/result_B/e4_{sft,opd,rlvr}_alpha/log.jsonl
Fig.C  split-half signal/noise energy    <- results/v2/result_A/phase1_*/metrics.csv
Fig.D  v1 coupled Spearman vs split-half <- results/v2/result_A/phase1_*/metrics.csv
All panels use medians over matrices with IQR bands; nothing is smoothed.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"sft": "tab:blue", "opd": "tab:green", "rlvr": "tab:red"}
LABEL = {"sft": "SFT", "opd": "OPD", "rlvr": "RLVR"}


def _band(ax, x, lo, med, hi, color, label):
    ax.plot(x, med, color=color, label=label, lw=1.6)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)


def fig_b_alpha(root, out):
    runs = sorted(glob.glob(os.path.join(root, "results/v1/result_B/e4_*_alpha")))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for r in runs:
        obj = os.path.basename(r).split("_")[1]
        rows = [json.loads(l) for l in open(os.path.join(r, "log.jsonl"))]
        pts = [(x["step"], x["alpha_mean"]) for x in rows if "alpha_mean" in x]
        if pts:
            s, a = zip(*pts)
            ax.plot(s, a, "-o", ms=3, color=COLORS[obj], label=LABEL[obj])
    ax.set_xlabel("step")
    ax.set_ylabel(r"adaptive $\alpha$ (mean over matrices)")
    ax.set_ylim(0, 0.45)
    ax.set_title("M2 gate closes the spectrum in all three paradigms", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figB_alpha.pdf"))
    plt.close(fig)


def _load_v2_phase1(root, optimizer="adamw"):
    d = {}
    for obj in ("sft", "opd", "rlvr"):
        p = os.path.join(root, f"results/v2/result_A/phase1_{obj}_{optimizer}/metrics.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["sig_over_noise"] = df.sig_energy_total / df.noise_energy
            d[obj] = df
    return d


def fig_c_signal_noise(root, out):
    for opt in ("adamw", "muon"):
        d = _load_v2_phase1(root, opt)
        if not d:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3))
        for obj, df in d.items():
            g = df.groupby("step").sig_over_noise
            x = g.median().index.values
            _band(axes[0], x, g.quantile(0.25).values, g.median().values,
                  g.quantile(0.75).values, COLORS[obj], LABEL[obj])
            g2 = df.groupby("step").apply(lambda t: (t.sig_energy_total > 0).mean())
            axes[1].plot(g2.index.values, g2.values, color=COLORS[obj], label=LABEL[obj], lw=1.6)
        axes[0].axhline(0, color="k", lw=0.6)
        axes[0].set_ylabel("cross-sample signal energy / noise energy")
        axes[0].set_title("median over matrices, IQR band", fontsize=9)
        axes[1].set_ylabel("fraction of matrices with signal > 0")
        axes[1].set_ylim(0, 1)
        axes[1].axhline(0.5, color="k", lw=0.6, ls=":")
        for ax in axes:
            ax.set_xlabel("step")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Qwen3.5-0.8B, {opt.upper()}, B=8 microbatches split 4|4", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"figC_signal_noise_{opt}.pdf"))
        plt.close(fig)


def fig_d_spearman(root, out):
    rows = []
    for opt in ("adamw", "muon"):
        for obj, df in _load_v2_phase1(root, opt).items():
            rows.append(dict(run=f"{LABEL[obj]}\n{opt}",
                             v1=df.spearman_absC_snr.median(),
                             v1_lo=df.spearman_absC_snr.quantile(0.25), v1_hi=df.spearman_absC_snr.quantile(0.75),
                             cross=df.spearman_absC_snr_cross.median(),
                             cross_lo=df.spearman_absC_snr_cross.quantile(0.25), cross_hi=df.spearman_absC_snr_cross.quantile(0.75),
                             split=df.spearman_absC_split.median(),
                             split_lo=df.spearman_absC_split.quantile(0.25), split_hi=df.spearman_absC_split.quantile(0.75)))
    if not rows:
        return
    t = pd.DataFrame(rows)
    x = np.arange(len(t))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.4, 3))
    for i, (k, lab, c) in enumerate([("v1", "v1: Spearman(|C|, SNR), same samples", "0.55"),
                                     ("cross", "split-half: |C| (A) vs SNR (B)", "tab:orange"),
                                     ("split", "split-half: |C| (A) vs |C| (B)", "tab:purple")]):
        err = np.vstack([t[k] - t[f"{k}_lo"], t[f"{k}_hi"] - t[k]])
        ax.bar(x + (i - 1) * w, t[k], w, yerr=err, color=c, label=lab, capsize=2, error_kw=dict(lw=0.8))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(t.run, fontsize=8)
    ax.set_ylabel("rank correlation (median, IQR)")
    ax.set_title("v1 H4 statistic is a same-sample coupling artefact", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figD_spearman_v1_vs_split.pdf"))
    plt.close(fig)


def fig_a_equal_norm(root, out):
    """Equal-norm H5 (P3). GSM8K/MMLU per mode, mean over seeds with per-seed dots."""
    runs = glob.glob(os.path.join(root, "results/v2/result_B/e4a_*"))
    if not runs:
        return
    recs = []
    for r in runs:
        name = os.path.basename(r)
        parts = name.split("_")
        obj = parts[1]
        seed = parts[-1]
        mode = "_".join(parts[2:-1])
        g = os.path.join(r, "eval_gsm8k.json")
        m = os.path.join(r, "eval_mmlu.json")
        recs.append(dict(obj=obj, mode=mode, seed=seed,
                         gsm8k=json.load(open(g))["pass@1"] if os.path.exists(g) else np.nan,
                         mmlu=json.load(open(m))["mmlu"] if os.path.exists(m) else np.nan))
    # seed-0 full baselines are the v1 phase2_*_none runs
    for obj in ("rlvr", "sft"):
        p = os.path.join(root, f"results/v1/result_B/phase2_{obj}_none")
        g, m = os.path.join(p, "eval_gsm8k.json"), os.path.join(p, "eval_mmlu.json")
        if os.path.exists(g):
            recs.append(dict(obj=obj, mode="full", seed="s0",
                             gsm8k=json.load(open(g))["pass@1"],
                             mmlu=json.load(open(m))["mmlu"] if os.path.exists(m) else np.nan))
    df = pd.DataFrame(recs)
    order = ["full", "frame_matched", "spectrum_matched", "random_ext", "exact_iso"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3), sharey=False)
    for ax, metric in zip(axes, ("gsm8k", "mmlu")):
        for j, obj in enumerate(("rlvr", "sft")):
            sub = df[df.obj == obj]
            modes = [m for m in order if m in set(sub["mode"])]
            xs = np.arange(len(modes)) + (j - 0.5) * 0.35
            means = [sub[sub["mode"] == m][metric].mean() for m in modes]
            ax.bar(xs, means, 0.33, color=COLORS[obj], alpha=0.75, label=LABEL[obj])
            for xi, m in zip(xs, modes):
                ys = sub[sub["mode"] == m][metric].values
                ax.plot([xi] * len(ys), ys, "k.", ms=4)
            ax.set_xticks(np.arange(len(modes)))
            ax.set_xticklabels(modes, rotation=20, fontsize=8)
        se = 1.96 * np.sqrt(0.5 * 0.5 / (500 if metric == "gsm8k" else 1000))
        ax.set_ylabel(f"{metric} ({'500 q, greedy' if metric == 'gsm8k' else '1000 q'}); 95% CI ±{se:.3f}")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Equal-Frobenius-step interventions, 0.8B, 300 steps", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figA_equal_norm_h5.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "figs", "v2")
    os.makedirs(out, exist_ok=True)
    fig_a_equal_norm(a.root, out)
    fig_b_alpha(a.root, out)
    fig_c_signal_noise(a.root, out)
    fig_d_spearman(a.root, out)
    print("figures in", out, sorted(os.listdir(out)))
