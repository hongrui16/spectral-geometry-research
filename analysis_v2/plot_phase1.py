"""Phase 1 figures from per-run metrics.csv files.

  python analysis/plot_phase1.py --runs runs/phase1_sft_adamw runs/phase1_opd_adamw \
      runs/phase1_rlvr_adamw --labels SFT OPD RLVR --out figs/

Fig.2  R_spectrum(G) vs step, per paradigm (median over matrices + IQR band)
Fig.3  R_spectrum(G) vs R_spectrum(H) scatter
Fig.4  |rho_sigma| vs |rho_frame| histograms (runs with rho_hist.pt)
Fig.5  spearman(|C|, SNR) distributions per paradigm
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

COLORS = {"SFT": "tab:blue", "OPD": "tab:green", "RLVR": "tab:red"}


SHAPE_MAP = {  # qwen3.5-0.8B fallback for CSVs written before r_null existed
    "q_proj": 4096, "k_proj": 1024, "v_proj": 1024, "o_proj": 2048,
    "gate_proj": 3584, "up_proj": 3584, "down_proj": 3584,
    "in_proj_qkv": 6144, "out_proj": 2048,
}


def load(run):
    df = pd.read_csv(os.path.join(run, "metrics.csv"))
    if "r_null" not in df.columns:
        dim = df.matrix.str.extract(r"(\w+)\.weight")[0].map(SHAPE_MAP)
        df["r_null"] = 1.0 / dim
    # enrichment of diagonal energy over the random-G null (shape-normalized)
    df["G_r_enrich"] = df["G_r_spectrum"] / df["r_null"]
    df["H_r_enrich"] = df["H_r_spectrum"] / df["r_null"]
    return df


def fig2(runs, labels, out):
    plt.figure(figsize=(6, 4))
    for run, lab in zip(runs, labels):
        df = load(run)
        g = df.groupby("step")["G_r_enrich"]
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        c = COLORS.get(lab.split("-")[0], None)
        plt.plot(med.index, med.values, label=lab, color=c)
        plt.fill_between(med.index, lo.values, hi.values, alpha=0.2, color=c)
    plt.axhline(1.0, color="k", lw=0.6, ls=":", label="random-G null")
    plt.xlabel("step")
    plt.ylabel(r"$R_{\rm spectrum}(G)\,/\,R_{\rm null}$")
    plt.yscale("log"), plt.legend(), plt.tight_layout()
    plt.savefig(os.path.join(out, "fig2_rspectrum.pdf"))
    plt.close()


def fig3(runs, labels, out):
    plt.figure(figsize=(5, 5))
    for run, lab in zip(runs, labels):
        df = load(run)
        plt.scatter(df["G_r_enrich"], df["H_r_enrich"], s=6, alpha=0.4,
                    label=lab, color=COLORS.get(lab.split("-")[0], None))
    lim = plt.gca().get_xlim()
    plt.plot(lim, lim, "k--", lw=0.5)
    plt.xscale("log"), plt.yscale("log")
    plt.xlabel(r"$R_{\rm spectrum}(G)/R_{\rm null}$")
    plt.ylabel(r"$R_{\rm spectrum}(H)/R_{\rm null}$")
    plt.legend(), plt.tight_layout()
    plt.savefig(os.path.join(out, "fig3_G_vs_H.pdf"))
    plt.close()


def fig4(runs, labels, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.2),
                             squeeze=False)
    for ax, run, lab in zip(axes[0], runs, labels):
        p = os.path.join(run, "rho_hist.pt")
        if not os.path.exists(p):
            ax.set_title(f"{lab}: no rho data")
            continue
        store = torch.load(p, weights_only=False)
        rs = torch.cat([r["rho_sigma"] for r in store]).abs().numpy()
        rf = torch.cat([r["rho_frame"] for r in store]).abs().numpy()
        ax.hist(rs, bins=50, alpha=0.6, density=True, label=r"$|\rho^\Sigma|$")
        ax.hist(rf, bins=50, alpha=0.6, density=True, label=r"$|\rho^{frame}|$")
        ax.set_title(lab), ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig4_rho.pdf"))
    plt.close()


def fig5(runs, labels, out):
    plt.figure(figsize=(6, 4))
    data, ticks = [], []
    for run, lab in zip(runs, labels):
        df = load(run)
        if "spearman_absC_snr" in df:
            data.append(df["spearman_absC_snr"].dropna().values)
            ticks.append(lab)
    plt.boxplot(data, labels=ticks)
    plt.ylabel(r"Spearman$(|\bar C_{ij}|, \widehat{SNR}_{ij})$")
    plt.axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig5_absC_vs_snr.pdf"))
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn in (fig2, fig3, fig4, fig5):
        try:
            fn(a.runs, a.labels, a.out)
            print(f"{fn.__name__} ok")
        except Exception as e:
            print(f"{fn.__name__} FAILED: {e}")
