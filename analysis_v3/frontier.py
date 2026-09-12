"""A4: (GSM8K, MMLU) frontier plots for v3 (doc §6 Fig.1 / Fig.2).

Reads run directories (results/v3/result_B/<run>/ and results/v2/result_B/e4a_*),
groups seeds by the run name with `_s<seed>` stripped, and draws per objective:

  * dense line: full at each lr (from args.json `lr`), ordered by lr
  * spectral line / random line: spectrum_matched / random_ext at each s_rel
    (args.json `intervention_scale`, missing -> 1.0), at the run's lr
  * points: frame_matched, exact_iso
  * error bars: binomial SE pooled over seeds

Also emits a CSV of the aggregated points and, if eval_step*.json files exist,
a second figure with the (GSM8K, MMLU) trajectories every 50 steps.

Usage:
  python analysis_v3/frontier.py results/v3/result_B results/v2/result_B --out figs/v3
"""

import argparse
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {"none": ("dense (full)", "k", "o"),
         "spectrum_matched": ("spectral dict", "C0", "s"),
         "random_ext": ("random dict", "C1", "^"),
         "frame_matched": ("frame_matched", "C2", "D"),
         "exact_iso": ("exact_iso", "C3", "v")}


def read_run(d):
    a = json.load(open(os.path.join(d, "args.json")))
    ev = {}
    for name in ("gsm8k", "mmlu"):
        f = os.path.join(d, f"eval_{name}.json")
        if os.path.exists(f):
            j = json.load(open(f))
            ev[name] = j.get("pass@1", j.get("mmlu"))
    traj = []
    for f in sorted(glob.glob(os.path.join(d, "eval_step*_gsm8k.json"))):
        step = int(re.search(r"eval_step(\d+)_", f).group(1))
        mf = f.replace("_gsm8k.json", "_mmlu.json")
        if os.path.exists(mf):
            traj.append((step, json.load(open(f))["pass@1"], json.load(open(mf))["mmlu"]))
    return dict(run=os.path.basename(d), cfg=re.sub(r"_s\d+$", "", os.path.basename(d)),
                objective=a["objective"], intervention=a.get("intervention", "none"),
                lr=a["lr"], s_rel=a.get("intervention_scale", 1.0), seed=a.get("seed"),
                gsm8k=ev.get("gsm8k"), mmlu=ev.get("mmlu"), traj=traj)


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / n) if p is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="figs/v3")
    ap.add_argument("--n-gsm8k", type=int, default=500)
    ap.add_argument("--n-mmlu", type=int, default=1000)
    ap.add_argument("--base-gsm8k", type=float, default=0.546)
    ap.add_argument("--base-mmlu", type=float, default=0.483)
    ap.add_argument("--exclude-prefix", default="phase2_,e1_,e2_,e3_,e4_,p1_",
                    help="comma-separated run-name prefixes to skip (v1 bf16-master runs)")
    a = ap.parse_args()
    skip = tuple(x for x in a.exclude_prefix.split(",") if x)
    os.makedirs(a.out, exist_ok=True)

    runs = []
    for root in a.roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.basename(d).startswith(skip):
                continue
            if os.path.isfile(os.path.join(d, "args.json")) and os.path.isfile(os.path.join(d, "eval_gsm8k.json")):
                runs.append(read_run(d))
    df = pd.DataFrame(runs)
    df = df[df.gsm8k.notna() & df.mmlu.notna()]
    agg = (df.groupby(["objective", "intervention", "lr", "s_rel", "cfg"])
             .agg(n=("seed", "count"), gsm8k=("gsm8k", "mean"), mmlu=("mmlu", "mean"),
                  gsm8k_spread=("gsm8k", lambda x: x.max() - x.min())).reset_index())
    agg["gsm8k_se"] = [se(p, a.n_gsm8k * n) for p, n in zip(agg.gsm8k, agg.n)]
    agg["mmlu_se"] = [se(p, a.n_mmlu * n) for p, n in zip(agg.mmlu, agg.n)]
    agg.to_csv(os.path.join(a.out, "frontier_points.csv"), index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.3g}", "display.width", 200):
        print(agg.to_string(index=False))

    objs = sorted(agg.objective.unique())
    fig, axes = plt.subplots(1, len(objs), figsize=(5.2 * len(objs), 4.4), squeeze=False)
    for ax, obj in zip(axes[0], objs):
        g = agg[agg.objective == obj]
        ax.axvline(a.base_gsm8k, color="gray", lw=0.8, ls=":")
        ax.axhline(a.base_mmlu, color="gray", lw=0.8, ls=":")
        for mode, (label, color, marker) in STYLE.items():
            h = g[g.intervention == mode]
            if h.empty:
                continue
            h = h.sort_values(["lr", "s_rel"])
            ax.errorbar(h.gsm8k, h.mmlu, xerr=h.gsm8k_se, yerr=h.mmlu_se, fmt=marker,
                        color=color, label=label, capsize=2, ms=6)
            if mode in ("none", "spectrum_matched", "random_ext") and len(h) > 1:
                ax.plot(h.gsm8k, h.mmlu, color=color, lw=1, alpha=0.6)
            for _, r in h.iterrows():
                tag = (f"lr{r.lr:.0e}" if mode == "none" else f"s{r.s_rel:g}")
                ax.annotate(tag, (r.gsm8k, r.mmlu), fontsize=6, xytext=(3, 3),
                            textcoords="offset points")
        ax.set_title(f"{obj.upper()}: (GSM8K, MMLU) at 300 steps")
        ax.set_xlabel("GSM8K pass@1")
        ax.set_ylabel("MMLU")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig1_frontier.pdf"))
    print("wrote", os.path.join(a.out, "fig1_frontier.pdf"))

    # trajectories every 50 steps (if available)
    tr = df[df.traj.map(len) > 0]
    if not tr.empty:
        fig, axes = plt.subplots(1, len(objs), figsize=(5.2 * len(objs), 4.4), squeeze=False)
        for ax, obj in zip(axes[0], objs):
            for _, r in tr[tr.objective == obj].iterrows():
                label, color, marker = STYLE.get(r.intervention, (r.intervention, "C4", "x"))
                xs = [t[1] for t in r.traj]
                ys = [t[2] for t in r.traj]
                ax.plot(xs, ys, marker=marker, color=color, lw=0.8, alpha=0.7, ms=3,
                        label=f"{r.cfg}")
            ax.set_title(f"{obj.upper()}: trajectory (every 50 steps)")
            ax.set_xlabel("GSM8K pass@1")
            ax.set_ylabel("MMLU")
            ax.legend(fontsize=5)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig1b_trajectories.pdf"))
        print("wrote", os.path.join(a.out, "fig1b_trajectories.pdf"))


if __name__ == "__main__":
    main()
