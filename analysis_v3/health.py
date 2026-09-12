"""Batch-1 gate (v3 doc §4 H-0): is a dense training run "healthy"?

Reads log.jsonl + eval json files of run directories, groups seeds by the run
name with the trailing `_s<seed>` removed, and reports per configuration:

  reward_peak      max over 50-step windows of the mean training reward (RLVR)
  reward_last100   mean reward over the last 100 steps (RLVR)
  decline          reward_peak - reward_last100            (criterion <= 0.03)
  mmlu             final MMLU, mean over seeds             (criterion >= base - 0.03)
  gsm8k            final GSM8K, mean over seeds
  gsm8k_spread     max - min GSM8K over seeds              (criterion <= 0.04, needs >= 2 seeds)
  trunc_frac_last  mean truncation fraction over the last 50 steps (if logged)
  verdict          HEALTHY if all applicable criteria pass, else the failing ones

lr* = the largest lr among HEALTHY dense configurations (decided by A, printed here).

Usage:
  python analysis_v3/health.py results/v3/result_B/e4a_rlvr_full_lr0.1_s* [more dirs...] \
      --base-gsm8k 0.546 --base-mmlu 0.483
"""

import argparse
import glob
import json
import os
import re

import pandas as pd


def load_run(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    log = pd.DataFrame([r for r in recs if "loss" in r])
    ev = {}
    for name in ("gsm8k", "mmlu"):
        f = os.path.join(d, f"eval_{name}.json")
        if os.path.exists(f):
            j = json.load(open(f))
            ev[name] = j.get("pass@1", j.get("mmlu"))
    traj = []
    for f in sorted(glob.glob(os.path.join(d, "eval_step*_gsm8k.json"))):
        step = int(re.search(r"eval_step(\d+)_", f).group(1))
        g = json.load(open(f)).get("pass@1")
        mf = f.replace("_gsm8k.json", "_mmlu.json")
        m = json.load(open(mf)).get("mmlu") if os.path.exists(mf) else None
        traj.append((step, g, m))
    return log, ev, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--base-gsm8k", type=float, default=0.546)
    ap.add_argument("--base-mmlu", type=float, default=0.483)
    ap.add_argument("--max-decline", type=float, default=0.03)
    ap.add_argument("--mmlu-tol", type=float, default=0.03)
    ap.add_argument("--max-spread", type=float, default=0.04)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    per_run = []
    for d in a.runs:
        d = d.rstrip("/")
        log, ev, traj = load_run(d)
        name = os.path.basename(d)
        cfg = re.sub(r"_s\d+$", "", name)
        row = dict(run=name, cfg=cfg, steps=int(log.step.max()),
                   gsm8k=ev.get("gsm8k"), mmlu=ev.get("mmlu"))
        if "reward_mean" in log:
            w = log.groupby((log.step - 1) // 50).reward_mean.mean()
            row["reward_peak"] = float(w.max())
            row["reward_last100"] = float(log[log.step > log.step.max() - 100].reward_mean.mean())
            row["decline"] = row["reward_peak"] - row["reward_last100"]
        if "trunc_frac" in log:
            row["trunc_frac_last"] = float(log[log.step > log.step.max() - 50].trunc_frac.mean())
        if "n_groups_kept" in log:
            row["groups_kept_last"] = float(log[log.step > log.step.max() - 50].n_groups_kept.mean())
        row["traj"] = traj
        per_run.append(row)
    df = pd.DataFrame(per_run)

    out = []
    for cfg, g in df.groupby("cfg"):
        r = dict(cfg=cfg, n_seeds=len(g), gsm8k=g.gsm8k.mean(), mmlu=g.mmlu.mean(),
                 gsm8k_spread=(g.gsm8k.max() - g.gsm8k.min()) if len(g) > 1 else None)
        fails = []
        if "decline" in g:
            r["reward_peak"] = g.reward_peak.mean()
            r["reward_last100"] = g.reward_last100.mean()
            r["decline"] = g.decline.mean()
            if r["decline"] > a.max_decline:
                fails.append(f"decline {r['decline']:.3f}")
        if pd.notna(r["mmlu"]) and r["mmlu"] < a.base_mmlu - a.mmlu_tol:
            fails.append(f"mmlu {r['mmlu']:.3f}")
        if r["gsm8k_spread"] is not None and r["gsm8k_spread"] > a.max_spread:
            fails.append(f"spread {r['gsm8k_spread']:.3f}")
        if "trunc_frac_last" in g:
            r["trunc_frac_last"] = g.trunc_frac_last.mean()
        if "groups_kept_last" in g:
            r["groups_kept_last"] = g.groups_kept_last.mean()
        r["verdict"] = "HEALTHY" if not fails else "FAIL: " + "; ".join(fails)
        out.append(r)
    res = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    healthy = res[res.verdict == "HEALTHY"]
    print("\nHEALTHY configs:", healthy.cfg.tolist() if len(healthy) else "none")
    if a.out:
        res.to_csv(a.out, index=False)
        df.drop(columns=["traj"]).to_csv(a.out.replace(".csv", "_runs.csv"), index=False)


if __name__ == "__main__":
    main()
