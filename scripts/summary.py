"""Print a compact text summary of one run for copy-paste delivery.
Usage: python scripts/summary.py $RUNS/<run> [...more runs]"""

import glob
import json
import os
import sys


def summarize(run):
    out = {"run": os.path.basename(run.rstrip("/"))}
    log = os.path.join(run, "log.jsonl")
    if os.path.exists(log):
        rows = [json.loads(l) for l in open(log)]
        steps = [r for r in rows if "loss" in r]
        out["steps"] = len(rows)
        out["loss_last"] = round(steps[-1]["loss"], 4) if steps else None
        rw = [(r["step"], round(r["reward_mean"], 3)) for r in steps
              if "reward_mean" in r]
        out["reward_curve"] = rw[::max(1, len(rw) // 20)]  # ~20 points
        al = [(r["step"], round(r["alpha_mean"], 3)) for r in steps
              if "alpha_mean" in r]
        if al:
            out["alpha_curve"] = al[::max(1, len(al) // 20)]
    for f in glob.glob(os.path.join(run, "eval*.json")):
        out[os.path.basename(f)] = json.load(open(f))
    m = os.path.join(run, "metrics.csv")
    if os.path.exists(m):
        import pandas as pd
        df = pd.read_csv(m)
        med = {}
        for c in ("G_r_spectrum", "H_r_spectrum", "r_null",
                  "spearman_absC_snr", "snr_diag_median"):
            if c in df:
                med[c] = float(df[c].median())
        if "r_null" in df.columns:
            med["R_enrich"] = float((df.G_r_spectrum / df.r_null).median())
        out["metrics_median"] = {k: round(v, 6) for k, v in med.items()}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    for r in sys.argv[1:]:
        summarize(r)
