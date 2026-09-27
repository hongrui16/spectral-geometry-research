"""v5 health gate (docs/unified_paper_document_v5.md part 3 §0, gate G0).

"Healthy" means the run trained and its measurements are valid. Forgetting is
the OUTCOME and is reported, never gated on (v3/v4 gated on MMLU, which mixed
the outcome into the selection of lr*).

Per configuration (seeds grouped by stripping a trailing _s<seed>):
  gsm8k / gain      final strict pass@1 and gain over base      (criterion: gain > --min-gain)
  eval_trunc        GSM8K eval truncation fraction              (criterion <= --max-eval-trunc)
  trunc_last50      training rollout truncation, last 50 steps  (RLVR; criterion <= --max-train-trunc)
  decline           reward peak (50-step windows) - last-100 mean (RLVR; criterion <= --max-decline)
  letter_mass       MMLU letter probability mass after the prefill (criterion >= --mass-frac x base)
  visible           fraction of decoder weights whose bf16 value changed (bf16_visible.json,
                    analysis_v5/bf16_visible.py; criterion >= --min-visible when present)
  retention / drop  retention mean and drop vs base            (reported only)
  gsm8k_sd          between-seed SD                            (reported only)

Usage:
  python analysis_v5/health.py $RUNS/v5_rlvr_dense_lr*_s* --base results/v5/result_A/base
"""

import argparse
import json
import os
import re
import statistics

import pandas as pd


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def load_run(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    log = pd.DataFrame([r for r in recs if "loss" in r])
    row = {"run": os.path.basename(d), "cfg": re.sub(r"_s\d+$", "", os.path.basename(d)),
           "steps": int(log.step.max())}
    g = jload(os.path.join(d, "eval_gsm8k.json"))
    if g:
        row.update(gsm8k=g["pass@1"], gsm8k_flex=g["pass@1_flex"], eval_trunc=g["trunc_frac"])
    r = jload(os.path.join(d, "eval_retention.json"))
    if r:
        row["retention"] = r.get("retention")
        row["letter_mass"] = r["tasks"].get("mmlu", {}).get("letter_mass")
        row["mmlu_gen"] = r["tasks"].get("mmlu_gen", {}).get("acc")
    v = jload(os.path.join(d, "bf16_visible.json"))
    if v:
        row["visible"] = v["visible_frac"]
    if "reward_mean" in log:
        w = log.groupby((log.step - 1) // 50).reward_mean.mean()
        row["reward_peak"] = float(w.max())
        row["reward_last100"] = float(log[log.step > log.step.max() - 100].reward_mean.mean())
        row["decline"] = row["reward_peak"] - row["reward_last100"]
    if "trunc_frac" in log:
        row["trunc_last50"] = float(log[log.step > log.step.max() - 50].trunc_frac.mean())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--base", required=True,
                    help="dir with base eval_gsm8k.json / eval_retention.json")
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--max-eval-trunc", type=float, default=0.05)
    ap.add_argument("--max-train-trunc", type=float, default=0.10)
    ap.add_argument("--max-decline", type=float, default=0.03)
    ap.add_argument("--mass-frac", type=float, default=0.8)
    ap.add_argument("--min-visible", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    bg = jload(os.path.join(a.base, "eval_gsm8k.json"))
    br = jload(os.path.join(a.base, "eval_retention.json"))
    base_g = bg["pass@1"]
    base_ret = br["retention"]
    base_mass = br["tasks"]["mmlu"]["letter_mass"]
    df = pd.DataFrame([load_run(d.rstrip("/")) for d in a.runs])

    out = []
    for cfg, g in df.groupby("cfg"):
        r = {"cfg": cfg, "n": len(g)}
        fails = []
        for k in ("gsm8k", "gsm8k_flex", "eval_trunc", "retention", "letter_mass", "mmlu_gen",
                  "visible", "decline", "trunc_last50"):
            if k in g and g[k].notna().any():
                r[k] = g[k].mean()
        if "gsm8k" in r:
            r["gain"] = r["gsm8k"] - base_g
            r["gsm8k_sd"] = statistics.stdev(g.gsm8k) if len(g) > 1 else None
            if r["gain"] <= a.min_gain:
                fails.append(f"gain {r['gain']:+.3f}")
            if r["eval_trunc"] > a.max_eval_trunc:
                fails.append(f"eval_trunc {r['eval_trunc']:.3f}")
        if "retention" in r:
            r["drop"] = base_ret - r["retention"]
        if "letter_mass" in r and r["letter_mass"] < a.mass_frac * base_mass:
            fails.append(f"letter_mass {r['letter_mass']:.3f}")
        if "visible" in r and r["visible"] < a.min_visible:
            fails.append(f"visible {r['visible']:.3f}")
        if "decline" in r and r["decline"] > a.max_decline:
            fails.append(f"decline {r['decline']:.3f}")
        if "trunc_last50" in r and r["trunc_last50"] > a.max_train_trunc:
            fails.append(f"train_trunc {r['trunc_last50']:.3f}")
        if "gsm8k" not in r:
            fails.append("no eval")
        r["verdict"] = "HEALTHY" if not fails else "FAIL: " + "; ".join(fails)
        out.append(r)
    res = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print(f"base: gsm8k {base_g:.3f}  retention {base_ret:.4f}  letter_mass {base_mass:.3f}")
    print(res.round(4).to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False)
        df.to_csv(a.out.replace(".csv", "_runs.csv"), index=False)


if __name__ == "__main__":
    main()
