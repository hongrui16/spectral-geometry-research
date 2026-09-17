"""E-v4-4: per-matrix lr multipliers from the attribution (share_task / share_off).
factor_i = clip((r_i / median r)^alpha, lo, hi), then normalised so the mean factor over matrices is 1
(keeps the overall step comparable). --by-type pools the ratio by matrix type (q/k/v/o/gate/up/down/...)
for robustness against per-matrix noise. Usage:
  python analysis_v4/make_lr_scales.py results/v4/result_A/matrix_attrib/v3_sft_full_lr0.1_s0.csv --alpha 1 --by-type --out results/v4/result_A/lr_scales/sft_type_a1.json
"""
import argparse, csv, json, os, statistics as st, collections
ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--lo", type=float, default=0.25); ap.add_argument("--hi", type=float, default=4.0)
ap.add_argument("--by-type", action="store_true"); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = list(csv.DictReader(open(a.csv)))
def ratio(so, stk): return max(stk, 1e-4) / max(so, 1e-4)
if a.by_type:
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        t = r["matrix"].split(".")[-2]; acc[t][0] += float(r["share_off"]); acc[t][1] += float(r["share_task"])
    rt = {t: ratio(v[0], v[1]) for t, v in acc.items()}
    raw = {r["matrix"]: rt[r["matrix"].split(".")[-2]] for r in rows}
else:
    raw = {r["matrix"]: ratio(float(r["share_off"]), float(r["share_task"])) for r in rows}
med = st.median(raw.values())
fac = {n: min(max((v / med) ** a.alpha, a.lo), a.hi) for n, v in raw.items()}
mean = sum(fac.values()) / len(fac); fac = {n: v / mean for n, v in fac.items()}
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(fac, open(a.out, "w"), indent=0)
bt = collections.defaultdict(list)
for n, v in fac.items(): bt[n.split(".")[-2]].append(v)
print({t: round(sum(v)/len(v), 2) for t, v in sorted(bt.items(), key=lambda kv: -sum(kv[1])/len(kv[1]))})
