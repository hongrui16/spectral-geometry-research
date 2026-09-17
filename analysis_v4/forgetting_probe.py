"""v4 §2.2: does an 8-step probe predict 300-step forgetting?
Joins, per configuration: probe value = step-8 (and step-4) KL / ||H||_F^2 on the OFF-TASK reference
(from results/v4/result_A/probe8_offtask_<obj>_step<k>/step_kl.csv, run names probe8_v4_<obj>_<tag>)
with the mean final MMLU drop (base - MMLU_300) and final GSM8K over all finished runs of that
configuration (results/v3/result_A, results/v3/result_B, results/v4/result_A; A and B pooled).
Prints the table and R^2 (log-log) overall and per objective. Usage:
  python analysis_v4/forgetting_probe.py --base-mmlu 0.483 --out results/v4/result_A/forgetting_probe.csv
"""
import argparse, csv, glob, json, math, os, re, collections
ap = argparse.ArgumentParser()
ap.add_argument("--probe-root", default="results/v4/result_A")
ap.add_argument("--run-roots", nargs="+", default=["results/v3/result_A", "results/v3/result_B", "results/v4/result_A"])
ap.add_argument("--base-mmlu", type=float, default=0.483)
ap.add_argument("--base-gsm8k", type=float, default=0.546)
ap.add_argument("--out", required=True)
a = ap.parse_args()
# map probe tag -> run-name regex (v3/v4 naming)
TAG2RE = {
 "full_lr0.033": r"v3_{o}_full_lr0\.033_s\d", "full_lr0.1": r"v3_{o}_full_lr0\.1_s\d", "full_lr0.33": r"v3_{o}_full_lr0\.33_s\d",
 "full_lr1": r"e4a_{o}_full_s\d|v4_{o}_full_lr1_s\d",
 "frame_lr0.1": r"v3_{o}_frame_matched_lr0\.1_s\d", "iso_lr0.1": r"v3_{o}_exact_iso_fixed_lr0\.1_s\d",
 "spec_lr0.1": r"v3_{o}_spectrum_matched_lr0\.1_s\d", "rand_lr0.1": r"v3_{o}_random_ext_lr0\.1_s\d",
 "frame_lr1": r"v4_{o}_frame_matched_lr1_s\d", "iso_lr1": r"v4_{o}_exact_iso_lr1_s\d",
 "spec_lr1": r"v3_{o}_spectrum_matched_lr1_s\d|e4a_{o}_spectrum_matched_s\d", "rand_lr1": r"v3_{o}_random_ext_lr1_s\d|e4a_{o}_random_ext_s\d",
 "spec_lr0.1_srel3": r"v3_{o}_spectrum_matched_lr0\.1_srel3_s\d", "rand_lr0.1_srel3": r"v3_{o}_random_ext_lr0\.1_srel3_s\d",
 "spec_lr0.1_srel10": r"v3_{o}_spectrum_matched_lr0\.1_srel10_s\d", "rand_lr0.1_srel10": r"v3_{o}_random_ext_lr0\.1_srel10_s\d",
}
def finals(obj, tag):
    pat = re.compile("^(" + TAG2RE[tag].format(o=obj) + ")$")
    g, m = [], []
    for root in a.run_roots:
        for d in glob.glob(root + "/*"):
            if pat.match(os.path.basename(d)) and os.path.exists(d + "/eval_mmlu.json"):
                g.append(json.load(open(d + "/eval_gsm8k.json"))["pass@1"]); m.append(json.load(open(d + "/eval_mmlu.json"))["mmlu"])
    return g, m
rows = []
for obj in ("sft", "rlvr"):
    probe = {}
    for step in (8, 4):
        f = f"{a.probe_root}/probe8_offtask_{obj}_step{step}/step_kl.csv"
        if not os.path.exists(f): continue
        for r in csv.DictReader(open(f)):
            tag = r["run"].replace(f"probe8_v4_{obj}_", "")
            probe.setdefault(tag, {})[step] = (float(r["kl_step"]), float(r["h_fro_total"]), float(r["kl_per_fro2"]))
    for tag, p in probe.items():
        g, m = finals(obj, tag)
        if not m: continue
        rows.append(dict(objective=obj, config=tag, n_runs=len(m), probe_kl8=p.get(8, (math.nan,)*3)[0], probe_kl4=p.get(4, (math.nan,)*3)[0],
                         probe_kl8_per_fro2=p.get(8, (math.nan,)*3)[2], mmlu_drop=a.base_mmlu - sum(m)/len(m), gsm8k=sum(g)/len(g)))
def r2(xs, ys):
    xs = [math.log(max(x, 1e-9)) for x in xs]; n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys)); sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
    return sxy*sxy/(sxx*syy) if sxx > 0 and syy > 0 else float("nan")
print(f"{'obj':5s} {'config':20s} {'n':>2s} {'probe KL8':>10s} {'KL8/|H|^2':>10s} {'MMLU drop':>9s} {'GSM8K':>6s}")
for r in rows: print(f"{r['objective']:5s} {r['config']:20s} {r['n_runs']:2d} {r['probe_kl8']:10.2e} {r['probe_kl8_per_fro2']:10.2f} {r['mmlu_drop']:9.3f} {r['gsm8k']:6.3f}")
for obj in ("sft", "rlvr", "all"):
    sel = [r for r in rows if obj == "all" or r["objective"] == obj]
    if len(sel) >= 3: print(f"R^2(log probe_kl8 -> MMLU drop) {obj}: {r2([r['probe_kl8'] for r in sel], [r['mmlu_drop'] for r in sel]):.2f}  n={len(sel)}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["objective"]); w.writeheader(); w.writerows(rows)
