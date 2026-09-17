"""Summarize matrix_attrib CSVs: concentration of off-task vs task load, per-type sums, and (if >=2 runs)
rank agreement between runs. Usage: python analysis_v4/attrib_summary.py results/v4/result_A/matrix_attrib/*.csv"""
import csv, sys, os, collections, math
def load(f):
    rows = list(csv.DictReader(open(f)))
    return {r["matrix"]: (float(r["share_off"]), float(r["share_task"])) for r in rows}, float(rows[0]["kl_off_full"]), float(rows[0]["kl_task_full"])
runs = {os.path.basename(f)[:-4]: load(f) for f in sys.argv[1:]}
for name, (d, ko, kt) in runs.items():
    off = sorted(((v[0], k) for k, v in d.items()), reverse=True); tot = sum(max(v, 0) for v, _ in off)
    c = 0; n50 = 0
    for v, _ in off:
        c += max(v, 0); n50 += 1
        if c >= 0.5 * tot: break
    bytype = collections.defaultdict(lambda: [0.0, 0.0])
    for k, (so, st) in d.items():
        t = k.split(".")[-2]; bytype[t][0] += so; bytype[t][1] += st
    print(f"\n== {name}: off-task KL {ko:.4f}, task KL {kt:.3f}; sum share_off {sum(v[0] for v in d.values()):.2f}, sum share_task {sum(v[1] for v in d.values()):.2f}; "
          f"{n50}/{len(d)} matrices carry 50% of off-task load")
    print("  top-8 off-task load (share_off | share_task | task/off):")
    for v, k in off[:8]:
        st = d[k][1]; print(f"    {k.replace('model.layers.', 'L').replace('.weight', ''):40s} {v:+.3f} | {st:+.3f} | {st / v if v > 1e-6 else float('nan'):.2f}")
    print("  by type (sum_off | sum_task | task/off):")
    for t, (so, st) in sorted(bytype.items(), key=lambda kv: -kv[1][0])[:7]:
        print(f"    {t:14s} {so:+.2f} | {st:+.2f} | {st / so if so > 1e-6 else float('nan'):.2f}")
names = list(runs)
if len(names) >= 2:
    from itertools import combinations
    def rank(d):
        order = sorted(d, key=lambda k: -d[k][0]); return {k: i for i, k in enumerate(order)}
    for a, b in combinations(names, 2):
        da, db = runs[a][0], runs[b][0]; common = [k for k in da if k in db]
        ra, rb = rank(da), rank(db); n = len(common)
        dsq = sum((ra[k] - rb[k]) ** 2 for k in common); rho = 1 - 6 * dsq / (n * (n * n - 1))
        top = set(sorted(common, key=lambda k: -da[k][0])[:30]) & set(sorted(common, key=lambda k: -db[k][0])[:30])
        print(f"\nrank agreement of off-task load {a} vs {b}: Spearman {rho:.2f}; top-30 overlap {len(top)}/30")
