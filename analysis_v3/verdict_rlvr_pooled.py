"""Pooled A+B RLVR verdict (2026-09-16): A's runs (MIG 3g.40gb) and B's runs (A100-80G) are treated as
independent samples (same-seed runs do not reproduce across hardware: median |dGSM8K| 0.010, max 0.044, B README §2).
t_binom: binomial SE pooled over runs; t_emp: empirical between-run SE (needs >=2 runs per arm).
Usage: python analysis_v3/verdict_rlvr_pooled.py results/v3/result_A results/v3/result_B
"""
import glob, json, os, re, math, sys, collections, statistics as st
def se(p, n): return math.sqrt(p * (1 - p) / n)
runs = collections.defaultdict(list)
for root in sys.argv[1:]:
    src = os.path.basename(root.rstrip('/'))[-1]
    for d in sorted(glob.glob(root + "/v3_rlvr_*")):
        if not os.path.exists(d + "/eval_gsm8k.json"): continue
        g = json.load(open(d + "/eval_gsm8k.json"))["pass@1"]; m = json.load(open(d + "/eval_mmlu.json"))["mmlu"]
        runs[re.sub(r"_s\d+$", "", os.path.basename(d))].append((src, g, m))
agg = {}
print(f"{'config':40s} {'n':>2s} {'GSM8K (A | B)':>40s} {'mean':>6s} {'MMLU':>6s}")
for cfg, rs in sorted(runs.items()):
    G = sum(r[1] for r in rs) / len(rs); M = sum(r[2] for r in rs) / len(rs); agg[cfg] = (G, M, len(rs))
    a = '/'.join(f"{r[1]:.3f}" for r in rs if r[0] == 'A'); b = '/'.join(f"{r[1]:.3f}" for r in rs if r[0] == 'B')
    print(f"{cfg:40s} {len(rs):2d} {a + ' | ' + b:>40s} {G:6.3f} {M:6.3f}")
def diff(a, b, label):
    if a not in agg or b not in agg: return
    (Ga, Ma, na), (Gb, Mb, nb) = agg[a], agg[b]
    sg = math.sqrt(se(Ga, 500 * na) ** 2 + se(Gb, 500 * nb) ** 2); sm = math.sqrt(se(Ma, 1000 * na) ** 2 + se(Mb, 1000 * nb) ** 2)
    ga = [r[1] for r in runs[a]]; gb = [r[1] for r in runs[b]]
    t_emp = float('nan')
    if len(ga) > 1 and len(gb) > 1:
        t_emp = (Ga - Gb) / math.sqrt(st.stdev(ga) ** 2 / len(ga) + st.stdev(gb) ** 2 / len(gb))
    print(f"  {label}: GSM8K {Ga - Gb:+.3f} (t_binom={(Ga - Gb) / sg:+.1f}, t_emp={t_emp:+.1f})  MMLU {Ma - Mb:+.3f} (t={(Ma - Mb) / sm:+.1f})")
print("\nPooled A+B (independent samples):")
diff("v3_rlvr_full_lr0.1", "v3_rlvr_frame_matched_lr0.1", "H-2 full - frame_matched")
diff("v3_rlvr_exact_iso_lr0.1", "v3_rlvr_full_lr0.1", "[void arm] exact_iso - full")
diff("v3_rlvr_spectrum_matched_lr0.1", "v3_rlvr_random_ext_lr0.1", "H-1 spectrum - random @lr*")
diff("v3_rlvr_spectrum_matched_lr1", "v3_rlvr_random_ext_lr1", "H-1 spectrum - random @lr x1")
diff("v3_rlvr_spectrum_matched_lr0.1", "v3_rlvr_full_lr0.1", "frontier spectrum - full @lr*")
diff("v3_rlvr_random_ext_lr0.1", "v3_rlvr_full_lr0.1", "frontier random - full @lr*")
for dct in ("spectrum_matched", "random_ext"):
    for s in ("srel3", "srel10"):
        diff(f"v3_rlvr_{dct}_lr0.1_{s}", f"v3_rlvr_{dct}_lr0.1", f"H-3 {dct} {s} - s_rel1")
diff("v3_rlvr_full_lr0.33", "v3_rlvr_full_lr0.1", "gate lr x1/3 - lr*")
diff("v3_rlvr_full_lr0.033", "v3_rlvr_full_lr0.1", "gate lr x1/30 - lr*")
