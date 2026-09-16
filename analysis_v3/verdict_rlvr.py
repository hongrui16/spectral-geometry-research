"""v3 RLVR verdict table (A, 2026-09-16): final GSM8K/MMLU per run, per-config means,
pairwise diffs with binomial SE (n=500/1000, pooled over seeds), RLVR reward health.
Usage: python analysis_v3/verdict_rlvr.py results/v3/result_A
"""
import glob, json, math, os, sys, re
root = sys.argv[1]
runs = {}
for d in sorted(glob.glob(os.path.join(root, "v3_rlvr_*"))):
    if not os.path.exists(os.path.join(d, "eval_gsm8k.json")): continue
    a = json.load(open(os.path.join(d, "args.json")))
    g = json.load(open(os.path.join(d, "eval_gsm8k.json")))["pass@1"]
    m = json.load(open(os.path.join(d, "eval_mmlu.json")))["mmlu"]
    rows = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if l.strip()]
    tr = [r for r in rows if "reward_mean" in r]
    rw = [r["reward_mean"] for r in tr]
    peak = max(sum(rw[i:i+50])/50 for i in range(0, len(rw)-49)) if len(rw) >= 50 else None
    last100 = sum(rw[-100:])/len(rw[-100:])
    loss = sum(r["loss"] for r in tr[-20:])/20
    trunc = sum(r.get("trunc_frac", 0) for r in tr[-50:])/50
    cfg = re.sub(r"_s\d+$", "", os.path.basename(d))
    runs.setdefault(cfg, []).append(dict(run=os.path.basename(d), g=g, m=m, peak=peak,
                                         last100=last100, decline=peak-last100, loss=loss, trunc=trunc))
def se(p, n): return math.sqrt(p*(1-p)/n)
print(f"{'config':42s} {'GSM8K seeds':>18s} {'mean':>6s} {'MMLU seeds':>16s} {'mean':>6s} {'decline':>7s} {'loss':>6s} {'trunc':>5s}")
agg = {}
for cfg, rs in runs.items():
    G = sum(r["g"] for r in rs)/len(rs); M = sum(r["m"] for r in rs)/len(rs)
    agg[cfg] = (G, M, len(rs))
    gs = '/'.join('%.3f' % r['g'] for r in rs); ms = '/'.join('%.3f' % r['m'] for r in rs)
    print(f"{cfg:42s} {gs:>18s} {G:6.3f} {ms:>16s} {M:6.3f} "
          f"{max(r['decline'] for r in rs):7.3f} {sum(r['loss'] for r in rs)/len(rs):6.3f} {sum(r['trunc'] for r in rs)/len(rs):5.2f}")
def diff(a, b):
    if a not in agg or b not in agg: return
    (Ga, Ma, na), (Gb, Mb, nb) = agg[a], agg[b]
    sg = math.sqrt(se(Ga,500*na)**2 + se(Gb,500*nb)**2); sm = math.sqrt(se(Ma,1000*na)**2 + se(Mb,1000*nb)**2)
    print(f"  {a.replace('v3_rlvr_','')} - {b.replace('v3_rlvr_','')}: GSM8K {Ga-Gb:+.3f} (t={(Ga-Gb)/sg:+.1f})  MMLU {Ma-Mb:+.3f} (t={(Ma-Mb)/sm:+.1f})")
print("\nH-2 (mn-dof at lr*):"); diff("v3_rlvr_full_lr0.1","v3_rlvr_frame_matched_lr0.1"); diff("v3_rlvr_full_lr0.1","v3_rlvr_exact_iso_lr0.1"); diff("v3_rlvr_frame_matched_lr0.1","v3_rlvr_exact_iso_lr0.1")
print("H-1 (r-dof dicts at lr*):"); diff("v3_rlvr_spectrum_matched_lr0.1","v3_rlvr_random_ext_lr0.1")
print("H-1 (lr x1, seed 2 only):"); diff("v3_rlvr_spectrum_matched_lr1","v3_rlvr_random_ext_lr1")
print("frontier (r-dof vs dense lr*):"); diff("v3_rlvr_spectrum_matched_lr0.1","v3_rlvr_full_lr0.1"); diff("v3_rlvr_random_ext_lr0.1","v3_rlvr_full_lr0.1")
print("H-3 (s_rel sweep, seed 0):")
for dct in ("spectrum_matched","random_ext"):
    for s in ("srel3","srel10"):
        diff(f"v3_rlvr_{dct}_lr0.1_{s}", f"v3_rlvr_{dct}_lr0.1")
