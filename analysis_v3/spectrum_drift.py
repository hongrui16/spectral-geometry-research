"""Singular-value drift of checkpoints vs base (exact_iso is supposed to pin W's spectrum to base).
Usage: python analysis_v3/spectrum_drift.py --runs-root $RUNS --out results/v3/result_A/spectrum_drift.csv v3_rlvr_full_lr0.1_s0 ...
"""
import argparse, glob, os, csv, torch
from safetensors import safe_open
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--runs-root", required=True)
ap.add_argument("--base", default=None); ap.add_argument("--out", required=True); a = ap.parse_args()
base = a.base or glob.glob(os.path.expandvars("$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
dev = "cuda" if torch.cuda.is_available() else "cpu"
def load(d):
    out = {}
    for f in glob.glob(d + "/*.safetensors"):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if k.endswith(".weight") and any(t in k for t in ("self_attn", "mlp", "linear_attn")) and "norm" not in k:
                    kk = k.split("layers.")[-1]
                    if len(sf.get_slice(k).get_shape()) == 2: out[kk] = sf.get_tensor(k).float().to(dev)
    return out
B = load(base); SB = {n: torch.linalg.svdvals(w) for n, w in B.items()}
rows = []
for run in a.runs:
    C = load(os.path.join(a.runs_root, run, "ckpt_000300"))
    for n in sorted(set(B) & set(C)):
        sc = torch.linalg.svdvals(C[n]); sb = SB[n]
        rows.append(dict(run=run, matrix=n, rel_dW=((C[n] - B[n]).norm() / B[n].norm()).item(),
                         rel_dS=((sc - sb).norm() / sb.norm()).item(), max_dS_over_S1=((sc - sb).abs().max() / sb[0]).item()))
    import statistics as st
    r = [x for x in rows if x["run"] == run]
    print(f"{run:34s} n={len(r)} median rel_dW={st.median(x['rel_dW'] for x in r):.2e} median rel_dS={st.median(x['rel_dS'] for x in r):.2e} max rel_dS={max(x['rel_dS'] for x in r):.2e}", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("SPECTRUM_DRIFT_DONE")
