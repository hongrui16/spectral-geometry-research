"""v4 A3(b): per-matrix attribution of a finished run's functional change.
For each tracked 2-D weight of the checkpoint, revert THAT matrix to the base weights and measure
how much the off-task KL (MMLU-validation stems) and task KL (GSM8K solutions) to base DROP.
Output per matrix: kl_off_full, kl_off_without, share_off = 1 - without/full; same for task; ratio.
Usage: python analysis_v4/matrix_attrib.py --ckpt $RUNS/v3_sft_full_lr0.1_s0/ckpt_000300 --out results/v4/result_A/matrix_attrib/v3_sft_full_lr0.1_s0.csv
"""
import argparse, csv, os, re, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_v4.kl_probe import build_reference, build_offtask_reference, kl_tokens, logp_batch
from specgeom_v4.modeling import load_model
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B"); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n-prompts", type=int, default=64); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--max-len", type=int, default=512)
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
model, tok, _ = load_model(a.ckpt, trainable=False)
base, _, _ = load_model(a.base, trainable=False)
pat = re.compile(r"layers\.\d+\.(?:self_attn|mlp|linear_attn)\.\w+\.weight$")
names = [n for n, p in model.named_parameters() if p.dim() == 2 and pat.search(n) and "visual" not in n]
bw = {n: p.detach().clone() for n, p in base.named_parameters() if n in set(names)}
refs = {"task": build_reference(tok, a.n_prompts, 0, "cuda", a.max_len), "off": build_offtask_reference(tok, a.n_prompts, 0, "cuda", a.max_len)}
@torch.no_grad()
def logps(m, ref):
    ids, attn, ans = ref; out = []
    for i in range(0, ids.shape[0], a.batch): out.append(logp_batch(m, ids[i:i+a.batch], attn[i:i+a.batch], False))
    return torch.cat(out), ans
@torch.no_grad()
def kl_to_base(ref_key):
    l1, ans = logps(model, refs[ref_key]); l0, _ = logps(base, refs[ref_key]); return kl_tokens(l0, l1, ans).item()
full = {k: kl_to_base(k) for k in refs}; print("full", full, flush=True)
params = dict(model.named_parameters()); rows = []; t0 = time.time()
for n in names:
    p = params[n]; keep = p.detach().clone(); p.copy_(bw[n])
    w = {k: kl_to_base(k) for k in refs}; p.copy_(keep)
    rows.append(dict(matrix=n, m=p.shape[0], n=p.shape[1], kl_off_full=full["off"], kl_off_without=w["off"], share_off=1 - w["off"]/max(full["off"], 1e-12),
                     kl_task_full=full["task"], kl_task_without=w["task"], share_task=1 - w["task"]/max(full["task"], 1e-12)))
    print(f"{n} share_off={rows[-1]['share_off']:+.3f} share_task={rows[-1]['share_task']:+.3f} ({time.time()-t0:.0f}s)", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f: wtr = csv.DictWriter(f, fieldnames=list(rows[0])); wtr.writeheader(); wtr.writerows(rows)
print("MATRIX_ATTRIB_DONE")
