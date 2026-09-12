"""E-v3-0a: cumulative displacement and cumulative KL of finished runs (v3 doc §3.2, §5).

For each checkpoint directory: load it in fp32 next to the base model and report

  * per decoder matrix: ||W_ckpt - W_base||_F, ||W_base||_F, relative displacement
  * one row per run: token-averaged forward KL( p_base || p_ckpt ) on the A5 reference
    set (GSM8K test prompts + gold solutions, teacher-forced answer tokens), fp32
    forward, TF32 off.

Usage (GPU, ~1 min per checkpoint after the base is loaded):
  python analysis_v3/cum_kl.py --ckpts $RUNS/e4a_rlvr_full_s0/ckpt_000300 [...] \
      --out results/v3/result_B/cum_kl
  python analysis_v3/cum_kl.py --runs-root $RUNS --glob 'e4a_*' --step 300 --out ...
Outputs: <out>/cum_kl_matrices.csv, <out>/cum_kl_runs.csv, <out>/manifest.json
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_v3.kl_probe import build_reference, kl_tokens, logp_batch
from specgeom_v3.modeling import load_model

DEC = re.compile(r"layers\.\d+\.(?:self_attn|mlp|linear_attn)\.\w+\.weight$")


def decoder_mats(model):
    return {n: p for n, p in model.named_parameters()
            if p.dim() == 2 and DEC.search(n) and "visual" not in n
            and not n.startswith("mtp.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--glob", default="e4a_*")
    ap.add_argument("--step", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(a.out, exist_ok=True)

    ckpts = list(a.ckpts)
    if a.runs_root:
        for d in sorted(glob.glob(os.path.join(a.runs_root, a.glob))):
            ck = os.path.join(d, f"ckpt_{a.step:06d}")
            if os.path.isdir(ck):
                ckpts.append(ck)
    assert ckpts, "no checkpoints"

    base, tok, _ = load_model(a.base, dtype=torch.float32, device="cuda", trainable=False)
    W0 = {n: p.detach().clone() for n, p in decoder_mats(base).items()}
    input_ids, attn, ans = build_reference(tok, a.n_prompts, a.seed, "cuda", a.max_len)
    batches = [(input_ids[i:i + a.batch], attn[i:i + a.batch], ans[i:i + a.batch])
               for i in range(0, input_ids.shape[0], a.batch)]
    lp_base = [logp_batch(base, ii, am, False) for ii, am, _ in batches]
    n_tok = int(ans.sum())
    del base
    torch.cuda.empty_cache()

    mat_rows, run_rows = [], []
    t0 = time.time()
    for ck in ckpts:
        run = os.path.basename(os.path.dirname(ck.rstrip("/")))
        model, _, _ = load_model(ck, dtype=torch.float32, device="cuda", trainable=False)
        mats = decoder_mats(model)
        disp2 = 0.0
        for n, p in mats.items():
            d = (p.detach() - W0[n]).norm().item()
            w = W0[n].norm().item()
            disp2 += d * d
            mat_rows.append(dict(run=run, ckpt=ck, matrix=n, m=p.shape[0], n=p.shape[1],
                                 disp_fro=d, w0_fro=w, rel_disp=d / max(w, 1e-30)))
        num, den = 0.0, 0.0
        for (ii, am, an), lp0 in zip(batches, lp_base):
            lp1 = logp_batch(model, ii, am, False)
            m = an[:, 1:].float().sum().item()
            num += kl_tokens(lp0, lp1, an).item() * m
            den += m
        kl = num / max(den, 1)
        run_rows.append(dict(run=run, ckpt=ck, n_matrices=len(mats),
                             disp_fro_total=disp2 ** 0.5, kl_base_to_ckpt=kl))
        print(f"{run}: total disp {disp2 ** 0.5:.4e}, KL {kl:.4e} nats/token "
              f"({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    import pandas as pd
    pd.DataFrame(mat_rows).to_csv(os.path.join(a.out, "cum_kl_matrices.csv"), index=False)
    pd.DataFrame(run_rows).to_csv(os.path.join(a.out, "cum_kl_runs.csv"), index=False)
    json.dump({"args": vars(a), "n_answer_tokens": n_tok, "ckpts": ckpts,
               "elapsed_s": time.time() - t0},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
