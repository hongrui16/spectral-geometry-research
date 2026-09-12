"""E-v3-0c: per-step functional step size of each intervention (v3 doc §5, batch 1).

Takes the captured update H (fp32 in v3 captures; bf16 in v2 smoke captures, fine
as a perturbation) of the tracked matrices at one save step from several runs that
share the prompt stream and seed (e.g. smoke_v3_sft_none and smoke_v3_sft_<mode>),
applies each run's H to the SAME base weights, and measures token-averaged forward
KL( p_base || p_base+H ) on the A5 reference set. This is the "functional step per
iteration" of full vs projected updates at matched Frobenius norm.

Approximation: H is applied to the base checkpoint rather than to that run's
step-t weights (the two differ by t small steps); this is a comparison across
modes at the same t, not an absolute number.

Usage (GPU):
  python analysis_v3/step_kl.py --runs $RUNS/smoke_v3_sft_none $RUNS/smoke_v3_sft_spectrum_matched ... \
      --step 10 --out results/v3/result_A/step_kl_smoke
"""

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_v3.kl_probe import build_reference, kl_tokens, logp_batch
from specgeom_v3.modeling import load_model


def load_capture(run, step):
    path = os.path.join(run, "capture", f"step_{step:06d}.pt")
    if not os.path.exists(path):
        cands = sorted(glob.glob(os.path.join(run, "capture", "step_*.pt")))
        raise FileNotFoundError(f"{path} (have: {[os.path.basename(c) for c in cands]})")
    return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(a.out, exist_ok=True)

    model, tok, _ = load_model(a.base, dtype=torch.float32, device="cuda", trainable=False)
    params = dict(model.named_parameters())
    input_ids, attn, ans = build_reference(tok, a.n_prompts, a.seed, "cuda", a.max_len)
    batches = [(input_ids[i:i + a.batch], attn[i:i + a.batch], ans[i:i + a.batch])
               for i in range(0, input_ids.shape[0], a.batch)]
    lp0 = [logp_batch(model, ii, am, False) for ii, am, _ in batches]

    rows = []
    t0 = time.time()
    for run in a.runs:
        rec = load_capture(run, a.step)
        args_json = os.path.join(run, "args.json")
        mode = json.load(open(args_json)).get("intervention", "?") if os.path.exists(args_json) else "?"
        saved = {}
        hnorm2, per_mat = 0.0, []
        with torch.no_grad():
            for n, d in rec["mats"].items():
                if "H" not in d:
                    continue
                H = d["H"].float().to("cuda")
                p = params[n]
                saved[n] = p.detach().clone()
                p.add_(H.to(p.dtype))
                h = H.norm().item()
                hnorm2 += h * h
                per_mat.append((n, h))
            num, den = 0.0, 0.0
            for (ii, am, an), l0 in zip(batches, lp0):
                l1 = logp_batch(model, ii, am, False)
                m = an[:, 1:].float().sum().item()
                num += kl_tokens(l0, l1, an).item() * m
                den += m
            for n, w in saved.items():
                params[n].copy_(w)
        kl = num / max(den, 1)
        rows.append(dict(run=os.path.basename(run.rstrip("/")), intervention=mode,
                         step=a.step, n_matrices=len(saved), h_fro_total=hnorm2 ** 0.5,
                         kl_step=kl, kl_per_fro2=kl / max(hnorm2, 1e-30)))
        print(f"{rows[-1]['run']:36s} {mode:18s} ||H||={hnorm2 ** 0.5:.3e} "
              f"KL={kl:.3e} nats/token ({time.time() - t0:.0f}s)", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "step_kl.csv"), index=False)
    json.dump({"args": vars(a), "elapsed_s": time.time() - t0},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(df.to_string())


if __name__ == "__main__":
    main()
