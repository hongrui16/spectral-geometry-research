"""E0/E2 (docs/unified_paper_document_v2.md §11 E2, §12.4): finite-step KL probe.

For each tracked matrix W of a frozen checkpoint and each unit-Frobenius
direction d, perturb W <- W + eta * ||W||_F * d and measure the token-averaged
forward KL( p_theta || p_theta+delta ) on a fixed reference set (GSM8K test
prompts with their gold solutions, teacher-forced). Reports

  * KL vs eta for eta in --etas  -> log-log slope (should be 2 in the local
    quadratic regime; deviations flag where the local approximation fails)
  * KL / (eta ||W||_F)^2         -> quadratic sensitivity of that direction
                                    (a Fisher-like cost per unit step energy)

Directions (all unit Frobenius norm, in the SVD basis of W):
  spec_rand  : U diag(c) V^T, c ~ N(0,1)            (random spectral)
  spec_scale : U diag(sigma) V^T ~ W                (pure rescaling of W)
  frame_rot  : U (A Sigma - Sigma B) V^T, A,B random antisymmetric
               (first-order rotation of the left/right frames, spectrum fixed)
  gauss      : iid Gaussian                          (isotropic reference)

Usage (GPU):
  python analysis_v2/kl_probe.py --out results/v2/result_A/kl_probe_0.8b
Options: --model, --n-prompts, --batch, --etas, --max-matrices, --seed, --cpu
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom_v2.data import build_prompt, load_gsm8k
from specgeom_v2.instrument import select_tracked
from specgeom_v2.modeling import load_model

FAM = ("self_attn", "linear_attn", "mlp")


def family(name):
    for f in FAM:
        if f".{f}." in name:
            return f
    return "other"


@torch.no_grad()
def directions(W, gen):
    """Unit-Frobenius directions in W's SVD basis. Returns {name: (m,n) tensor}."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    V = Vh.T
    r = S.numel()
    dev = W.device
    c = torch.randn(r, generator=gen).to(dev)
    A = torch.randn(r, r, generator=gen).to(dev)
    B = torch.randn(r, r, generator=gen).to(dev)
    A = A - A.T
    B = B - B.T
    d = {
        "spec_rand": U @ torch.diag(c) @ V.T,
        "spec_scale": U @ torch.diag(S) @ V.T,
        # dW = U (A Sigma - Sigma B) V^T for U <- U exp(A), V <- V exp(B)
        "frame_rot": U @ (A @ torch.diag(S) - torch.diag(S) @ B) @ V.T,
        "gauss": torch.randn(W.shape, generator=gen).to(dev),
    }
    return {k: v / v.norm().clamp_min(1e-30) for k, v in d.items()}


@torch.no_grad()
def build_reference(tok, n_prompts, seed, device, max_len=512):
    data = load_gsm8k("test")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(data), generator=g)[:n_prompts].tolist()
    ids, masks = [], []
    for i in idx:
        ex = data[i]
        p = build_prompt(tok, ex["question"])
        a = ex["solution"]
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        a_ids = tok(a, add_special_tokens=False)["input_ids"]
        seq = (p_ids + a_ids)[:max_len]
        m = ([0] * len(p_ids) + [1] * len(a_ids))[:max_len]
        ids.append(seq)
        masks.append(m)
    T = max(len(s) for s in ids)
    pad = tok.pad_token_id
    input_ids = torch.full((len(ids), T), pad, dtype=torch.long)
    attn = torch.zeros((len(ids), T), dtype=torch.long)
    ans = torch.zeros((len(ids), T), dtype=torch.bool)
    for j, (s, m) in enumerate(zip(ids, masks)):
        input_ids[j, :len(s)] = torch.tensor(s)
        attn[j, :len(s)] = 1
        ans[j, :len(s)] = torch.tensor(m, dtype=torch.bool)
    return input_ids.to(device), attn.to(device), ans.to(device)


@torch.no_grad()
def logp_batch(model, input_ids, attn, use_autocast):
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else torch.autocast("cpu", enabled=False)
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attn)
    return F.log_softmax(out.logits.float()[:, :-1], dim=-1)  # predicts token t+1


@torch.no_grad()
def kl_tokens(logp0, logp1, mask):
    """Forward KL(p0 || p1) per position, averaged over masked positions."""
    kl = (logp0.exp() * (logp0 - logp1)).sum(-1)           # (B, T-1)
    m = mask[:, 1:].float()
    return (kl * m).sum() / m.sum().clamp_min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--etas", default="1e-3,3e-3,1e-2,3e-2")
    ap.add_argument("--max-matrices", type=int, default=0, help="0 = all tracked")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cpu" if a.cpu else "cuda"
    etas = [float(x) for x in a.etas.split(",")]

    model, tok, n_layers = load_model(a.model, dtype=torch.float32, device=device, trainable=False)
    tracked = select_tracked(model, n_layers)
    names = list(tracked)
    if a.max_matrices:
        names = names[:a.max_matrices]
    input_ids, attn, ans = build_reference(tok, a.n_prompts, a.seed, device, a.max_len)
    n_tok = int(ans.sum())
    print(f"reference: {a.n_prompts} prompts, {n_tok} answer tokens, T={input_ids.shape[1]}; "
          f"{len(names)} matrices x 4 directions x {len(etas)} etas", flush=True)

    batches = [(input_ids[i:i + a.batch], attn[i:i + a.batch], ans[i:i + a.batch])
               for i in range(0, input_ids.shape[0], a.batch)]
    base = [logp_batch(model, ii, am, not a.cpu) for ii, am, _ in batches]
    gen = torch.Generator().manual_seed(a.seed)
    rows = []
    t0 = time.time()
    for k, n in enumerate(names):
        W = tracked[n]
        W0 = W.detach().clone()
        wnorm = W0.norm().item()
        dirs = directions(W0, gen)
        for dname, d in dirs.items():
            for eta in etas:
                W.copy_(W0 + eta * wnorm * d.to(W.dtype))
                num, den = 0.0, 0.0
                for (ii, am, an), lp0 in zip(batches, base):
                    lp1 = logp_batch(model, ii, am, not a.cpu)
                    m = an[:, 1:].float().sum().item()
                    num += kl_tokens(lp0, lp1, an).item() * m
                    den += m
                W.copy_(W0)
                kl = num / max(den, 1)
                rows.append(dict(matrix=n, family=family(n), m=W0.shape[0], n=W0.shape[1],
                                 w_fro=wnorm, direction=dname, eta=eta, step_fro=eta * wnorm,
                                 kl=kl, kl_per_step2=kl / (eta * wnorm) ** 2))
        del dirs
        print(f"[{k + 1}/{len(names)}] {n} done ({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(a.out, "kl_probe.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "kl_probe.csv"), index=False)
    # log-log slope of KL vs eta per (matrix, direction)
    fits = []
    for (mat, dn), g in df.groupby(["matrix", "direction"]):
        x = torch.log(torch.tensor(g.eta.values))
        y = torch.log(torch.tensor(g.kl.values).clamp_min(1e-30))
        slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        fits.append(dict(matrix=mat, family=family(mat), direction=dn, slope=slope.item(),
                         kl_per_step2_at_min_eta=float(g.sort_values("eta").kl_per_step2.iloc[0])))
    fd = pd.DataFrame(fits)
    fd.to_csv(os.path.join(a.out, "kl_probe_fits.csv"), index=False)
    summ = fd.groupby(["family", "direction"]).agg(slope_median=("slope", "median"),
                                                   sens_median=("kl_per_step2_at_min_eta", "median")).reset_index()
    print(summ.to_string(), flush=True)
    json.dump({"args": vars(a), "n_answer_tokens": n_tok, "n_matrices": len(names),
               "elapsed_s": time.time() - t0}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
