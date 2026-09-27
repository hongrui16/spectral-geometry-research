"""Greedy pass@1 on GSM8K test (v5).

  python scripts_v5/eval_gsm8k.py --model <hf-id-or-ckpt-dir> [--limit 500]

v5 readouts (docs/unified_paper_document_v5.md part 3 §0):
- pass@1 (primary) = terminated AND strict '####' answer correct (same rule as the RLVR reward)
- pass@1_flex      = terminated AND last-number fallback correct
- pass@1_legacy    = v2-v4 rule (no termination check, last-number fallback)
- trunc_frac, format_frac (#### present), resp_len_mean
Numerics: fp32 weights + bf16 autocast, i.e. the forward the training loop runs
(v2-v4 loaded bf16 weights, which rounds away sub-ulp updates).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom_v5.data import (build_prompt, extract_answer, has_format, is_correct,
                              load_gsm8k, split_at_stop)
from specgeom_v5.modeling import load_model, stop_token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--numerics", choices=["amp", "bf16", "fp32"], default="amp",
                    help="amp = fp32 weights + bf16 autocast (training forward); "
                         "bf16 = v2-v4 behaviour; fp32 = no autocast")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-samples", type=int, default=0,
                    help="save first N generations alongside --out for debugging")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.numerics == "bf16" else torch.float32
    model, tok, _ = load_model(args.model, dtype=dtype, trainable=False)
    stops = stop_token_ids(tok)
    data = load_gsm8k("test")
    if args.limit:
        data = data[:args.limit]

    recs = []
    for i in range(0, len(data), args.batch):
        chunk = data[i:i + args.batch]
        prompts = [build_prompt(tok, ex["question"]) for ex in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=args.numerics == "amp"):
            gen = model.generate(**enc, do_sample=False,
                                 max_new_tokens=args.max_new_tokens,
                                 eos_token_id=stops,
                                 pad_token_id=tok.pad_token_id)
        for ex, row in zip(chunk, gen[:, enc["input_ids"].shape[1]:].tolist()):
            ids, term = split_at_stop(row, stops)
            t = tok.decode(ids, skip_special_tokens=True)
            rec = {"gold": ex["gold"], "terminated": term, "len": len(ids),
                   "strict": term and is_correct(extract_answer(t, True), ex["gold"]),
                   "flex": term and is_correct(extract_answer(t, False), ex["gold"]),
                   "legacy": is_correct(extract_answer(t, False), ex["gold"]),
                   "format": has_format(t)}
            if len(recs) < args.dump_samples:
                rec["text"] = t
            recs.append(rec)
        n = len(recs)
        print(f"{n}/{len(data)} strict={sum(r['strict'] for r in recs) / n:.4f} "
              f"trunc={sum(not r['terminated'] for r in recs) / n:.3f}", flush=True)

    n = len(recs)
    out = {"model": args.model, "n": n, "numerics": args.numerics,
           "max_new_tokens": args.max_new_tokens,
           "pass@1": sum(r["strict"] for r in recs) / n,
           "pass@1_flex": sum(r["flex"] for r in recs) / n,
           "pass@1_legacy": sum(r["legacy"] for r in recs) / n,
           "trunc_frac": sum(not r["terminated"] for r in recs) / n,
           "format_frac": sum(r["format"] for r in recs) / n,
           "resp_len_mean": sum(r["len"] for r in recs) / n,
           "correct": [int(r["strict"]) for r in recs]}
    print("FINAL " + " ".join(f"{k}={v:.4f}" for k, v in out.items()
                              if isinstance(v, float)))
    if args.out:
        if args.dump_samples:
            out["samples"] = [r for r in recs if "text" in r]
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
