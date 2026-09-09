"""Greedy pass@1 on GSM8K test. Usage:
  python scripts/eval_gsm8k.py --model <hf-id-or-ckpt-dir> [--limit 200]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom.data import build_prompt, load_gsm8k, reward_fn
from specgeom.modeling import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-samples", type=int, default=0,
                    help="save first N generations alongside --out for debugging")
    args = ap.parse_args()

    model, tok, _ = load_model(args.model, dtype=torch.bfloat16, trainable=False)
    data = load_gsm8k("test")
    if args.limit:
        data = data[:args.limit]

    correct, results = 0, []
    for i in range(0, len(data), args.batch):
        chunk = data[i:i + args.batch]
        prompts = [build_prompt(tok, ex["question"]) for ex in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        for ex, t in zip(chunk, texts):
            r = reward_fn(t, ex["gold"])
            correct += r
            rec = {"gold": ex["gold"], "reward": r}
            if len(results) < args.dump_samples:
                rec["text"] = t
            results.append(rec)
        print(f"{i + len(chunk)}/{len(data)} acc={correct / len(results):.4f}",
              flush=True)

    acc = correct / len(results)
    print(f"FINAL pass@1 = {acc:.4f} on {len(results)} examples")
    if args.out:
        with open(args.out, "w") as f:
            payload = {"model": args.model, "n": len(results), "pass@1": acc}
            if args.dump_samples:
                payload["samples"] = [r for r in results if "text" in r]
            json.dump(payload, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
