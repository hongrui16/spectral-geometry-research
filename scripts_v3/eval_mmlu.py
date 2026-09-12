"""MMLU subset eval (forgetting metric). Zero-shot, letter-choice by rank of
first-token logits over A/B/C/D — fast, no generation.

  python scripts/eval_mmlu.py --model <ckpt-or-hf-id> [--limit 1000]
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom_v3.modeling import load_model

LETTERS = ["A", "B", "C", "D"]


def build_prompt(tok, q, choices):
    body = q + "\n" + "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, choices))
    msgs = [{"role": "user", "content":
             body + "\n\nAnswer with a single letter (A, B, C, or D)."}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok, _ = load_model(args.model, dtype=torch.bfloat16, trainable=False)
    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=0).select(range(min(args.limit, len(ds))))

    correct = 0
    for i in range(0, len(ds), args.batch):
        rows = ds[i:i + args.batch]
        prompts = [build_prompt(tok, q, ch)
                   for q, ch in zip(rows["question"], rows["choices"])]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1]
        picks = logits[:, letter_ids].argmax(-1)
        correct += (picks.cpu() == torch.tensor(rows["answer"])).sum().item()
        print(f"{i + len(rows['answer'])}/{len(ds)} "
              f"acc={correct / (i + len(rows['answer'])):.4f}", flush=True)

    acc = correct / len(ds)
    print(f"FINAL mmlu = {acc:.4f} on {len(ds)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "n": len(ds), "mmlu": acc}, f)


if __name__ == "__main__":
    main()
