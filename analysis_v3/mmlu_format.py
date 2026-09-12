"""E-v3-0b: is the MMLU collapse a format drift? (v3 doc §4 H-4)

For a checkpoint, on the first --limit MMLU questions (same shuffle as
scripts_v3/eval_mmlu.py) report at the answer position:
  * mean probability mass on the four letter tokens {A,B,C,D}
  * fraction of questions whose full-vocab argmax is one of the four letters
  * letter-logit accuracy (the standard metric) for reference
and dump --gen greedy generations (64 tokens) so the drift can be read by eye.

Usage (GPU):
  python analysis_v3/mmlu_format.py --model $RUNS/e4a_rlvr_full_s0/ckpt_000300 \
      --out results/v3/result_B/mmlu_format/e4a_rlvr_full_s0.json
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts_v3.eval_mmlu import LETTERS, build_prompt
from specgeom_v3.modeling import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, tok, _ = load_model(a.model, dtype=torch.bfloat16, trainable=False)
    letter_ids = torch.tensor([tok.encode(L, add_special_tokens=False)[0] for L in LETTERS])
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0)
    ds = ds.select(range(min(a.limit, len(ds))))

    mass, in_letters, correct, n = 0.0, 0, 0, 0
    top_tokens = {}
    for i in range(0, len(ds), a.batch):
        rows = ds[i:i + a.batch]
        prompts = [build_prompt(tok, q, ch) for q, ch in zip(rows["question"], rows["choices"])]
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1].float()
        probs = logits.softmax(-1).cpu()
        mass += probs[:, letter_ids].sum(-1).sum().item()
        am = probs.argmax(-1)
        in_letters += sum(int(t in letter_ids.tolist()) for t in am.tolist())
        for t in am.tolist():
            s = tok.decode([t])
            top_tokens[s] = top_tokens.get(s, 0) + 1
        picks = logits[:, letter_ids.to(logits.device)].argmax(-1).cpu()
        correct += (picks == torch.tensor(rows["answer"])).sum().item()
        n += len(rows["answer"])

    gens = []
    for i in range(0, min(a.gen, len(ds)), a.batch):
        rows = ds[i:i + a.batch]
        prompts = [build_prompt(tok, q, ch) for q, ch in zip(rows["question"], rows["choices"])]
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=64,
                               pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for q, t, ans in zip(rows["question"], texts, rows["answer"]):
            gens.append({"question": q[:200], "gold": LETTERS[ans], "generation": t})

    res = {"model": a.model, "n": n, "letter_mass_mean": mass / n,
           "argmax_in_letters_frac": in_letters / n, "mmlu_letter_logit": correct / n,
           "top_argmax_tokens": sorted(top_tokens.items(), key=lambda kv: -kv[1])[:10],
           "generations": gens}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "generations"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
