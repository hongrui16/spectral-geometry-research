"""v5 SFT data: sampled solutions that terminated and are strictly correct.

  python scripts_v5/make_sft_data.py --model Qwen/Qwen3.5-2B --k 4 --temperature 0.7 \
      --shard 0 --nshards 4 --out $RUNS/data/teacher2b.shard0.jsonl

teacher data = a larger model of the same family (off-policy SFT);
self data    = the student itself at T=1 (rejection-sampled on-policy SFT, Q3).
Keeps the first correct sample per question (no length selection). Rows:
{"question", "completion", "gold", "idx", "n_correct", "k"}; questions with no
correct sample are listed in the stats file only.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom_v5.data import build_prompt, load_gsm8k, reward_fn, split_at_stop
from specgeom_v5.modeling import load_model, stop_token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch", type=int, default=32, help="questions per generate call")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="teacher weights; bf16 is fine for data generation (e.g. an 8B teacher)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed + args.shard)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tok, _ = load_model(args.model, dtype=dtype, trainable=False)
    stops = stop_token_ids(tok)
    data = load_gsm8k("train")
    idxs = list(range(len(data)))[args.shard::args.nshards]
    if args.limit:
        idxs = idxs[:args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    kept, trunc, total, t0 = 0, 0, 0, time.time()
    lengths, n_correct_all = [], 0
    with open(args.out, "w") as f:
        for b in range(0, len(idxs), args.batch):
            chunk = idxs[b:b + args.batch]
            prompts = [build_prompt(tok, data[i]["question"]) for i in chunk]
            # chat templates already carry BOS (Llama); default add_special_tokens doubled it
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left",
                      add_special_tokens=False).to(model.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(**enc, do_sample=True, temperature=args.temperature,
                                     top_p=args.top_p, top_k=args.top_k,
                                     max_new_tokens=args.max_new_tokens,
                                     num_return_sequences=args.k, eos_token_id=stops,
                                     pad_token_id=tok.pad_token_id)
            rows = gen[:, enc["input_ids"].shape[1]:].tolist()
            for qi, i in enumerate(chunk):
                ex, first, n_ok = data[i], None, 0
                for j in range(args.k):
                    ids, term = split_at_stop(rows[qi * args.k + j], stops)
                    total += 1
                    trunc += int(not term)
                    lengths.append(len(ids))
                    # drop the stop token from the stored text; train.py appends eos
                    text = tok.decode([t for t in ids if t not in stops],
                                      skip_special_tokens=True).strip()
                    if reward_fn(text, ex["gold"], terminated=term, strict=True):
                        n_ok += 1
                        n_correct_all += 1
                        first = first or text
                if first:
                    kept += 1
                    f.write(json.dumps({"question": ex["question"], "completion": first,
                                        "gold": ex["gold"], "idx": i, "n_correct": n_ok,
                                        "k": args.k}) + "\n")
            done = min(b + args.batch, len(idxs))
            print(f"{done}/{len(idxs)} kept={kept} trunc={trunc / max(1, total):.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    stats = {"model": args.model, "k": args.k, "temperature": args.temperature,
             "top_p": args.top_p, "top_k": args.top_k,
             "max_new_tokens": args.max_new_tokens, "shard": args.shard,
             "nshards": args.nshards, "questions": len(idxs), "kept": kept,
             "sample_trunc_frac": trunc / max(1, total),
             "sample_acc": n_correct_all / max(1, total),
             "len_pct": {p: sorted(lengths)[min(len(lengths) - 1, int(p / 100 * len(lengths)))]
                         for p in (50, 90, 95, 99)} if lengths else {}}
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print("DONE", stats)


if __name__ == "__main__":
    main()
