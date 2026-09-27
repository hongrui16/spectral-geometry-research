"""Retention suite (v5 forgetting readout; docs/unified_paper_document_v5.md part 3 §0).

  python scripts_v5/eval_retention.py --model <ckpt-or-hf-id> --out ret.json
         [--tasks mmlu,arc,hellaswag,triviaqa,mmlu_gen]

Tasks (fixed, seeded subsets):
- mmlu      MMLU test shuffle(seed=0)[:2000]. Chat prompt, assistant turn prefilled
            with "Answer:", score of each letter = logsumexp over its two token
            variants ("A", " A"). Also records the letter probability mass: v2-v4
            scored the first assistant token, which measured answer format
            (collapsed models put 2.5% of argmax on letters yet reasoned correctly).
- arc       ARC-Challenge test (1172), same scoring as mmlu (up to 5 options).
- hellaswag validation shuffle(seed=0)[:1000], plain-text continuation log-likelihood,
            acc_norm (sum logprob / ending characters), lm-eval-harness preprocessing.
- triviaqa  rc.nocontext validation shuffle(seed=0)[:1000], greedy short answer,
            normalized exact match against the alias list.
- mmlu_gen  MMLU first 300 of the same shuffle, greedy generation (<=512 tokens),
            'Answer: X' extraction; format-independent cross-check, not in the mean.
retention = mean(mmlu, arc, hellaswag, triviaqa).
Numerics: fp32 weights + bf16 autocast (training forward) unless --numerics.
"""

import argparse
import json
import os
import re
import string
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specgeom_v5.data import split_at_stop
from specgeom_v5.modeling import load_model, stop_token_ids

LETTERS = "ABCDE"
PRIMARY = ("mmlu", "arc", "hellaswag", "triviaqa")
DEFAULT_LIMITS = {"mmlu": 2000, "arc": 0, "hellaswag": 1000, "triviaqa": 1000,
                  "mmlu_gen": 300}


def chat(tok, user, prefill=""):
    msgs = [{"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + prefill


def mc_user(question, options):
    body = question.strip() + "\n" + "\n".join(
        f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    letters = ", ".join(LETTERS[:len(options)])
    return body + f"\n\nAnswer with a single letter ({letters})."


class Evaluator:
    def __init__(self, model, tok, numerics, batch):
        self.model, self.tok, self.batch = model, tok, batch
        self.amp = numerics == "amp"
        self.stops = stop_token_ids(tok)
        # token variants per letter: bare and space-prefixed (first token of each)
        self.letter_ids = [sorted({tok.encode(L, add_special_tokens=False)[0],
                                   tok.encode(" " + L, add_special_tokens=False)[0]})
                           for L in LETTERS]

    def ctx(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp)

    @torch.no_grad()
    def last_logits(self, prompts):
        enc = self.tok(prompts, return_tensors="pt", padding=True, padding_side="left",
                       add_special_tokens=False).to(self.model.device)
        with self.ctx():
            out = self.model(**enc).logits[:, -1]
        return out.float()

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens):
        enc = self.tok(prompts, return_tensors="pt", padding=True, padding_side="left",
                       add_special_tokens=False).to(self.model.device)
        with self.ctx():
            gen = self.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                      eos_token_id=self.stops,
                                      pad_token_id=self.tok.pad_token_id)
        texts, terms = [], []
        for row in gen[:, enc["input_ids"].shape[1]:].tolist():
            ids, term = split_at_stop(row, self.stops)
            texts.append(self.tok.decode(ids, skip_special_tokens=True))
            terms.append(term)
        return texts, terms

    def mc_prefill(self, items):
        """items: list of (question, options, gold_index)."""
        correct, mass, in_letters = [], [], []
        for i in range(0, len(items), self.batch):
            chunk = items[i:i + self.batch]
            prompts = [chat(self.tok, mc_user(q, o), prefill="Answer:") for q, o, _ in chunk]
            logp = F.log_softmax(self.last_logits(prompts), -1)
            for (q, o, g), lp in zip(chunk, logp):
                k = len(o)
                scores = torch.stack([torch.logsumexp(lp[self.letter_ids[j]], 0)
                                      for j in range(k)])
                correct.append(int(scores.argmax().item() == g))
                mass.append(scores.exp().sum().item())
                valid = {t for j in range(k) for t in self.letter_ids[j]}
                in_letters.append(int(lp.argmax().item() in valid))
        n = len(correct)
        return {"acc": sum(correct) / n, "n": n, "letter_mass": sum(mass) / n,
                "argmax_in_letters": sum(in_letters) / n, "correct": correct}

    @torch.no_grad()
    def continuation_logprob(self, pairs):
        """pairs: list of (context, continuation) strings -> summed logprob of continuation."""
        out = []
        for i in range(0, len(pairs), self.batch):
            chunk = pairs[i:i + self.batch]
            rows = []
            for c, x in chunk:
                ci = self.tok.encode(c, add_special_tokens=False)
                xi = self.tok.encode(x, add_special_tokens=False)
                rows.append((ci + xi, len(ci), len(xi)))
            L = max(len(r[0]) for r in rows)
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.full((len(rows), L), pad, dtype=torch.long)
            att = torch.zeros((len(rows), L), dtype=torch.long)
            for j, (r, _, _) in enumerate(rows):
                ids[j, :len(r)] = torch.tensor(r)
                att[j, :len(r)] = 1
            ids, att = ids.to(self.model.device), att.to(self.model.device)
            with self.ctx():
                logits = self.model(input_ids=ids, attention_mask=att).logits
            logp = F.log_softmax(logits.float(), -1)
            for j, (r, lc, lx) in enumerate(rows):
                tgt = ids[j, lc:lc + lx]
                lp = logp[j, lc - 1:lc - 1 + lx].gather(1, tgt[:, None]).sum()
                out.append(lp.item())
        return out


def load_mmlu(limit, offset=0):
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0)
    ds = ds.select(range(offset, min(offset + limit, len(ds))))
    return [(r["question"], r["choices"], r["answer"]) for r in ds]


def load_arc(limit):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for r in ds:
        labels = r["choices"]["label"]
        items.append((r["question"], r["choices"]["text"], labels.index(r["answerKey"])))
    return items[:limit] if limit else items


def _hs_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag(limit):
    ds = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=0)
    ds = ds.select(range(min(limit, len(ds))))
    items = []
    for r in ds:
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        query = _hs_pre(r["activity_label"] + ": " + ctx)
        items.append((query, [_hs_pre(e) for e in r["endings"]], int(r["label"])))
    return items


def _norm_ans(s):
    """Official TriviaQA normalisation: punctuation and '_' become spaces (v5 review fix: the
    first version deleted them, so "Children's" -> "childrens" missed the alias "children s")."""
    s = s.lower()
    s = "".join(" " if (ch in set(string.punctuation) or ch == "_") else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def load_triviaqa(limit):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=0)
    ds = ds.select(range(min(limit, len(ds))))
    return [(r["question"], r["answer"]["normalized_aliases"]) for r in ds]


_GEN_ANS = re.compile(r"Answer\s*:\s*\(?\s*([A-E])\b")


def run(task, ev, limit):
    if task == "mmlu":
        return ev.mc_prefill(load_mmlu(limit))
    if task == "arc":
        return ev.mc_prefill(load_arc(limit))
    if task == "hellaswag":
        items = load_hellaswag(limit)
        pairs = [(q, " " + e) for q, ends, _ in items for e in ends]
        lps = ev.continuation_logprob(pairs)
        correct, k = [], 0
        for q, ends, g in items:
            s = [lps[k + j] / max(1, len(ends[j])) for j in range(len(ends))]
            k += len(ends)
            correct.append(int(max(range(len(s)), key=s.__getitem__) == g))
        return {"acc": sum(correct) / len(correct), "n": len(correct), "correct": correct}
    if task == "triviaqa":
        items = load_triviaqa(limit)
        correct, preds = [], []
        for i in range(0, len(items), ev.batch):
            chunk = items[i:i + ev.batch]
            prompts = [chat(ev.tok, "Answer the following question with only the answer, "
                            "no explanation.\nQuestion: " + q) for q, _ in chunk]
            texts, _ = ev.generate(prompts, 24)
            for (q, aliases), t in zip(chunk, texts):
                p = _norm_ans(t.strip().split("\n")[0])
                correct.append(int(p in set(aliases)))
                preds.append(p)
        return {"acc": sum(correct) / len(correct), "n": len(correct), "correct": correct,
                "preds": preds}
    if task == "mmlu_gen":
        items = load_mmlu(limit)
        correct, extracted, terminated = [], [], []
        for i in range(0, len(items), ev.batch):
            chunk = items[i:i + ev.batch]
            prompts = [chat(ev.tok, mc_user(q, o).rsplit("\n\n", 1)[0] +
                            "\n\nThink briefly, then end with 'Answer: X' where X is the letter.")
                       for q, o, _ in chunk]
            texts, terms = ev.generate(prompts, 512)
            for (q, o, g), t, term in zip(chunk, texts, terms):
                m = _GEN_ANS.findall(t)
                pred = LETTERS.index(m[-1]) if m else None
                correct.append(int(pred == g))
                extracted.append(int(pred is not None))
                terminated.append(int(term))
        n = len(correct)
        return {"acc": sum(correct) / n, "n": n, "extract_frac": sum(extracted) / n,
                "term_frac": sum(terminated) / n, "correct": correct}
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="mmlu,arc,hellaswag,triviaqa,mmlu_gen")
    ap.add_argument("--limit", action="append", default=[],
                    help="task=N overrides, e.g. --limit mmlu=500 (0 = all)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--numerics", choices=["amp", "bf16", "fp32"], default="amp")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    limits = dict(DEFAULT_LIMITS)
    for s in args.limit:
        k, v = s.split("=")
        limits[k] = int(v)
    dtype = torch.bfloat16 if args.numerics == "bf16" else torch.float32
    model, tok, _ = load_model(args.model, dtype=dtype, trainable=False)
    ev = Evaluator(model, tok, args.numerics, args.batch)

    res = {"model": args.model, "numerics": args.numerics, "tasks": {}}
    for task in args.tasks.split(","):
        t0 = time.time()
        r = run(task, ev, limits[task])
        r["secs"] = round(time.time() - t0, 1)
        res["tasks"][task] = r
        extra = {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and k != "acc"}
        print(f"[{task}] acc={r['acc']:.4f} n={r['n']} {extra}", flush=True)
    prim = [res["tasks"][t]["acc"] for t in PRIMARY if t in res["tasks"]]
    if len(prim) == len(PRIMARY):
        res["retention"] = sum(prim) / len(prim)
        print(f"FINAL retention = {res['retention']:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
