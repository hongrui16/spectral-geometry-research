"""GSM8K data handling: prompts, gold answers, reward, SFT targets.

v5 changes (docs/unified_paper_document_v5.md, part 3 §0):
- reward requires a terminated completion (stop token emitted); truncated
  completions score 0. v2-v4 scored the last number of never-ending outputs,
  which RLVR at lr x1 learned to exploit (trunc_frac 1.0 with reward 0.5-0.7).
- strict extraction (default) only accepts the number after the last '####';
  the v4 'last number anywhere' fallback is kept as the 'flex' readout.
- SFT targets are selectable: raw GSM8K solutions (v4), GSM8K with the
  <<calc>> annotations stripped, or a jsonl of generated solutions
  (teacher / self-distilled, scripts_v5/make_sft_data.py).
"""

import json
import random
import re

from datasets import load_dataset

SYSTEM = "You are a helpful math assistant."
INSTR = ("Solve the problem step by step. "
         "Put the final numeric answer after '####'.\n\nProblem: ")

_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_STRICT = re.compile(r"####\s*\$?\s*(" + _NUM + r")")
_ANY = re.compile(_NUM)
_CALC = re.compile(r"<<[^<>]*>>")


def load_gsm8k(split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for ex in ds:
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"question": ex["question"], "solution": ex["answer"], "gold": gold})
    return out


def build_prompt(tokenizer, question: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTR + question}]
    try:
        # Qwen3-family templates: disable chain-of-thought mode so completions
        # are short enough for the token budget and '####' extraction.
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def _norm(s: str) -> str:
    return s.replace(",", "").rstrip(".")


def extract_answer(text: str, strict: bool = True):
    """Number after the last '####' (strict); flex falls back to the last number."""
    m = _STRICT.findall(text)
    if m:
        return _norm(m[-1])
    if strict:
        return None
    m = _ANY.findall(text)
    return _norm(m[-1]) if m else None


def is_correct(pred, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def reward_fn(completion: str, gold: str, terminated: bool = True,
              strict: bool = True) -> float:
    """1.0 iff the completion terminated and its answer matches gold."""
    if not terminated:
        return 0.0
    return float(is_correct(extract_answer(completion, strict=strict), gold))


def has_format(completion: str) -> bool:
    return bool(_STRICT.search(completion))


def strip_calc(solution: str) -> str:
    """GSM8K reference solution without the <<a*b=c>> calculator annotations."""
    return _CALC.sub("", solution)


def load_sft_examples(source: str):
    """SFT (question, completion, gold) triples.

    source: 'gsm8k_raw' (v4 behaviour), 'gsm8k_clean' (<<calc>> stripped), or a
    path to a jsonl with fields question / completion / gold.
    """
    if source in ("gsm8k_raw", "gsm8k_clean"):
        out = []
        for ex in load_gsm8k("train"):
            comp = ex["solution"] if source == "gsm8k_raw" else strip_calc(ex["solution"])
            out.append({"question": ex["question"], "completion": comp, "gold": ex["gold"]})
        return out
    with open(source) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        assert {"question", "completion", "gold"} <= set(r), f"bad sft row keys {sorted(r)}"
    return rows


def split_at_stop(ids, stop_ids):
    """Cut a generated id list after its first stop token.

    Returns (kept_ids, terminated). The stop token is kept (it is trained on);
    everything after it (padding) is dropped. v2-v4 stripped trailing pad ids,
    and since pad == <|endoftext|> is also a stop id, that stop token was lost.
    """
    stop = set(stop_ids)
    for i, t in enumerate(ids):
        if t in stop:
            return list(ids[:i + 1]), True
    return list(ids), False


class PromptSampler:
    def __init__(self, examples, seed=0):
        self.examples = list(examples)
        self.rng = random.Random(seed)
        self._order = []

    def next(self, n):
        out = []
        for _ in range(n):
            if not self._order:
                self._order = list(range(len(self.examples)))
                self.rng.shuffle(self._order)
            out.append(self.examples[self._order.pop()])
        return out
