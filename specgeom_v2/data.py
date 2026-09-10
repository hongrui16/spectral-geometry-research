"""GSM8K data handling: prompts, gold answers, reward."""

import re
import random
from datasets import load_dataset

SYSTEM = "You are a helpful math assistant."
INSTR = ("Solve the problem step by step. "
         "Put the final numeric answer after '####'.\n\nProblem: ")


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


def extract_answer(text: str):
    m = re.findall(r"####\s*([\-\d\.,]+)", text)
    if m:
        return m[-1].strip().replace(",", "").rstrip(".")
    # fallback: last number in the text
    m = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    return m[-1].replace(",", "") if m else None


def reward_fn(completion: str, gold: str) -> float:
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    try:
        return float(abs(float(pred) - float(gold)) < 1e-6)
    except ValueError:
        return float(pred == gold)


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
