"""v6 decisive test: does a format reminder recover SFT's 'forgotten' MATH / Countdown?

For each dataset, k exemplars = items the BASE model answered correctly (strict, finished) in the
zero-shot eval, shown as previous chat turns with the base model's own answer. The exemplars are
excluded from scoring. Same prompts / greedy / 4096 tokens / authors' scoring as the zero-shot eval,
so the k-shot accuracy compares directly with the zero-shot accuracy on the same items.

  PYTHONPATH=$RBD/code:$RBD/pydeps $PY analysis_v6/fewshot_eval.py --model <ckpt> --tag sft \
      --base-eval $RUNS/eval/base_qwen15 --out results/v6/result_A/fewshot_sft_qwen15.json
"""
import argparse
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rescore import countdown_lenient, dataset_obj, math_lenient  # noqa: E402


def load(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        if isinstance(r.get("datapoint"), str):
            r["datapoint"] = ast.literal_eval(r["datapoint"])
        rows.append(r)
    return rows


def correct(r):
    return str(r.get("correct")).lower() == "true"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-eval", required=True)
    ap.add_argument("--zero-eval", default=None, help="this model's zero-shot eval dir (for comparison)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit-math", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=4096)

    res = {"model": a.model, "tag": a.tag, "k": a.k, "datasets": {}}
    for ds_name in ("math", "countdown"):
        base = load(os.path.join(a.base_eval, ds_name, "eval", "predictions.jsonl"))
        if ds_name == "math":
            base = base[:a.limit_math]
        shots = [i for i, r in enumerate(base) if correct(r) and r.get("finish_reason") == "stop"
                 and len(r["output_text"]) < 2500][:a.k]
        msgs_prefix = []
        for i in shots:
            msgs_prefix += [{"role": "user", "content": base[i]["datapoint"]["messages"][0]["content"]},
                            {"role": "assistant", "content": base[i]["output_text"]}]
        items = [i for i in range(len(base)) if i not in shots]
        prompts = [tok.apply_chat_template(
            msgs_prefix + [{"role": "user", "content": base[i]["datapoint"]["messages"][0]["content"]}],
            tokenize=False, add_generation_prompt=True) for i in items]
        outs = llm.generate(prompts, sp)
        ds = dataset_obj(ds_name)
        strict, lenient, fin = [], [], []
        for i, o in zip(items, outs):
            text, dp = o.outputs[0].text, base[i]["datapoint"]
            f = o.outputs[0].finish_reason == "stop"
            if ds_name == "math":
                s = bool(ds.reward_fn(ds.parse_output_text(text), dp))
                l = s or math_lenient(ds, text, dp)
            else:
                s = bool(ds.reward_fn(text, dp))
                l = s or countdown_lenient(ds, text, dp, f)
            strict.append(int(s)); lenient.append(int(l)); fin.append(int(f))
        n = len(items)
        r = {"n": n, "shots": shots, "fewshot_strict": sum(strict) / n,
             "fewshot_lenient": sum(lenient) / n, "fewshot_finished": sum(fin) / n,
             "fewshot_boxed": sum("\\boxed" in o.outputs[0].text for o in outs) / n}
        if a.zero_eval:
            z = load(os.path.join(a.zero_eval, ds_name, "eval", "predictions.jsonl"))
            r["zeroshot_strict_same_items"] = sum(correct(z[i]) for i in items) / n
        r["samples"] = [o.outputs[0].text[-500:] for o in outs[:5]]
        res["datasets"][ds_name] = r
        print(a.tag, ds_name, {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in r.items() if k != "samples"}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
