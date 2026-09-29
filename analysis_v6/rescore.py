"""v6: re-score saved Retaining-by-Doing predictions (docs/unified_paper_document_v6.md part 2 §1).

Reads <eval_dir>/<dataset>/eval/predictions.jsonl written by the authors' core.evaluation.run and
reports, per dataset:
  strict        the authors' own `correct` field (their parser + reward_fn)          [reproduction]
  finished      fraction with finish_reason == 'stop' (did not hit the 4096-token cap)
  acc_finished  strict accuracy among finished generations
  lenient       format-lenient re-extraction (never uses the gold answer to pick the extraction):
                  mmlu      : first hit of an ordered list of answer patterns (see MMLU_PATTERNS),
                              then a lone option letter on the last non-empty line
                  countdown : FINISHED answers only: the authors' validate_expression on the LAST
                              '<expr> = target' equation (rescues e.g. \\boxed{41} after '9 - 14 + 46 = 41'
                              and \\boxed{32 + 43 - 47 = 28}, both marked wrong by the authors' parser)
                  math      : the authors' parser, plus 'answer is X' / 'Final answer: X' fallbacks
                  ifeval    : = strict (following the format IS the task)
  lenient_tiers (mmlu) how many answers each pattern tier recovered.
Every lenient-only hit can be dumped (--dump N) for the manual audit required by v6 rule 3.

  PYTHONPATH=/scratch/rhong5/rbd/code:/scratch/rhong5/rbd/pydeps ~/envs_py310_new/bin/python \
      analysis_v6/rescore.py /scratch/rhong5/rbd/runs/eval/base_qwen15 [--dump 20]
"""
import argparse
import ast
import json
import os
import re
import sys

MMLU_PATTERNS = [
    ("authors", re.compile(r"\bThe(.*)answer is(?: option)?:?\s*(\w+)", re.I), 2),
    ("answer_is", re.compile(r"answer\s*(?:is|:)\s*[:\-]?\s*\(?\*{0,2}([ABCD])\b", re.I), 1),
    ("option_is", re.compile(r"\b(?:option|choice)\s*\(?([ABCD])\)?\s*(?:is|would be)\s*(?:the\s*)?(?:correct|right|best)", re.I), 1),
    ("correct_is", re.compile(r"(?:correct|right|best)\s*(?:answer|option|choice)?\s*(?:is|would be|:)\s*\(?\*{0,2}([ABCD])\b", re.I), 1),
    ("bold", re.compile(r"\*\*\(?([ABCD])[.)]?\s", re.I), 1),
]
_LONE = re.compile(r"^\W*\(?([ABCD])\)?\W*$")


def load(path):
    rows = []
    with open(path) as f:
        for l in f:
            r = json.loads(l)
            for k in ("datapoint",):
                if isinstance(r.get(k), str):
                    try:
                        r[k] = ast.literal_eval(r[k])
                    except Exception:
                        pass
            rows.append(r)
    return rows


def truthy(v):
    return v is True or str(v).lower() == "true" or v == 1


def mmlu_lenient(text):
    t = text or ""
    t_nostar = t.replace("*", "")
    for name, pat, g in MMLU_PATTERNS:
        src = t_nostar if name == "authors" else t
        m = list(pat.finditer(src))
        if m:
            val = m[-1].group(g).upper().strip()
            if val in "ABCD" and len(val) == 1:
                return val, name
    lines = [l for l in t.strip().splitlines() if l.strip()]
    if lines:
        m = _LONE.match(lines[-1])
        if m:
            return m.group(1).upper(), "lone_letter"
    return None, None


def countdown_lenient(ds, text, datapoint, finished):
    """Rescue only FINISHED answers without a valid \\boxed{}: take the LAST equation whose right-hand
    side is the target and validate its left-hand side with the authors' validator. Unfinished
    generations are never rescued: enumerating loops contain mis-evaluated lines such as
    '34 - 48 + 48 = 2' whose left side happens to validate (first version of this rescorer
    counted those; caught in the base-model sanity check, 2026-09-28)."""
    if truthy_eval(ds, text, datapoint):
        return True
    if not finished:
        return False
    t = (text or "").replace("×", "*").replace("÷", "/").replace("\\times", "*").replace("\\div", "/")
    for x_list, _, target in datapoint["targets"]:
        last = None
        for m in re.finditer(r"([\d\s()+\-*/]+?)=\s*" + re.escape(str(target)) + r"(?!\d)", t):
            last = m.group(1).strip()
        # also the final statement "... the equation ... is: <expr>" (SFT-on-MMLU models write the
        # final expression after 'is:' without \\boxed{} and without '= target')
        m2 = re.findall(r"(?:equation|expression|answer)[^\n:]{0,80}\bis:?\s*\$?\\?\(?\s*([\d\s()+\-*/]{3,}?)\s*\$?\.?\s*(?:\n|$)", t, re.I)
        if m2 and (last is None or t.rfind(m2[-1]) > t.rfind(last)):
            last = m2[-1].strip()
        if last is None:
            return False
        try:
            if not ds.validate_expression(x_list, last, target):
                return False
        except Exception as e:
            _err("countdown_validate", e)
            return False
    return True


def truthy_eval(ds, text, datapoint):
    try:
        return bool(ds.reward_fn(text, datapoint))
    except Exception as e:
        _err("countdown_reward", e)
        return False


def math_lenient(ds, text, datapoint):
    """Authors' parser first; then the content of the LAST 'answer is: X' / 'Final answer: X'
    statement (SFT-on-MMLU models end math answers with the MMLU format 'The answer is: X')."""
    try:
        if ds.reward_fn(ds.parse_output_text(text), datapoint):
            return True
    except Exception as e:
        _err("math_reward_parsed", e)
    for pat in (r"answer is[:\s]*\$?([^\n$]+?)\$?\s*\.?\s*(?:\n|$)", r"Final answer[:\s]*\$?([^\n$]+?)\$?\s*\.?\s*(?:\n|$)"):
        m = re.findall(pat, text or "", re.I)
        if m:
            cand = m[-1].strip().rstrip(".").strip()
            try:
                if ds.reward_fn(cand, datapoint):
                    return True
            except Exception as e:
                _err("math_reward_lenient", e)
    return False


def dataset_obj(name):
    """Scoring-only instance of the authors' dataset class (no tokenisation)."""
    import core.data as D
    cls = {"countdown": D.CountdownDataset, "math": D.MATHDataset}[name]
    obj = cls.__new__(cls)
    if name == "math":
        # set in MATHDataset.__init__ (skipped here); missing it made every lenient MATH check raise
        # and, with the exception swallowed, silently return False (bug found 2026-09-28)
        from core.evaluation.math_utils import normalize_final_answer
        obj.normalize_final_answer = normalize_final_answer
    return obj


ERRORS = {}


def _err(where, e):
    k = f"{where}:{type(e).__name__}"
    ERRORS[k] = ERRORS.get(k, 0) + 1


def score(eval_dir, name, dump=0):
    p = os.path.join(eval_dir, name, "eval", "predictions.jsonl")
    if not os.path.exists(p):
        return None
    rows = load(p)
    n = len(rows)
    strict = [truthy(r.get("correct")) for r in rows]
    fin = [r.get("finish_reason") == "stop" for r in rows]
    out = {"n": n, "strict": sum(strict) / n, "finished": sum(fin) / n,
           "acc_finished": sum(s for s, f in zip(strict, fin) if f) / max(1, sum(fin))}
    lenient, tiers, dumps = [], {}, []
    ds = dataset_obj(name) if name in ("countdown", "math") else None
    for r, s in zip(rows, strict):
        text = r.get("output_text") or ""
        dp = r.get("datapoint", {})
        if name == "mmlu":
            pred, tier = mmlu_lenient(text)
            gold = dp.get("targets", [None])[0] if isinstance(dp, dict) else None
            ok = pred is not None and pred == gold
            if tier:
                tiers[tier] = tiers.get(tier, 0) + 1
        elif name == "countdown":
            ok = s or countdown_lenient(ds, text, dp, r.get("finish_reason") == "stop")
        elif name == "math":
            ok = s or math_lenient(ds, text, dp)
        else:
            ok = s
        lenient.append(ok)
        if ok and not s and len(dumps) < dump:
            dumps.append({"dataset": name, "idx": dp.get("datapoint_idx") if isinstance(dp, dict) else None,
                          "tail": text[-400:]})
    out["lenient"] = sum(lenient) / n
    out["lenient_only"] = sum(1 for a, b in zip(lenient, strict) if a and not b) / n
    out["strict_only"] = sum(1 for a, b in zip(lenient, strict) if b and not a) / n
    if tiers:
        out["lenient_tiers"] = tiers
    out["correct_strict"] = [int(x) for x in strict]
    out["correct_lenient"] = [int(x) for x in lenient]
    out["finished_vec"] = [int(x) for x in fin]
    return out, dumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dirs", nargs="+")
    ap.add_argument("--datasets", default="mmlu,countdown,math,ifeval_verify")
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--out", default=None, help="json with per-dataset summaries (no vectors)")
    a = ap.parse_args()
    allres = {}
    for d in a.eval_dirs:
        tag = os.path.basename(d.rstrip("/"))
        allres[tag] = {}
        for name in a.datasets.split(","):
            r = score(d, name, a.dump)
            if r is None:
                continue
            res, dumps = r
            allres[tag][name] = {k: v for k, v in res.items() if not isinstance(v, list)}
            print(f"{tag:28s} {name:14s} " + " ".join(
                f"{k}={v:.4f}" for k, v in res.items() if isinstance(v, float)) +
                (f" tiers={res['lenient_tiers']}" if "lenient_tiers" in res else ""))
            for x in dumps:
                print("   LENIENT-ONLY", json.dumps(x)[:500])
    if ERRORS:
        print("SCORING EXCEPTIONS (counted, not silent):", ERRORS)
    if a.out:
        json.dump(allres, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
