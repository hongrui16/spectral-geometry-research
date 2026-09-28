"""v5 primary forgetting readout from eval_retention json files (docs/unified_paper_document_v5.md).

retention_v2 = mean(mmlu prefill, arc prefill, hellaswag acc_norm, triviaqa acc_contains)
  - triviaqa acc_contains is recomputed from the saved predictions when the json predates the
    field (evaluations before 2026-09-27 21:xx); exact match is format-sensitive: SFT models
    answer in full sentences that contain the right alias.
  - also reports the three likelihood-only tasks (mmlu, arc, hellaswag) as 'lik3', which has no
    generation in it at all, and mmlu_gen accuracy among terminated answers when available.

  python analysis_v5/retention.py results/v5/result_A/base results/v5/result_A/v5p_* [--step final|100|...]
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_v5"))

_ALIASES = None


def aliases():
    global _ALIASES
    if _ALIASES is None:
        from eval_retention import load_triviaqa
        _ALIASES = [a for _, a in load_triviaqa(1000)]
    return _ALIASES


def score(path):
    from eval_retention import alias_contained
    d = json.load(open(path))
    t = d["tasks"]
    tq = t["triviaqa"]
    if "acc_contains" in tq:
        tq_c = tq["acc_contains"]
    else:
        al = aliases()
        tq_c = sum(alias_contained(p, al[i]) for i, p in enumerate(tq["preds"])) / len(tq["preds"])
    out = {"mmlu": t["mmlu"]["acc"], "arc": t["arc"]["acc"], "hellaswag": t["hellaswag"]["acc"],
           "triviaqa_em": tq["acc"], "triviaqa_contains": tq_c}
    out["lik3"] = (out["mmlu"] + out["arc"] + out["hellaswag"]) / 3
    out["retention_v2"] = (out["mmlu"] + out["arc"] + out["hellaswag"] + tq_c) / 4
    if "mmlu_gen" in t:
        g = t["mmlu_gen"]
        out["mmlu_gen"] = g["acc"]
        out["mmlu_gen_term"] = g.get("term_frac")
        out["mmlu_gen_acc_extracted"] = g["acc"] / g["extract_frac"] if g.get("extract_frac") else None
    return out


def files(run, step):
    if step == "final":
        f = os.path.join(run, "eval_retention.json")
        return [f] if os.path.exists(f) else []
    return sorted(glob.glob(os.path.join(run, f"eval_step{int(step):06d}_retention.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--step", default="final")
    a = ap.parse_args()
    for r in a.runs:
        for f in files(r, a.step):
            s = score(f)
            print(f"{os.path.basename(r.rstrip('/')):32s} " + " ".join(
                f"{k}={v:.4f}" for k, v in s.items() if isinstance(v, float)))


if __name__ == "__main__":
    main()
