"""CPU tests for the v5 pipeline fixes (docs/unified_paper_document_v5.md part 3 §0).

  HF_HOME=/scratch/rhong5/dataset/hf_home HF_HUB_OFFLINE=1 ~/envs_spectral/bin/python analysis_v5/unit_test_cpu_v5.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts_v5"))

from specgeom_v5.data import (extract_answer, has_format, is_correct, load_sft_examples,
                              reward_fn, split_at_stop, strip_calc)
from analysis_v5.stats import mdd, paired, t_cdf, t_ppf, tost, verdict, welch


def test_extraction_and_reward():
    t = "Step 1: 3*4 = 12.\n#### 1,234."
    assert extract_answer(t, strict=True) == "1234"
    assert extract_answer("so the answer is 7", strict=True) is None
    assert extract_answer("so the answer is 7", strict=False) == "7"
    assert extract_answer("#### $18.00") == "18.00" and is_correct("18.00", "18")
    assert extract_answer("#### -5") == "-5"
    # multiple #### -> last one
    assert extract_answer("#### 3 ... #### 4") == "4"
    # truncated completions score 0 even when the answer is right (v2-v4 hack)
    assert reward_fn("#### 42", "42", terminated=False) == 0.0
    assert reward_fn("#### 42", "42", terminated=True) == 1.0
    # strict: no '####' -> 0; flex keeps the v4 last-number readout
    assert reward_fn("answer 42", "42", terminated=True, strict=True) == 0.0
    assert reward_fn("answer 42", "42", terminated=True, strict=False) == 1.0
    # number spam that v4 rewarded: last number fallback only under flex
    spam = "1 2 3 " * 50 + "42"
    assert reward_fn(spam, "42", terminated=False, strict=False) == 0.0
    assert has_format("x #### 5") and not has_format("x 5")
    # review fixes (2026-09-27): markdown/LaTeX dollar forms, headings, no backtracking
    cases = {"#### **18**": "18", "#### \\$18": "18", "#### -$7": "-7", "#### 18 dollars": "18",
             "####\n18": "18", "#### 1. Compute the total": None, "#### 12. Compute": None,
             "#### 2) Step": None, "#### 1. Compute x\n#### 42": "42", "####  $1,000,000": "1000000"}
    for t, want in cases.items():
        assert extract_answer(t) == want, (t, extract_answer(t), want)
    assert not has_format("#### 1. Compute the total")
    print("extraction/reward OK")


def test_split_at_stop():
    stops = [248044, 248046]  # <|endoftext|> (== pad) and <|im_end|>
    ids, term = split_at_stop([5, 6, 248046, 248044, 248044], stops)
    assert ids == [5, 6, 248046] and term
    # stop token equal to pad is kept (v2-v4 stripped it as padding)
    ids, term = split_at_stop([5, 248044, 248044], stops)
    assert ids == [5, 248044] and term
    ids, term = split_at_stop([5, 6, 7], stops)
    assert ids == [5, 6, 7] and not term
    print("split_at_stop OK")


def test_sft_sources():
    s = "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72"
    assert strip_calc(s) == "Natalia sold 48/2 = 24 clips.\n#### 72"
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"question": "q", "completion": "c #### 1", "gold": "1"}) + "\n")
    rows = load_sft_examples(f.name)
    assert rows[0]["completion"] == "c #### 1"
    clean = load_sft_examples("gsm8k_clean")
    assert len(clean) == 7473 and not any("<<" in r["completion"] for r in clean)
    raw = load_sft_examples("gsm8k_raw")
    assert sum("<<" in r["completion"] for r in raw) > 7000
    print("sft sources OK")


def test_stats():
    assert abs(t_cdf(2.228, 10) - 0.975) < 1e-3 and abs(t_ppf(0.975, 4) - 2.776) < 1e-3
    a, b = [0.50, 0.51, 0.49, 0.50], [0.50, 0.505, 0.495, 0.50]
    r = verdict(a, b, 0.02)
    assert r["verdict"] == "equivalent", r
    r = verdict([0.50, 0.51, 0.49, 0.50], [0.56, 0.57, 0.55, 0.56], 0.02)
    assert r["verdict"] == "different", r
    # tiny n and big noise must not be called equivalent (the v3/v4 error)
    r = verdict([0.40, 0.44], [0.41, 0.38], 0.03)
    assert r["verdict"] == "inconclusive", r
    assert paired([1, 2, 3], [2, 3, 4.5])["diff"] > 0
    assert mdd(0.015, 2, 2) > 0.03 > mdd(0.015, 8, 8) / 2
    print("stats OK")


def test_eval_helpers():
    import eval_retention as er
    assert er._norm_ans("The  Beatles!") == "beatles"
    assert er._norm_ans("Children's") == "children s" and er._norm_ans("Forty-Fourth") == "forty fourth"
    assert er._GEN_ANS.findall("blah Answer: (C). Answer: B") == ["C", "B"]
    assert er._hs_pre("Roof [title] Removing shingles [step] x") == "Roof. Removing shingles x"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    ev = er.Evaluator.__new__(er.Evaluator)
    ev.tok = tok
    ids = [sorted({tok.encode(L, add_special_tokens=False)[0],
                   tok.encode(" " + L, add_special_tokens=False)[0]}) for L in er.LETTERS]
    assert ids[0] == [32, 357], ids[0]          # 'A', ' A'
    p = er.chat(tok, er.mc_user("Q?", ["x", "y"]), prefill="Answer:")
    assert p.endswith("</think>\n\nAnswer:") and "A. x\nB. y" in p
    items = er.load_arc(0)
    assert len(items) == 1172 and all(0 <= g < len(o) for _, o, g in items)
    print("eval helpers OK")


def test_offtask_reference():
    from transformers import AutoTokenizer
    from analysis_v5.kl_probe import build_offtask_reference
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    ids, attn, ans = build_offtask_reference(tok, 4, 0, "cpu")
    ids2, _, ans2 = build_offtask_reference(tok, 4, 0, "cpu", answer_only=True)
    for j in range(4):
        L = int(attn[j].sum())
        last = tok.decode([int(ids[j, L - 1])])
        assert last.strip() in "ABCD", last            # gold letter is the final token
        assert bool(ans[j, L - 1]) and int(ans2[j].sum()) == 1 and bool(ans2[j, L - 1])
        assert "Answer:" in tok.decode(ids[j, :L - 1].tolist())[-10:]
    print("offtask reference OK")


def test_lr_scales_param_weighted():
    rows = [dict(matrix="model.layers.0.mlp.gate_proj.weight", m=3584, n=1024, share_off=0.1, share_task=0.4),
            dict(matrix="model.layers.0.self_attn.o_proj.weight", m=1024, n=2048, share_off=0.4, share_task=0.1),
            dict(matrix="model.layers.0.self_attn.q_proj.weight", m=4096, n=1024, share_off=0.2, share_task=0.2)]
    d = tempfile.mkdtemp()
    c = os.path.join(d, "a.csv")
    with open(c, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = os.path.join(d, "s.json")
    subprocess.run([sys.executable, os.path.join(ROOT, "analysis_v5/make_lr_scales.py"), c,
                    "--out", out], check=True, capture_output=True)
    fac = json.load(open(out))
    size = {r["matrix"]: r["m"] * r["n"] for r in rows}
    wmean = sum(fac[k] * size[k] for k in fac) / sum(size.values())
    assert abs(wmean - 1) < 1e-9, wmean
    print("lr scales OK (param-weighted mean 1)")


def test_bf16_visibility_logic():
    w0 = torch.tensor([0.02, 0.02, 1e-4], dtype=torch.bfloat16).float()
    w1 = w0 + torch.tensor([1e-6, 1e-4, 1e-6])
    vis = (w1.bfloat16() != w0.bfloat16())
    assert vis.tolist() == [False, True, True], vis   # small drift on a large weight is invisible
    print("bf16 visibility OK")


def test_triviaqa_alias_consistency():
    """Our normalisation must map the canonical answer into normalized_aliases (review check)."""
    import eval_retention as er
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=0)
    ds = ds.select(range(1000))
    miss = sum(er._norm_ans(r["answer"]["value"]) not in set(r["answer"]["normalized_aliases"]) for r in ds)
    assert miss <= 5, miss
    print(f"triviaqa canonical-answer misses {miss}/1000 OK")


def test_rlvr_temperature_scaling():
    import re as _re
    src = open(os.path.join(ROOT, "scripts_v5/train.py")).read()
    assert _re.search(r"logits = logits / self\.args\.temperature", src), "rlvr logits not tempered"
    print("rlvr temperature scaling present OK")


if __name__ == "__main__":
    test_extraction_and_reward()
    test_split_at_stop()
    test_sft_sources()
    test_stats()
    test_eval_helpers()
    test_offtask_reference()
    test_lr_scales_param_weighted()
    test_bf16_visibility_logic()
    test_triviaqa_alias_consistency()
    test_rlvr_temperature_scaling()
    print("ALL V5 CPU TESTS PASS")
