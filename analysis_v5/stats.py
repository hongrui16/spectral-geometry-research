"""v5 comparison statistics (docs/unified_paper_document_v5.md part 4, rules 3-4).

Unit of analysis = one training run. Variance = empirical between-run variance of
the metric (v3/v4 used a binomial SE on the pooled item count, which is wrong: all
runs are scored on the same items, and it ignores training variance).

- welch(a, b)            independent groups
- paired(a, b)           same seeds / same prompt streams, matched by position
- tost(...)              equivalence within +-margin (two one-sided tests)
- verdict(...)           'different' / 'equivalent' / 'inconclusive' (the ONLY
                         allowed wording for a comparison; see rule 4)
- mdd(sd, n1, n2)        minimum detectable difference (two-sided alpha, power)

No scipy in the env: the t CDF is the regularized incomplete beta (Lentz CF).

CLI:
  python analysis_v5/stats.py --metric retention --margin 0.015 \
      --a $RUNS/v5_sft_dense_lr1e-6_s* --b $RUNS/v5_sft_offp_lr1e-6_s* [--paired]
"""

import argparse
import json
import math
import os
import statistics


def _betacf(a, b, x, it=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p, df):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _res(diff, se, df, level=0.90):
    t = diff / se if se > 0 else float("inf") * (1 if diff >= 0 else -1)
    p = 2 * (1 - t_cdf(abs(t), df)) if se > 0 else 0.0
    q = t_ppf(0.5 + level / 2, df)
    return {"diff": diff, "se": se, "df": df, "t": t, "p": p,
            "ci_level": level, "ci": (diff - q * se, diff + q * se)}


def welch(a, b, level=0.90):
    """diff = mean(b) - mean(a)."""
    na, nb = len(a), len(b)
    assert na >= 2 and nb >= 2, "need >= 2 runs per group"
    va, vb = statistics.variance(a), statistics.variance(b)
    se2 = va / na + vb / nb
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if se2 > 0 else na + nb - 2
    return _res(statistics.mean(b) - statistics.mean(a), math.sqrt(se2), df, level)


def paired(a, b, level=0.90):
    assert len(a) == len(b) >= 2, "paired needs equal-length groups (matched seeds)"
    d = [y - x for x, y in zip(a, b)]
    return _res(statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d)), len(d) - 1, level)


def tost(res, margin):
    """Equivalence p-value for |true diff| < margin given a welch/paired result."""
    if res["se"] == 0:
        return 0.0 if abs(res["diff"]) < margin else 1.0
    p_lo = 1 - t_cdf((res["diff"] + margin) / res["se"], res["df"])
    p_hi = t_cdf((res["diff"] - margin) / res["se"], res["df"])
    return max(p_lo, p_hi)


def verdict(a, b, margin, is_paired=False, alpha=0.05):
    """90% CI inside +-margin (TOST at alpha=0.05) -> equivalent; two-sided p < alpha -> different."""
    r = paired(a, b) if is_paired else welch(a, b)
    r["p_tost"] = tost(r, margin)
    r["margin"] = margin
    if r["p_tost"] < alpha:
        r["verdict"] = "equivalent"
    elif r["p"] < alpha:
        r["verdict"] = "different"
    else:
        r["verdict"] = "inconclusive"
    return r


def mdd(sd, n1, n2, alpha=0.05, power=0.8):
    """Smallest true difference detected with the given power (two-sided t, pooled df)."""
    df = n1 + n2 - 2
    se = sd * math.sqrt(1 / n1 + 1 / n2)
    return (t_ppf(1 - alpha / 2, df) + t_ppf(power, df)) * se


def read_metric(run, metric):
    """metric: gsm8k | gsm8k_flex | retention | <task> (retention-suite task acc)."""
    if metric.startswith("gsm8k"):
        d = json.load(open(os.path.join(run, "eval_gsm8k.json")))
        return d["pass@1_flex" if metric == "gsm8k_flex" else "pass@1"]
    d = json.load(open(os.path.join(run, "eval_retention.json")))
    return d["retention"] if metric == "retention" else d["tasks"][metric]["acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--a", nargs="+", required=True, help="reference group runs")
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--paired", action="store_true")
    a = ap.parse_args()
    A = [read_metric(r, a.metric) for r in sorted(a.a)]
    B = [read_metric(r, a.metric) for r in sorted(a.b)]
    r = verdict(A, B, a.margin, a.paired)
    sd = statistics.stdev(A + B)
    print(f"{a.metric}: A n={len(A)} mean={statistics.mean(A):.4f}  B n={len(B)} mean={statistics.mean(B):.4f}")
    print(f"B-A = {r['diff']:+.4f}  90% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  p={r['p']:.3f}  "
          f"p_tost(+-{a.margin})={r['p_tost']:.3f}  -> {r['verdict']}")
    print(f"MDD (80% power, pooled sd {sd:.4f}) = {mdd(sd, len(A), len(B)):.4f}")


if __name__ == "__main__":
    main()
