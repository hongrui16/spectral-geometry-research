"""v5 stage 1 (Q1): the opponents' descriptive update-geometry statistics, per checkpoint.

Object alignment (docs/unified_paper_document_v5.md part 4 rule 2): these follow the
definitions of The Path Not Taken (2511.08567) and Jin et al. (2509.12235), NOT the
v2-v4 "spectral component" (diagonal of U^T H V, singular-value change only).

For each 2D decoder matrix, with W0 = U S V^T (base, fp64) and dW = W_t - W0:
  rot_k        principal-subspace rotation: mean sin^2 of the principal angles between the
               top-k left singular subspaces of W0 and W_t (and right: rot_k_right);
               k = ceil(frac * rank) for frac in --fracs.
  nss          normalized spectral shift ||s(W_t) - s(W0)|| / ||s(W0)||.
  blk_*_k      share of ||dW||^2 in the four blocks of W0's top-k frame:
               PP = U_k U_k^T dW V_k V_k^T, PQ (top-left/rest-right), QP, QQ (both rest);
               random expectation: PP (k/m)(k/n), QQ (1-k/m)(1-k/n) -> enrichment = share / expectation.
  pw_enrich    principal-weight overlap: M = top-alpha fraction of |W0_k| entries (rank-k
               reconstruction, k = --pw-frac of rank, 2511.08567 "principal weights");
               energy share of dW inside M divided by alpha (1 = no preference).
Aggregates are parameter-weighted over matrices; per-matrix rows are also written.

  python analysis_v5/principal_geometry.py --ckpts $RUNS/<run>/ckpt_000300 ... --out results/v5/result_A/geom
"""

import argparse
import glob
import json
import math
import os
import re

import torch
from safetensors import safe_open

PAT = re.compile(r"layers\.\d+\.(self_attn|mlp|linear_attn)\.(\w+)\.weight$")


def mats(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if PAT.search(k) and not k.startswith("mtp.") and ".mtp." not in k \
                        and "visual" not in k and len(sf.get_slice(k).get_shape()) == 2:
                    out[k[k.index("layers."):]] = sf.get_tensor(k)
    return out


def sin2_mean(A, B):
    """mean sin^2 of principal angles between column spaces of orthonormal A, B (m x k)."""
    s = torch.linalg.svdvals(A.T @ B).clamp(max=1.0)
    return float((1 - s ** 2).mean())


@torch.no_grad()
def matrix_stats(W0, Wt, fracs, pw_frac, pw_alpha, base_svd=None):
    W0, Wt = W0.double(), Wt.double()
    dW = Wt - W0
    e = float((dW ** 2).sum())
    U, S, Vh = base_svd if base_svd is not None else torch.linalg.svd(W0, full_matrices=False)
    Ut, St, Vth = torch.linalg.svd(Wt, full_matrices=False)
    m, n = W0.shape
    r = S.numel()
    row = {"m": m, "n": n, "dW_fro": math.sqrt(e), "rel_dW": math.sqrt(e) / float(W0.norm()),
           "nss": float((St - S).norm() / S.norm())}
    for fr in fracs:
        k = max(1, math.ceil(fr * r))
        Uk, Vk = U[:, :k], Vh[:k].T
        row[f"rot_{fr}"] = sin2_mean(Uk, Ut[:, :k])
        row[f"rot_{fr}_right"] = sin2_mean(Vk, Vth[:k].T)
        A = Uk.T @ dW            # k x n
        PP = float(((A @ Vk) ** 2).sum())
        PQ = float((A ** 2).sum()) - PP           # top-left rows, rest right
        B = dW @ Vk              # m x k
        QP = float((B ** 2).sum()) - PP
        QQ = e - PP - PQ - QP
        exp = {"PP": (k / m) * (k / n), "PQ": (k / m) * (1 - k / n),
               "QP": (1 - k / m) * (k / n), "QQ": (1 - k / m) * (1 - k / n)}
        for name, val in (("PP", PP), ("PQ", PQ), ("QP", QP), ("QQ", QQ)):
            sh = val / e if e > 0 else float("nan")
            row[f"blk_{name}_{fr}"] = sh
            row[f"blk_{name}_{fr}_enrich"] = sh / exp[name] if e > 0 else float("nan")
    kp = max(1, math.ceil(pw_frac * r))
    Wk = (U[:, :kp] * S[:kp]) @ Vh[:kp]
    thr = torch.quantile(Wk.abs().flatten().float()[torch.randperm(Wk.numel())[:2_000_000]],
                         1 - pw_alpha).double()
    M = Wk.abs() >= thr
    row["pw_enrich"] = (float((dW[M] ** 2).sum()) / e) / float(M.float().mean()) if e > 0 else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--fracs", default="0.01,0.05,0.1")
    ap.add_argument("--pw-frac", type=float, default=0.01)
    ap.add_argument("--pw-alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fracs = [float(x) for x in a.fracs.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = a.base or glob.glob(os.path.expandvars(
        "$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
    B = {k: v.to(dev) for k, v in mats(base).items()}
    svd0 = {k: torch.linalg.svd(v.double(), full_matrices=False) for k, v in B.items()}
    os.makedirs(a.out, exist_ok=True)
    import pandas as pd
    summ = []
    for ck in a.ckpts:
        run = os.path.basename(os.path.dirname(ck.rstrip("/")))
        C = mats(ck)
        rows = []
        for k in sorted(set(B) & set(C)):
            r = matrix_stats(B[k], C[k].to(dev), fracs, a.pw_frac, a.pw_alpha, svd0[k])
            r.update(run=run, matrix=k, type=PAT.search(k).group(2))
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out, f"{run}.csv"), index=False)
        w = df.m * df.n
        agg = {"run": run, "ckpt": ck, "n_matrices": len(df),
               "dW_fro_total": float((df.dW_fro ** 2).sum() ** 0.5)}
        for c in df.columns:
            if c.startswith(("rot_", "blk_", "nss", "pw_enrich", "rel_dW")):
                agg[c] = float((df[c] * w).sum() / w.sum())
        summ.append(agg)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in agg.items()
                          if not k.startswith("blk_") or k.endswith("_enrich")}), flush=True)
    pd.DataFrame(summ).to_csv(os.path.join(a.out, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
