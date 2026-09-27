"""v5: how much of a checkpoint's weight change survives bf16 rounding.

Training runs fp32 master weights under bf16 autocast, so every matmul sees bf16(W).
Base weights are bf16-exact, so an element whose accumulated drift stays below half a
bf16 ulp is invisible to the forward pass. Reports, over the 2D decoder matrices,
  visible_frac  = #{bf16(W_ckpt) != bf16(W_base)} / #{W_ckpt != W_base}
  changed_frac  = #{W_ckpt != W_base} / #elements
and per matrix type. Writes <run>/bf16_visible.json when --write-run is given.

  python analysis_v5/bf16_visible.py --ckpt $RUNS/<run>/ckpt_000300 [--write-run]
"""
import argparse, collections, glob, json, os, re
import torch
from safetensors import safe_open

PAT = re.compile(r"layers\.\d+\.(self_attn|mlp|linear_attn)\.(\w+)\.weight$")


def tensors(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                m = PAT.search(k)
                if m and not k.startswith("mtp.") and ".mtp." not in k and "visual" not in k \
                        and len(sf.get_slice(k).get_shape()) == 2:
                    out[k[k.index("layers."):]] = sf.get_tensor(k)   # base and ckpt prefixes differ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--write-run", action="store_true")
    a = ap.parse_args()
    base = a.base or glob.glob(os.path.expandvars(
        "$HF_HOME/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
    B, C = tensors(base), tensors(a.ckpt)
    common = sorted(set(B) & set(C))
    assert common, "no matching matrices (base/ckpt key layout differs?)"
    tot = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    for k in common:
        w0, w1 = B[k].float(), C[k].float()
        ch = (w1 != w0)
        vis = (w1.bfloat16() != w0.bfloat16())
        t = PAT.search(k).group(2)
        for c in (tot, by_type[t]):
            c["n"] += w0.numel(); c["changed"] += int(ch.sum()); c["visible"] += int(vis.sum())
    res = {"ckpt": a.ckpt, "n_matrices": len(common),
           "changed_frac": tot["changed"] / tot["n"],
           "visible_frac": tot["visible"] / max(1, tot["changed"]),
           "by_type": {t: {"changed_frac": c["changed"] / c["n"],
                           "visible_frac": c["visible"] / max(1, c["changed"])}
                       for t, c in sorted(by_type.items())}}
    print(json.dumps(res, indent=1))
    if a.write_run:
        json.dump(res, open(os.path.join(os.path.dirname(a.ckpt.rstrip("/")), "bf16_visible.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
