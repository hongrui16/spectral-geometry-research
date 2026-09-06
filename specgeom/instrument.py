"""Per-step gradient/update capture for tracked weight matrices.

At every `save_every`-th optimizer step, records for each tracked matrix:
  G   raw mean gradient (before optimizer transform)
  G_b per-microbatch gradients (B of them, for SNR estimation, H4)
  H   actual applied update  W_after - W_before  (optimizer output)
  W   weight before the step (basis for U, Sigma, V)
plus optional per-rollout grad-log-pi tensors for RLVR (H3).

Everything is stored in bf16 on CPU, one file per save step:
  <out_dir>/capture/step_XXXXXX.pt
Metric computation (SVD etc.) happens offline in analysis/.
"""

import os
import re
import torch


DEFAULT_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# matrix families for hybrid archs (qwen3_5: some layers full attention, some
# gated linear attention). For each family, early/mid/late layers among the
# layers that actually contain it.
FAMILIES = {
    "self_attn": ("self_attn.q_proj", "self_attn.k_proj",
                  "self_attn.v_proj", "self_attn.o_proj"),
    "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "linear_attn": ("linear_attn.in_proj_qkv", "linear_attn.out_proj"),
}


def select_tracked(model, n_layers: int, types=None, layer_picks=None):
    """Pick early/mid/late layers per matrix family. Returns {param_name: param}.

    Hybrid-arch aware: each family only exists in a subset of layers, so the
    early/mid/late picks are made within that subset.
    """
    layer_of = re.compile(r"\blayers\.(\d+)\.")
    params = {n: p for n, p in model.named_parameters()
              if p.dim() == 2 and layer_of.search(n) and "visual" not in n
              and not n.startswith("mtp.")}
    tracked = {}
    for fam, suffixes in FAMILIES.items():
        fam_layers = sorted({int(layer_of.search(n).group(1)) for n in params
                             if f".{suffixes[0]}.weight" in n})
        if not fam_layers:
            continue
        picks = sorted({fam_layers[min(1, len(fam_layers) - 1)],
                        fam_layers[len(fam_layers) // 2],
                        fam_layers[-2] if len(fam_layers) > 1 else fam_layers[-1]})
        for l in picks:
            for suf in suffixes:
                for n, p in params.items():
                    if n.endswith(f"layers.{l}.{suf}.weight"):
                        tracked[n] = p
    return tracked


class Instrumenter:
    def __init__(self, model, out_dir, n_layers, save_every=10, types=DEFAULT_TYPES,
                 layer_picks=None):
        self.model = model
        self.out_dir = os.path.join(out_dir, "capture")
        os.makedirs(self.out_dir, exist_ok=True)
        self.save_every = save_every
        self.tracked = select_tracked(model, n_layers, types, layer_picks)
        assert self.tracked, "no tracked parameters matched"
        # G_b (SNR) storage is 8x per matrix; restrict to one representative
        # matrix per (family, depth) to keep capture files manageable
        self.snr_names = [n for n in self.tracked
                          if any(s in n for s in
                                 ("q_proj", "down_proj", "in_proj_qkv"))]
        self._record = None
        self._w_before = None

    def is_save_step(self, step: int) -> bool:
        return step % self.save_every == 0

    def names(self):
        return list(self.tracked.keys())

    @torch.no_grad()
    def begin_step(self, step: int):
        """Call at the start of a save step, before any backward."""
        self._record = {"step": step, "mats": {n: {} for n in self.tracked}}

    @torch.no_grad()
    def capture_microbatch_grads(self, tag: str):
        """Copy current p.grad for tracked params under key `tag` (e.g. 'b0'...)."""
        for n, p in self.tracked.items():
            if p.grad is not None:
                self._record["mats"][n].setdefault("G_b", {})[tag] = \
                    p.grad.detach().to(torch.bfloat16).cpu().clone()

    @torch.no_grad()
    def capture_grad(self):
        """Copy the accumulated mean gradient G (call right before optimizer.step)."""
        for n, p in self.tracked.items():
            if p.grad is not None:
                self._record["mats"][n]["G"] = p.grad.detach().to(torch.bfloat16).cpu().clone()

    @torch.no_grad()
    def capture_rollout_grad(self, k: int, names=None):
        """Copy grad-log-pi for rollout k (no advantage weighting) for a subset."""
        names = names or self.names()
        for n in names:
            p = self.tracked[n]
            if p.grad is not None:
                self._record["mats"][n].setdefault("rollout_G", {})[k] = \
                    p.grad.detach().to(torch.bfloat16).cpu().clone()

    @torch.no_grad()
    def set_extra(self, key, value):
        self._record[key] = value

    @torch.no_grad()
    def pre_optimizer(self):
        self._w_before = {n: p.detach().clone() for n, p in self.tracked.items()}
        for n, p in self.tracked.items():
            self._record["mats"][n]["W"] = p.detach().to(torch.bfloat16).cpu().clone()

    @torch.no_grad()
    def post_optimizer(self):
        for n, p in self.tracked.items():
            H = (p.detach() - self._w_before[n])
            self._record["mats"][n]["H"] = H.to(torch.bfloat16).cpu().clone()
        self._w_before = None

    @torch.no_grad()
    def flush(self):
        step = self._record["step"]
        path = os.path.join(self.out_dir, f"step_{step:06d}.pt")
        torch.save(self._record, path)
        self._record = None
        return path
