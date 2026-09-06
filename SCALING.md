# Scaling Guide — running larger Qwen3.5 models

For collaborators reproducing Phase 1 / E1 on bigger checkpoints.

## Model family (dense — use these)
| Model | Params | Single-GPU trainable with this repo? |
|---|---|---|
| Qwen3.5-0.8B | 0.87B | YES (fp32, done here, even fits 40GB MIG) |
| Qwen3.5-2B | 2.3B | YES on 1x A100.80gb (fp32 weights ~9G + AdamW ~18G) |
| Qwen3.5-4B | ~4B | YES on 1x A100.80gb, borderline — reduce seqs/microbatch to 1 |
| Qwen3.5-9B | ~9B | NO as-is — needs 8-bit optimizer or 2+ GPUs (see below) |
| Qwen3.5-27B | ~27B | NO — needs 4-8 GPU FSDP port |

There is **no dense 20B**; closest are 9B and 27B. Avoid the MoE variants
(35B-A3B etc., arch `qwen3_5_moe`): per-expert matrix tracking is not
implemented and expert routing confounds the G-projection statistics.

## Why fp32 weights matter here
`H_t = W_after - W_before` is captured by subtraction. With bf16 master
weights the update (~1e-4 relative) drowns in quantization noise (~4e-3
relative). Keep fp32 masters, or implement analytic-H (recompute the AdamW
update from optimizer state for the 27 tracked matrices only).

## Recipe: 9B on a single 80GB GPU
1. bf16 weights + bitsandbytes 8-bit AdamW (weights 18G + states 18G + bf16
   grads 18G ≈ 54G).
2. Add fp32 *shadow copies* of the 27 tracked matrices only (~1.5G): update
   them by replaying the optimizer math, capture G/H from the shadows.
3. `--seqs-per-microbatch 1`, `--save-every 25` (capture files scale ~10x).
4. RLVR generation: HF generate will be slow; wire TRL-style vLLM colocate
   or accept ~3-4x wall-clock.

## Recipe: 9B/27B multi-GPU (FSDP2 / ZeRO-2)
- Shard params + optimizer states; gradients reduce-scatter.
- At save steps, gather ONLY the 27 tracked matrices (full-param summon on
  those modules is cheap) for G/H/W capture; everything else stays sharded.
- Per-rollout backward (H3) must run under `no_sync()` on one rank.
- All analysis code (`analysis/`, `specgeom/metrics.py`) is scale-agnostic —
  it reads capture files offline.

## What to replicate at scale (per the paper plan)
Priority order, matching the 0.8B main line:
1. Phase 1, AdamW only, 3 paradigms -> Fig.2 (R_spectrum ordering) + Fig.4
   (rho^Sigma vs rho^frame). This is the core scientific claim.
2. E1 RLVR row ({AdamW, Muon, SSD}) if compute allows.
3. Everything else stays on 0.8B.

## Shared setup
- Env: python 3.10 venv, `pip install torch --index-url .../cu128` then
  `pip install -U transformers trl datasets accelerate math-verify matplotlib pandas tensorboard`
- `HF_HOME` on a shared filesystem; pre-download models + `openai/gsm8k`,
  run jobs with `HF_HUB_OFFLINE=1`.
- Same seeds, same prompt stream (`specgeom/data.py` is deterministic per seed).
