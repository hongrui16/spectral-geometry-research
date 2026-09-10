# Scaling Guide — running larger Qwen3.5 models

For collaborators reproducing Phase 1 / E1 on bigger checkpoints.

## Model family (dense — use these)
| Model | Params | Single-GPU trainable with this repo? |
|---|---|---|
| Qwen3.5-0.8B | 0.87B | YES (fp32, run by Author A, fits a 40GB slice) |
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

## Recipe: 9B on 2-4x 80GB — `--device-map auto` (RECOMMENDED, zero extra code)
Built in. Naive model parallelism: layers are sharded across all visible
GPUs, each shard's optimizer state lives with it, math is IDENTICAL to
single-GPU fp32 (no bf16 quantization of H, no FSDP complexity).

    CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --objective rlvr \
      --model Qwen/Qwen3.5-9B --device-map auto \
      --seqs-per-microbatch 1 --save-every 25 --out <runs>/p1_9b_rlvr_adamw

Memory: 9B fp32 params 36G + grads 36G + AdamW 72G ≈ 144G -> 2x80GB tight,
3x80GB comfortable (activations + rollout generation headroom). OPD loads
the 27B teacher bf16 (~54G) into the same process — give it 4x80GB total.
Throughput: GPUs run sequentially (pipeline without overlap); expect
~1.5-2x single-GPU step time. Fine for Phase 1 volumes.

FSDP/ZeRO port is NOT needed for this paper; only consider it if 27B full
runs ever become a requirement (27B fp32+AdamW ≈ 432G -> 6x80GB naive MP).

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
