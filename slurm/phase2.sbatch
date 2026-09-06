#!/bin/bash
# Phase 2 interventions (H5/H6). Submit: sbatch --array=0-7 slurm/phase2.sbatch
#SBATCH --job-name=spec-p2
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm/common.sh

# objective intervention q
CONFIGS=(
  "rlvr spectrum_only 0"  "rlvr frame_only 0"
  "sft  spectrum_only 0"  "sft  frame_only 0"
  "rlvr snr_topq 0.1"     "rlvr mag_topq 0.1"
  "sft  snr_topq 0.1"     "sft  mag_topq 0.1"
)
read -r OBJ INT Q <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
OUT=runs/phase2_${OBJ}_${INT}$([ "$Q" != "0" ] && echo _q$Q)
echo "=== phase2 $OBJ $INT q=$Q -> $OUT ==="

$PY scripts/train.py --objective $OBJ --optimizer adamw \
  --steps 300 --save-every 20 --prompts-per-step 8 --rollouts 8 \
  --max-new-tokens 384 --intervention $INT \
  $([ "$Q" != "0" ] && echo --intervention-q $Q) \
  --out $OUT

$PY scripts/eval_gsm8k.py --model $OUT/ckpt_000300 --limit 500 \
  --out $OUT/eval_gsm8k.json
echo "PHASE2_DONE $OBJ $INT"
