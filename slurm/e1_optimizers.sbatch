#!/bin/bash
# E1: optimizer x paradigm main table. Submit: sbatch --array=0-8 slurm/e1_optimizers.sbatch
# (seeds added via --export=SEED=n or a second array dimension later)
#SBATCH --job-name=spec-e1
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm/common.sh
SEED=${SEED:-0}

CONFIGS=(
  "sft adamw" "sft muon" "sft ssd"
  "opd adamw" "opd muon" "opd ssd"
  "rlvr adamw" "rlvr muon" "rlvr ssd"
)
read -r OBJ OPT <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
OUT=runs/e1_${OBJ}_${OPT}_s${SEED}
echo "=== e1 $OBJ $OPT seed $SEED -> $OUT ==="

$PY scripts/train.py --objective $OBJ --optimizer $OPT --seed $SEED \
  --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 \
  --max-new-tokens 384 --out $OUT

$PY scripts/eval_gsm8k.py --model $OUT/ckpt_000500 --limit 500 \
  --out $OUT/eval_gsm8k.json
echo "E1_DONE $OBJ $OPT $SEED"
