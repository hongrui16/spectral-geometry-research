#!/bin/bash
# Phase 1: three paradigms x {adamw, muon}. Submit as:
#   sbatch --array=0-3 slurm/phase1.sbatch      (main: sft/opd/rlvr adamw + rlvr muon)
#   sbatch --array=4-5 slurm/phase1.sbatch      (extra: sft muon, opd muon)
#SBATCH --job-name=spec-p1
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm/common.sh

CONFIGS=(
  "sft adamw" "opd adamw" "rlvr adamw" "rlvr muon" "sft muon" "opd muon"
)
read -r OBJ OPT <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
OUT=$RUNS/phase1_${OBJ}_${OPT}
echo "=== phase1 $OBJ $OPT -> $OUT ==="

# RL-standard lr for policy gradient: 1e-5 (SFT parity) collapses the policy
# (reward 23%->2% by step 40, incoherent rollouts). SFT/OPD keep 1e-5.
EXTRA=""
if [ "$OBJ" = "rlvr" ]; then EXTRA="--lr 2e-6 --muon-lr 2e-5"; fi

$PY scripts/train.py --objective $OBJ --optimizer $OPT $EXTRA \
  --steps 500 --save-every 10 --prompts-per-step 8 --rollouts 8 \
  --max-new-tokens 384 \
  --out $OUT

$PY analysis/compute_metrics.py $OUT
echo "PHASE1_DONE $OBJ $OPT"
