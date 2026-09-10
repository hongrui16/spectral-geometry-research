#!/bin/bash
# GPU smoke test of the four v2 intervention modes on 0.8B SFT, 20 steps each,
# followed by the v2 metrics self-check. Must pass before 作者B starts P3.
# Submit: sbatch --array=0-3 slurm_v2/smoke.sbatch
#SBATCH --job-name=spec-smk2
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v2/common.sh
MODES=(spectrum_matched frame_matched random_ext exact_iso)
INT=${MODES[$SLURM_ARRAY_TASK_ID]}
OUT=$RUNS/smoke_v2_sft_$INT
echo "=== smoke_v2 $INT -> $OUT ($(git -C $PROJ rev-parse --short HEAD)) ==="

$PY scripts_v2/train.py --objective sft --steps 20 --save-every 10 \
  --prompts-per-step 8 --max-new-tokens 384 --seed 0 \
  --intervention $INT --out $OUT
$PY analysis_v2/compute_metrics.py $OUT --out-root $PROJ/results/v2/result_A
$PY scripts_v2/summary.py $OUT
echo "SMOKE_V2_DONE $INT"
