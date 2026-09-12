#!/bin/bash
# v3 batch 2, SFT half, at SFT lr* = 1e-6 (batch-1 gate, 2026-09-12): frame_matched, exact_iso,
# spectrum_matched, random_ext x 2 seeds, 300 steps, eval every 50. Run by A on idle A100.40gb.
# Submit: sbatch --partition=gpuq --gres=gpu:A100.40gb:1 --array=0-7 slurm_v3/batch2_sft.sbatch
#SBATCH --job-name=spec-b2sft
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
MODES=(frame_matched frame_matched exact_iso exact_iso spectrum_matched spectrum_matched random_ext random_ext)
SEEDS=(0 1 0 1 0 1 0 1)
i=$SLURM_ARRAY_TASK_ID
OUT=$RUNS/v3_sft_${MODES[$i]}_lr0.1_s${SEEDS[$i]}
echo "=== batch2 sft ${MODES[$i]} lr=1e-6 seed=${SEEDS[$i]} -> $OUT ==="

$PY scripts_v3/train.py --objective sft --steps 300 --save-every 20 --lr 1e-6 \
  --prompts-per-step 8 --max-new-tokens 384 --seed ${SEEDS[$i]} \
  --intervention ${MODES[$i]} --intervention-scale 1 \
  --eval-every 50 --out $OUT
$PY analysis_v3/compute_metrics.py $OUT --out-root $PROJ/results/v3/result_A
$PY scripts_v3/summary.py $OUT
D=$PROJ/results/v3/result_A/$(basename $OUT); mkdir -p $D
cp $OUT/log.jsonl $OUT/args.json $OUT/summary.txt $OUT/eval_*.json $D/ 2>/dev/null
echo "BATCH2_SFT_DONE ${MODES[$i]} s${SEEDS[$i]}"
