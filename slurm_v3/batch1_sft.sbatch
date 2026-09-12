#!/bin/bash
# v3 batch 1 (gate), SFT half: dense lr sweep {1/3, 1/10, 1/30} x 2 seeds, 300 steps,
# eval every 50 steps. Same commands as docs/TASKS_B_v3.md §1.1 (A runs these on idle
# A100.40gb cards; B runs the RLVR six). ~2 h per run incl. evals.
# Submit: sbatch --partition=gpuq --gres=gpu:A100.40gb:1 --array=0-5 slurm_v3/batch1_sft.sbatch
#SBATCH --job-name=spec-b1sft
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
LRS=(3.33e-6 3.33e-6 1e-6 1e-6 3.33e-7 3.33e-7)
TAGS=(lr0.33 lr0.33 lr0.1 lr0.1 lr0.033 lr0.033)
SEEDS=(0 1 0 1 0 1)
i=$SLURM_ARRAY_TASK_ID
OUT=$RUNS/v3_sft_full_${TAGS[$i]}_s${SEEDS[$i]}
echo "=== batch1 sft lr=${LRS[$i]} seed=${SEEDS[$i]} -> $OUT ==="

$PY scripts_v3/train.py --objective sft --steps 300 --save-every 20 --lr ${LRS[$i]} \
  --prompts-per-step 8 --max-new-tokens 384 --seed ${SEEDS[$i]} \
  --eval-every 50 --out $OUT
$PY analysis_v3/compute_metrics.py $OUT --out-root $PROJ/results/v3/result_A
$PY scripts_v3/summary.py $OUT
# deliver the small files next to the metrics (same layout as B's deliveries)
D=$PROJ/results/v3/result_A/$(basename $OUT); mkdir -p $D
cp $OUT/log.jsonl $OUT/args.json $OUT/summary.txt $OUT/eval_*.json $D/ 2>/dev/null
echo "BATCH1_SFT_DONE ${TAGS[$i]} s${SEEDS[$i]}"
