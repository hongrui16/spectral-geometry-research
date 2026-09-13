#!/bin/bash
# v3 batch 3, SFT half (2026-09-13, after SFT batch-2 verdict: H-2 holds, H-1 holds at lr*):
#   E-v3-4  r-dof third seed at lr x1 (1e-5):        spectrum_matched s2, random_ext s2         (tasks 0-1)
#   E-v3-5  s_rel sweep at lr* = 1e-6, seed 0:       spectrum_matched / random_ext x s_rel {3,10} (tasks 2-5)
# Same $OUT.lock claim as batch2; submit the same array under several GPU types:
#   sbatch --partition=gpuq         --gres=gpu:3g.40gb:1 --array=0-5 slurm_v3/batch3_sft.sbatch
#   sbatch --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=0-5 slurm_v3/batch3_sft.sbatch
#SBATCH --job-name=spec-b3sft
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
MODES=(spectrum_matched random_ext spectrum_matched random_ext spectrum_matched random_ext)
SEEDS=(2 2 0 0 0 0)
LRS=(1e-5 1e-5 1e-6 1e-6 1e-6 1e-6)
SCALES=(1 1 3 3 10 10)
NAMES=(lr1_s2 lr1_s2 lr0.1_srel3_s0 lr0.1_srel3_s0 lr0.1_srel10_s0 lr0.1_srel10_s0)
i=$SLURM_ARRAY_TASK_ID
OUT=$RUNS/v3_sft_${MODES[$i]}_${NAMES[$i]}
if ! mkdir "$OUT.lock" 2>/dev/null; then echo "SKIP: $OUT already claimed by $(cat $OUT.lock/job 2>/dev/null)"; exit 0; fi
echo "$SLURM_JOB_ID $SLURMD_NODENAME $SLURM_JOB_GRES" > "$OUT.lock/job"
echo "=== batch3 sft ${MODES[$i]} lr=${LRS[$i]} s_rel=${SCALES[$i]} seed=${SEEDS[$i]} -> $OUT ==="

$PY scripts_v3/train.py --objective sft --steps 300 --save-every 20 --lr ${LRS[$i]} \
  --prompts-per-step 8 --max-new-tokens 384 --seed ${SEEDS[$i]} \
  --intervention ${MODES[$i]} --intervention-scale ${SCALES[$i]} \
  --eval-every 50 --out $OUT
$PY analysis_v3/compute_metrics.py $OUT --out-root $PROJ/results/v3/result_A
$PY scripts_v3/summary.py $OUT
D=$PROJ/results/v3/result_A/$(basename $OUT); mkdir -p $D
cp $OUT/log.jsonl $OUT/args.json $OUT/summary.txt $OUT/eval_*.json $D/ 2>/dev/null
echo "BATCH3_SFT_DONE ${MODES[$i]} ${NAMES[$i]}"
