#!/bin/bash
# E-v4-2: full MMLU (14042 items, SE 0.004) on every finished 300-step checkpoint. ~12 min each on MIG.
# Array index = line number in the sorted list of $RUNS/v3_*/ckpt_000300 (and v4_* later). Lock-claim per ckpt.
#   sbatch --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=0-44 slurm_v4/mmlu_full.sbatch
#SBATCH --job-name=spec-mmluF
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
OUTDIR=$PROJ/results/v4/result_A/mmlu_full; mkdir -p $OUTDIR
CK=$(ls -d $RUNS/v3_*/ckpt_000300 $RUNS/v4_*/ckpt_000300 2>/dev/null | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$CK" ] && { echo "no ckpt for index $SLURM_ARRAY_TASK_ID"; exit 0; }
RUN=$(basename $(dirname $CK)); OUT=$OUTDIR/$RUN.json
[ -f "$OUT" ] && { echo "SKIP: $OUT exists"; exit 0; }
if ! mkdir "$OUT.lock" 2>/dev/null; then echo "SKIP: $OUT claimed"; exit 0; fi
echo "=== full MMLU $RUN ==="
$PY scripts_v4/eval_mmlu.py --model $CK --limit 14042 --out $OUT && rm -rf "$OUT.lock" && echo "MMLUF_DONE $RUN"
