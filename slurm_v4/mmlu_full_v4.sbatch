#!/bin/bash
# full MMLU for every finished v4 run (method, lr x1 arms, trajectory) that does not have it yet; sequential, ~12 min each.
#SBATCH --job-name=spec-mmluF4
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
OUTDIR=$PROJ/results/v4/result_A/mmlu_full; mkdir -p $OUTDIR
for CK in $(ls -d $RUNS/v4_*/ckpt_000300 2>/dev/null | sort); do
  RUN=$(basename $(dirname $CK)); OUT=$OUTDIR/$RUN.json
  [ -f "$OUT" ] && continue; mkdir "$OUT.lock" 2>/dev/null || continue
  echo "=== full MMLU $RUN ==="; $PY scripts_v4/eval_mmlu.py --model $CK --limit 14042 --out $OUT && echo "MMLUF_DONE $RUN"; rm -rf "$OUT.lock"
done; echo MMLUF4_DONE
