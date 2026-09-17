#!/bin/bash
# recompute the post-run cumulative KL for E-v4-4 runs whose in-job cum_kl OOMed on MIG
#SBATCH --job-name=spec-e4fix
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
for OUT in $RUNS/v4_*_lrscale_*; do [ -d $OUT/ckpt_000300 ] || continue; R=$(basename $OUT)
  for REF in offtask task; do [ -f $PROJ/results/v4/result_A/cum_kl_${REF}_e4/$R/cum_kl_runs.csv ] && continue
    $PY analysis_v4/cum_kl.py --ckpts $OUT/ckpt_000300 --reference $REF --n-prompts 64 --out $PROJ/results/v4/result_A/cum_kl_${REF}_e4/$R; done; done
echo E4FIX_DONE
