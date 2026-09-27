#!/bin/bash
# v5 ad-hoc diagnostic: run a command given as $CMD on one MIG slice.
#SBATCH --job-name=v5-diag
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v5/common.sh
echo "=== diag ($REV) on $(hostname): $CMD ==="
eval "$CMD"
echo DIAG_DONE
