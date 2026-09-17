#!/bin/bash
# A3(b): per-matrix off-task/task attribution on representative final checkpoints (186 x 2 refs x forward each, ~30-40 min per ckpt on A100).
#   sbatch --partition=contrib-gpuq --gres=gpu:A100.80gb:1 --array=0-5 slurm_v4/matrix_attrib.sbatch
#SBATCH --job-name=spec-attrib
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
R=(v3_sft_full_lr0.1_s0 v3_rlvr_full_lr0.1_s0 v3_sft_full_lr0.33_s0 v3_sft_spectrum_matched_lr0.1_s0 v3_sft_full_lr0.033_s0 v3_rlvr_full_lr0.33_s0)
RUN=${R[$SLURM_ARRAY_TASK_ID]}
$PY analysis_v4/matrix_attrib.py --ckpt $RUNS/$RUN/ckpt_000300 --out $PROJ/results/v4/result_A/matrix_attrib/$RUN.csv
