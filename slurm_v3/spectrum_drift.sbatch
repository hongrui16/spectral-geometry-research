#!/bin/bash
#SBATCH --job-name=spec-sdrift
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
$PY analysis_v3/spectrum_drift.py --runs-root $RUNS --out $PROJ/results/v3/result_A/spectrum_drift.csv \
  v3_rlvr_full_lr0.1_s0 v3_rlvr_full_lr0.1_s1 v3_rlvr_frame_matched_lr0.1_s0 v3_rlvr_exact_iso_lr0.1_s0 v3_rlvr_exact_iso_lr0.1_s1 \
  v3_rlvr_spectrum_matched_lr0.1_s0 v3_sft_full_lr0.1_s0 v3_sft_frame_matched_lr0.1_s0 v3_sft_exact_iso_lr0.1_s0 v3_sft_exact_iso_lr0.1_s1
