#!/bin/bash
# E-v3-0c (A side): per-step functional step size of full vs projected updates on the
# v3 SFT smoke captures (same prompt stream/seed, step 8). ~20 min on one A100.
# Submit: sbatch --partition=contrib-gpuq --gres=gpu:A100.80gb:1 slurm_v3/step_kl.sbatch
#SBATCH --job-name=spec-stepkl
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
$PY analysis_v3/step_kl.py --step 8 --out $PROJ/results/v3/result_A/step_kl_smoke \
  --runs $RUNS/smoke_v3_sft_none_s1 $RUNS/smoke_v3_sft_spectrum_matched_s3 $RUNS/smoke_v3_sft_random_ext_s1
echo "STEP_KL_DONE"
