#!/bin/bash
# E0/E2 finite-step KL probe on the frozen 0.8B checkpoint (docs v2 §11 E2, §12.4).
# Submit: sbatch slurm_v2/kl_probe.sbatch     (any GPU partition; override with --partition/--gres)
#SBATCH --job-name=spec-klp
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v2/common.sh
OUT=$PROJ/results/v2/result_A/kl_probe_0.8b
$PY analysis_v2/kl_probe.py --out $OUT --n-prompts 32 --batch 8 --etas 1e-3,3e-3,1e-2,3e-2
echo "KL_PROBE_DONE"
