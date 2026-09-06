#!/bin/bash
#SBATCH --job-name=spec-evalbase
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm/common.sh

$PY scripts/eval_gsm8k.py --model Qwen/Qwen3.5-0.8B --limit 200 \
  --out $RUNS/eval_base_0.8b.json
echo "EVALBASE_DONE"
