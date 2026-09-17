#!/bin/bash
#SBATCH --job-name=spec-mmluB
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
$PY scripts_v4/eval_mmlu.py --model Qwen/Qwen3.5-0.8B --limit 14042 --out $PROJ/results/v4/result_A/mmlu_full/base.json && echo MMLUB_DONE
