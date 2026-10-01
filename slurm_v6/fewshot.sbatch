#!/bin/bash
# v6 decisive test: k-shot format reminder (analysis_v6/fewshot_eval.py). Env: MODEL, TAG, ZERO (zero-shot eval dir)
#SBATCH --job-name=v5-fewshot
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v6/common.sh
M=$MODEL; [ -d "$M" ] || M=$(snap $MODEL)
cd $RBD/code
$PY $PROJ/analysis_v6/fewshot_eval.py --model "$M" --tag $TAG --base-eval $RUNS/eval/base_qwen15 \
    --zero-eval $ZERO --out $PROJ/results/v6/result_A/fewshot_${TAG}.json 2>&1 | grep -vE "it/s\]|%\|"
echo "FEWSHOT_DONE $TAG"
