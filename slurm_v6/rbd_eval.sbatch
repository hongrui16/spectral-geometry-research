#!/bin/bash
# v6 stage 0: authors' evaluation (core.evaluation.run, vLLM, greedy, 4096 tokens) of one model on
# the four retention/task sets. Generations are saved under $RUNS/eval/<tag>/<dataset>/eval for
# later re-scoring. Env: MODEL (HF id or ckpt dir), TAG.
#   bash slurm_v5/submit_limited.sh "sbatch --export=ALL,MODEL=Qwen/Qwen2.5-1.5B-Instruct,TAG=base_qwen15 slurm_v6/rbd_eval.sbatch"
#SBATCH --job-name=v5-rbdeval
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v6/common.sh
M=$MODEL; [ -d "$M" ] || M=$(snap $MODEL)
# 2026-09-29: single-GPU jobs on contrib-gpuq can land on a MIG slice even when an A100 is
# requested; vLLM 0.11 then fails parsing CUDA_VISIBLE_DEVICES=MIG-<uuid> as an int. Inside the
# job the allocated slice is the only visible device, so index 0 addresses it.
if [[ "$CUDA_VISIBLE_DEVICES" == MIG-* ]]; then
  echo "MIG slice $CUDA_VISIBLE_DEVICES -> CUDA_VISIBLE_DEVICES=0"; export CUDA_VISIBLE_DEVICES=0
fi
nvidia-smi -L
echo "=== rbd eval $TAG ($REV, upstream $(cat $RBD/UPSTREAM_REV)) model=$M ==="
cd $RBD/code
for ds in ${DS_LIST:-mmlu countdown ifeval_verify math}; do
  $PY -m core.evaluation.run --model_name "$M" --dataset_name $ds --run_dir $RUNS/eval/$TAG \
      --batch_size 128 --model_module_path core.vllm_utils.vLLMCausalLM --dataset_split eval \
      --temperature 0.0 2>&1 | grep -vE "it/s\]|%\|" | tail -25
done
echo "RBD_EVAL_DONE $TAG"
