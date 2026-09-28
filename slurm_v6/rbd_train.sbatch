#!/bin/bash
# v6 stage 0: run the AUTHORS' training script (scripts/{sft,self_sft,rl}.sh) unchanged except for
# paths / environment: code dir, run dir, local model snapshots, model_name_short, dataset_name.
# Hyperparameters are exactly the authors' script defaults (see results_draft v6 for the
# script-vs-paper discrepancies). 2026-09-28: 4-GPU nodes queue for hours, so the job uses 2 GPUs
# and scales gradient_accumulation_steps so the TOTAL batch is unchanged (SFT: 2x2x64 = 2x4x32 = 256;
# RL: 1 train GPU x 32 = 2 train GPUs x 16 = 32, 1 generation GPU instead of 2).
# Env: ALGO=sft|self_sft|rl  MS=llama-3.2-1b-inst|qwen-2.5-1.5b-inst  DS=<dataset class name>
#SBATCH --job-name=v5-rbdtrain
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v6/common.sh
export PATH=$HOME/envs_py310_new/bin:$PATH
RB=$RUNS/train; mkdir -p $RB
S=$RUNS/scripts/${ALGO}_${MS}_${DS}_$SLURM_JOB_ID.sh; mkdir -p $(dirname $S)
sed -e "s#^conda activate py310n#: # env from slurm_v6/common.sh#" \
    -e "s#root_dir=\"/path/to/your/project/dir\"#root_dir=\"$RBD/code\"#" \
    -e "s#run_base_dir=\"/path/to/your/run/dir\"#run_base_dir=\"$RB\"#" \
    -e "s#/path/to/your/local_models/meta-llama/Llama-3.2-1B-Instruct#$(snap meta-llama/Llama-3.2-1B-Instruct)#" \
    -e "s#/path/to/your/local_models/Qwen/Qwen2.5-1.5B-Instruct#$(snap Qwen/Qwen2.5-1.5B-Instruct)#" \
    -e "s#^model_name_short=\"[^\"]*\"#model_name_short=\"$MS\"#" \
    -e "s#^dataset_name=\"[^\"]*\"#dataset_name=\"$DS\"#" \
    -e "s#^num_gpus=4#num_gpus=2#" \
    -e "s#^num_gen_gpus=2#num_gen_gpus=1#" \
    $RBD/code/scripts/$ALGO.sh > $S
# keep the total batch: accumulation x (old train GPUs / new train GPUs) = x2 for both SFT and RL
sed -i -E 's#^(\s*)gradient_accumulation_steps=([0-9]+)#echo_accum=1; \1gradient_accumulation_steps=$((\2 * 2))#' $S
echo "=== rbd train $ALGO $MS $DS ($REV, upstream $(cat $RBD/UPSTREAM_REV)) script=$S ==="
grep -nE "(model_name_short|dataset_name|lr|num_epochs|kl_beta|num_gpus|num_gen_gpus|gradient_accumulation_steps|train_batch_size_per_device|root_dir|run_base_dir)=" $S | grep -v "^[0-9]*:#"
bash $S
echo "RBD_TRAIN_DONE $ALGO $MS $DS rc=$?"
