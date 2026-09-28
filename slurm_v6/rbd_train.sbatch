#!/bin/bash
# v6 stage 0: run the AUTHORS' training script (scripts/{sft,self_sft,rl}.sh) unchanged except for
# paths / environment: code dir, run dir, local model snapshots, model_name_short, dataset_name.
# Hyperparameters are exactly the authors' script defaults (see results_draft v6 for the
# script-vs-paper discrepancies). One 4xA100 node = one job (4-job cap).
# Env: ALGO=sft|self_sft|rl  MS=llama-3.2-1b-inst|qwen-2.5-1.5b-inst  DS=<dataset class name>
#SBATCH --job-name=v5-rbdtrain
#SBATCH --partition=contrib-gpuq,gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
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
    $RBD/code/scripts/$ALGO.sh > $S
echo "=== rbd train $ALGO $MS $DS ($REV, upstream $(cat $RBD/UPSTREAM_REV)) script=$S ==="
grep -nE "^(model_name_short|dataset_name|lr|num_epochs|kl_beta|total_batch_size|root_dir|run_base_dir)=" $S
bash $S
echo "RBD_TRAIN_DONE $ALGO $MS $DS rc=$?"
