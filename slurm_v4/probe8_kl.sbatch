#!/bin/bash
# After probe8_configs: single-step KL (task and off-task) at steps 8 and 4 for all 16 configs per objective.
# Submit with --dependency=afterany:<probe8 array ids>: sbatch --dependency=afterany:J1:J2 --partition=contrib-gpuq --gres=gpu:A100.80gb:1 slurm_v4/probe8_kl.sbatch
#SBATCH --job-name=spec-probe8kl
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
OUT=$PROJ/results/v4/result_A
for OBJ in sft rlvr; do
  L=$(ls -d $RUNS/probe8_v4_${OBJ}_* | grep -v '\.lock$' | tr '\n' ' ')
  for REF in offtask task; do for STEP in 8 4; do
    $PY analysis_v4/step_kl.py --step $STEP --reference $REF --n-prompts 64 --out $OUT/probe8_${REF}_${OBJ}_step${STEP} --runs $L
  done; done
done
echo PROBE8_KL_DONE
