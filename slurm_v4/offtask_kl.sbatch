#!/bin/bash
# v4 §2.1-1: off-task (MMLU-validation) KL for every finished v3 run (cumulative, 300-step ckpt)
# and for the 8-step captures (single-step), plus the task-reference numbers on the same runs
# for a like-for-like table. Zero training. ~1.5 h on one A100.
# Submit: sbatch --partition=contrib-gpuq --gres=gpu:A100.80gb:1 slurm_v4/offtask_kl.sbatch
#SBATCH --job-name=spec-offkl
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
OUT=$PROJ/results/v4/result_A
$PY analysis_v4/cum_kl.py --runs-root $RUNS --glob 'v3_*' --step 300 --reference offtask --n-prompts 64 --out $OUT/cum_kl_offtask
for OBJ in sft rlvr; do for STEP in 8 4; do
  L=""; for I in none frame_matched exact_iso spectrum_matched random_ext; do L="$L $RUNS/klcap_v3_${OBJ}_${I}"; done
  $PY analysis_v4/step_kl.py --step $STEP --reference offtask --n-prompts 64 --out $OUT/step_kl_offtask_${OBJ}_step${STEP} --runs $L
done; done
echo OFFTASK_KL_DONE
