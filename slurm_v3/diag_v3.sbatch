#!/bin/bash
# E-v3-0a (cum_kl over all v3 ckpt_000300) + E-v3-0b (MMLU format check) run by A on the v3 runs
# (B's e4a-based versions never delivered; v3 ckpts are the ones the verdicts use).
# Submit: sbatch --partition=contrib-gpuq --gres=gpu:A100.80gb:1 slurm_v3/diag_v3.sbatch
#SBATCH --job-name=spec-diag
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
OUT=$PROJ/results/v3/result_A
$PY analysis_v3/cum_kl.py --runs-root $RUNS --glob 'v3_*' --step 300 --out $OUT/cum_kl
for r in v3_rlvr_full_lr0.1_s0 v3_rlvr_exact_iso_lr0.1_s0 v3_rlvr_frame_matched_lr0.1_s0 v3_rlvr_spectrum_matched_lr0.1_s0 v3_rlvr_random_ext_lr0.1_s0 v3_rlvr_random_ext_lr1_s2 \
         v3_sft_full_lr0.1_s0 v3_sft_full_lr0.33_s0 v3_sft_full_lr0.033_s0 v3_sft_exact_iso_lr0.1_s0 v3_sft_spectrum_matched_lr0.1_s0 v3_sft_spectrum_matched_lr1_s2; do
  $PY analysis_v3/mmlu_format.py --model $RUNS/$r/ckpt_000300 --out $OUT/mmlu_format/$r.json
done
$PY analysis_v3/mmlu_format.py --model Qwen/Qwen3.5-0.8B --out $OUT/mmlu_format/base.json
echo DIAG_DONE
