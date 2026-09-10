#!/bin/bash
# v2 metrics (split-half SNR, R_sigma_cross, H self-check, manifest) recomputed
# from the v1 captures still on scratch. CPU only. One array task per run.
# Submit: sbatch --array=0-8 slurm/recompute_v2.sbatch
# Output: results/v2/result_A/<run>/{metrics.csv,manifest.json[,rho_hist.pt]}
#SBATCH --job-name=spec-rc2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm/common.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUNLIST=(
  phase1_sft_adamw phase1_opd_adamw phase1_rlvr_adamw
  phase1_sft_muon  phase1_opd_muon  phase1_rlvr_muon
  h3power_rlvr phase1_llama1b_rlvr_adamw phase1_llama3b_rlvr_adamw
)
RUN=${RUNLIST[$SLURM_ARRAY_TASK_ID]}
echo "=== recompute_v2 $RUN ($(git -C $PROJ rev-parse --short HEAD), branch $(git -C $PROJ branch --show-current)) ==="
$PY analysis/compute_metrics_v2.py $RUNS/$RUN --cpu --out-root $PROJ/results/v2/result_A
# keep the run's small v1 files next to the v2 metrics for traceability
cp $RUNS/$RUN/args.json $RUNS/$RUN/log.jsonl $PROJ/results/v2/result_A/$RUN/ 2>/dev/null || true
cp $RUNS/$RUN/eval*.json $PROJ/results/v2/result_A/$RUN/ 2>/dev/null || true
test -s $PROJ/results/v2/result_A/$RUN/manifest.json && echo "RECOMPUTE_V2_DONE $RUN"
