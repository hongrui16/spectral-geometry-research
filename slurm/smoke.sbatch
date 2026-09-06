#!/bin/bash
#SBATCH --job-name=spec-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm/common.sh

echo "=== GPU ==="; nvidia-smi | head -12

for OBJ in sft rlvr opd; do
  echo "=== smoke $OBJ ==="
  $PY scripts/train.py --objective $OBJ \
    --steps 4 --save-every 2 --prompts-per-step 4 --rollouts 4 \
    --max-new-tokens 256 --microbatches 4 \
    --out $RUNS/smoke_$OBJ || exit 1
done

echo "=== capture files ==="
ls -la $RUNS/smoke_*/capture/
$PY analysis/check_capture.py $RUNS/smoke_sft/capture $RUNS/smoke_rlvr/capture
echo "SMOKE_OK"
