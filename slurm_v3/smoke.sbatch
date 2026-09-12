#!/bin/bash
# v3 GPU smoke: exercises --intervention-scale, per-step basis refresh, fp32 W capture,
# --eval-every (tiny eval limits), --kl-beta reference path and resp_len/trunc logging.
# Submit: sbatch --array=0-3 slurm_v3/smoke.sbatch   (any GPU partition; override --partition/--gres)
#SBATCH --job-name=spec-smk3
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
case $SLURM_ARRAY_TASK_ID in
  0) OBJ=sft;  INT=spectrum_matched; SCALE=3;   EXTRA="";;
  1) OBJ=sft;  INT=random_ext;       SCALE=1;   EXTRA="";;
  2) OBJ=rlvr; INT=none;             SCALE=1;   EXTRA="--kl-beta 0.01";;
  3) OBJ=rlvr; INT=spectrum_matched; SCALE=1;   EXTRA="";;
esac
OUT=$RUNS/smoke_v3_${OBJ}_${INT}_s${SCALE}
rm -rf $OUT
echo "=== smoke_v3 $OBJ $INT scale=$SCALE $EXTRA -> $OUT ($(git -C $PROJ rev-parse --short HEAD)) ==="

$PY scripts_v3/train.py --objective $OBJ --steps 8 --save-every 4 --save-ckpt-every 1000 \
  --prompts-per-step 4 --rollouts 4 --max-new-tokens 128 --seed 0 \
  --intervention $INT --intervention-scale $SCALE \
  --eval-every 8 --eval-limit-gsm8k 16 --eval-limit-mmlu 32 $EXTRA --out $OUT
$PY analysis_v3/compute_metrics.py $OUT --out-root $PROJ/results/v3/result_A
$PY scripts_v3/summary.py $OUT
$PY - <<EOF
import torch, glob, json
f = sorted(glob.glob("$OUT/capture/step_*.pt"))[-1]
rec = torch.load(f, map_location="cpu")
n, d = next(iter(rec["mats"].items()))
print("capture dtypes:", {k: str(v.dtype) for k, v in d.items() if hasattr(v, "dtype")})
assert d["W"].dtype == torch.float32 and d["H"].dtype == torch.float32, "W/H not fp32"
logs = [json.loads(l) for l in open("$OUT/log.jsonl")]
print("last log records:", logs[-3:])
EOF
echo "SMOKE_V3_DONE $OBJ $INT $SCALE"
