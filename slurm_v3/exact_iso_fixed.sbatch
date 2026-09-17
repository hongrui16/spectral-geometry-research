#!/bin/bash
# exact_iso arm rerun after the engine fix (2026-09-17: CUDA SVD driver gesvda + norm matching).
# Old exact_iso dirs are void (registry "exact_iso engine defect"); new dirs carry the _fixed tag.
# 0-2: RLVR lr*=2e-7 seeds 0/1/2; 3-4: SFT lr*=1e-6 seeds 0/1.
# Submit: sbatch --partition=gpuq --gres=gpu:3g.40gb:1 --array=0-4 slurm_v3/exact_iso_fixed.sbatch  (+ contrib-gpuq copy)
#SBATCH --job-name=spec-isofix
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
OBJS=(rlvr rlvr rlvr sft sft); SEEDS=(0 1 2 0 1); LRS=(2e-7 2e-7 2e-7 1e-6 1e-6)
i=$SLURM_ARRAY_TASK_ID; OBJ=${OBJS[$i]}
OUT=$RUNS/v3_${OBJ}_exact_iso_fixed_lr0.1_s${SEEDS[$i]}
if ! mkdir "$OUT.lock" 2>/dev/null; then echo "SKIP: $OUT claimed"; exit 0; fi
echo "$SLURM_JOB_ID $SLURMD_NODENAME $SLURM_JOB_GRES" > "$OUT.lock/job"
ROLL=""; [ $OBJ = rlvr ] && ROLL="--rollouts 8"
echo "=== $OBJ exact_iso(fixed) lr=${LRS[$i]} seed=${SEEDS[$i]} -> $OUT ==="
$PY scripts_v3/train.py --objective $OBJ --steps 300 --save-every 20 --lr ${LRS[$i]} \
  --prompts-per-step 8 $ROLL --max-new-tokens 384 --seed ${SEEDS[$i]} \
  --intervention exact_iso --intervention-scale 1 --eval-every 50 --out $OUT
rc=$?; [ $rc -ne 0 ] && { echo "TRAIN_FAILED rc=$rc"; rm -rf "$OUT.lock"; exit $rc; }
$PY analysis_v3/compute_metrics.py $OUT --out-root $PROJ/results/v3/result_A
$PY scripts_v3/summary.py $OUT
D=$PROJ/results/v3/result_A/$(basename $OUT); mkdir -p $D
cp $OUT/log.jsonl $OUT/args.json $OUT/eval_*.json $D/ 2>/dev/null
rm -rf $OUT/capture; ls -d $OUT/ckpt_* 2>/dev/null | grep -v ckpt_000300 | xargs -r rm -rf
echo "ISOFIX_DONE $OBJ s${SEEDS[$i]}"
