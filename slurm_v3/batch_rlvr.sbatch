#!/bin/bash
# v3 RLVR side run by A (2026-09-16; B's RLVR deliveries never arrived). lr* = 2e-7 (B batch-1 report).
# Tier 1 (0-5): dense lr* x2, frame_matched x2, exact_iso x2  -> H-0 local check + H-2 (C4).
# Tier 2 (6-9): spectrum_matched x2, random_ext x2 at lr*      -> H-1 + frontier r-dof points.
# Tier 3 (10-15): lr x1 seed 2 (x2), s_rel 3 / 10 x both dicts -> H-3 (C5 final).
# Tier 4 (16-18, added 2026-09-16 after tier 1): third seed of dense / frame_matched / exact_iso at lr*
#   (exact_iso > dense on GSM8K at 2 seeds, t=2.4; third seed to confirm).
# Submit the same array under several GPU types; first task to start claims $OUT.lock, duplicates exit;
# a failed train releases the lock so another GPU type can retry:
#   sbatch --partition=contrib-gpuq --gres=gpu:A100.80gb:1 --array=0-9 slurm_v3/batch_rlvr.sbatch
#   sbatch --partition=gpuq         --gres=gpu:A100.80gb:1 --array=0-9 slurm_v3/batch_rlvr.sbatch
#   sbatch --partition=gpuq,contrib-gpuq --gres=gpu:3g.40gb:1 --array=0-9 slurm_v3/batch_rlvr.sbatch
#SBATCH --job-name=spec-rlvr
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
MODES=(none none frame_matched frame_matched exact_iso exact_iso spectrum_matched spectrum_matched random_ext random_ext spectrum_matched random_ext spectrum_matched random_ext spectrum_matched random_ext none frame_matched exact_iso)
SEEDS=(0 1 0 1 0 1 0 1 0 1 2 2 0 0 0 0 2 2 2)
LRS=(2e-7 2e-7 2e-7 2e-7 2e-7 2e-7 2e-7 2e-7 2e-7 2e-7 2e-6 2e-6 2e-7 2e-7 2e-7 2e-7 2e-7 2e-7 2e-7)
LRTAG=(lr0.1 lr0.1 lr0.1 lr0.1 lr0.1 lr0.1 lr0.1 lr0.1 lr0.1 lr0.1 lr1 lr1 lr0.1 lr0.1 lr0.1 lr0.1 lr0.1 lr0.1 lr0.1)
SCALES=(1 1 1 1 1 1 1 1 1 1 1 1 3 3 10 10 1 1 1)
i=$SLURM_ARRAY_TASK_ID
M=${MODES[$i]}; [ "$M" = none ] && NAME=full || NAME=$M
SR=""; [ "${SCALES[$i]}" != 1 ] && SR="_srel${SCALES[$i]}"
OUT=$RUNS/v3_rlvr_${NAME}_${LRTAG[$i]}${SR}_s${SEEDS[$i]}
if ! mkdir "$OUT.lock" 2>/dev/null; then echo "SKIP: $OUT already claimed by $(cat $OUT.lock/job 2>/dev/null)"; exit 0; fi
echo "$SLURM_JOB_ID $SLURMD_NODENAME $SLURM_JOB_GRES" > "$OUT.lock/job"
echo "=== rlvr $M lr=${LRS[$i]} s_rel=${SCALES[$i]} seed=${SEEDS[$i]} -> $OUT ==="
INT=""; [ "$M" != none ] && INT="--intervention $M --intervention-scale ${SCALES[$i]}"

$PY scripts_v3/train.py --objective rlvr --steps 300 --save-every 20 --lr ${LRS[$i]} \
  --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed ${SEEDS[$i]} \
  $INT --eval-every 50 --out $OUT
rc=$?
if [ $rc -ne 0 ]; then echo "TRAIN_FAILED rc=$rc on $SLURM_JOB_GRES; releasing lock"; rm -rf "$OUT.lock"; exit $rc; fi
$PY analysis_v3/compute_metrics.py $OUT --out-root $PROJ/results/v3/result_A
$PY scripts_v3/summary.py $OUT
D=$PROJ/results/v3/result_A/$(basename $OUT); mkdir -p $D
cp $OUT/log.jsonl $OUT/args.json $OUT/summary.txt $OUT/eval_*.json $D/ 2>/dev/null
rm -rf $OUT/capture; ls -d $OUT/ckpt_* 2>/dev/null | grep -v ckpt_000300 | xargs -r rm -rf
echo "RLVR_DONE $NAME ${LRTAG[$i]}${SR} s${SEEDS[$i]}"
