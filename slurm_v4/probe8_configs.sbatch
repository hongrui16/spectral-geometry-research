#!/bin/bash
# v4 §2.2 forgetting probe: 8-step captures for every configuration that has a finished 300-step run
# (v3 captures were deleted by the scratch rule). Same prompt stream (seed 0) per objective.
# 16 configs x 2 objectives = 32 captures, ~15 min each on MIG. Then step_kl --reference offtask/task per group.
#   sbatch --partition=gpuq         --gres=gpu:3g.40gb:1 --array=0-31 slurm_v4/probe8_configs.sbatch
#   sbatch --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=0-31 slurm_v4/probe8_configs.sbatch
#SBATCH --job-name=spec-probe8
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
# config table: tag | intervention | lr multiplier vs lr x1 | s_rel
CFG=( "full_lr0.033|none|0.0333|1" "full_lr0.1|none|0.1|1" "full_lr0.33|none|0.333|1" "full_lr1|none|1|1"
      "frame_lr0.1|frame_matched|0.1|1" "iso_lr0.1|exact_iso|0.1|1" "spec_lr0.1|spectrum_matched|0.1|1" "rand_lr0.1|random_ext|0.1|1"
      "frame_lr1|frame_matched|1|1" "iso_lr1|exact_iso|1|1" "spec_lr1|spectrum_matched|1|1" "rand_lr1|random_ext|1|1"
      "spec_lr0.1_srel3|spectrum_matched|0.1|3" "rand_lr0.1_srel3|random_ext|0.1|3" "spec_lr0.1_srel10|spectrum_matched|0.1|10" "rand_lr0.1_srel10|random_ext|0.1|10" )
i=$SLURM_ARRAY_TASK_ID; OBJ=$([ $i -lt 16 ] && echo sft || echo rlvr); IFS='|' read TAG INT MULT SREL <<< "${CFG[$((i % 16))]}"
BASE=1e-5; MNT=128; PPS=4; ROLL=""; [ $OBJ = rlvr ] && { BASE=2e-6; MNT=384; PPS=8; ROLL="--rollouts 8"; }
LR=$($PY -c "print($BASE*$MULT)")
OUT=$RUNS/probe8_v4_${OBJ}_${TAG}
if ! mkdir "$OUT.lock" 2>/dev/null; then echo "SKIP: $OUT claimed"; exit 0; fi
echo "=== probe8 $OBJ $TAG int=$INT lr=$LR s_rel=$SREL -> $OUT ==="
$PY scripts_v4/train.py --objective $OBJ --steps 8 --save-every 4 --save-ckpt-every 1000 --lr $LR \
  --prompts-per-step $PPS $ROLL --max-new-tokens $MNT --seed 0 \
  --intervention $INT --intervention-scale $SREL --eval-every 1000 --out $OUT
rc=$?; [ $rc -ne 0 ] && { echo "TRAIN_FAILED rc=$rc"; rm -rf "$OUT.lock"; exit $rc; }
ls $OUT/capture | head -3; echo "PROBE8_DONE $OBJ $TAG"
