#!/bin/bash
# v4 §2.1-2: spectrum-preserving (frame_matched) and spectrum-pinned (exact_iso, fixed engine) arms at
# the COMMON learning rate (lr x1: SFT 1e-5, RLVR 2e-6), where r-dof arms actually move, so the
# direction comparison has power. 2 seeds x 2 arms x 2 objectives = 8 runs. Lock-claim as in v3.
#   sbatch --partition=gpuq         --gres=gpu:3g.40gb:1 --array=0-7 slurm_v4/batch1_lr1_arms.sbatch
#   sbatch --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=0-7 slurm_v4/batch1_lr1_arms.sbatch
#SBATCH --job-name=spec-v4b1
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
OBJS=(sft sft sft sft rlvr rlvr rlvr rlvr); MODES=(frame_matched frame_matched exact_iso exact_iso frame_matched frame_matched exact_iso exact_iso); SEEDS=(0 1 0 1 0 1 0 1)
i=$SLURM_ARRAY_TASK_ID; OBJ=${OBJS[$i]}; M=${MODES[$i]}; LR=1e-5; ROLL=""; [ $OBJ = rlvr ] && { LR=2e-6; ROLL="--rollouts 8"; }
OUT=$RUNS/v4_${OBJ}_${M}_lr1_s${SEEDS[$i]}
if ! mkdir "$OUT.lock" 2>/dev/null; then echo "SKIP: $OUT claimed"; exit 0; fi
echo "$SLURM_JOB_ID $SLURMD_NODENAME $SLURM_JOB_GRES" > "$OUT.lock/job"
echo "=== v4 b1 $OBJ $M lr=$LR seed=${SEEDS[$i]} -> $OUT ==="
$PY scripts_v4/train.py --objective $OBJ --steps 300 --save-every 20 --lr $LR \
  --prompts-per-step 8 $ROLL --max-new-tokens 384 --seed ${SEEDS[$i]} \
  --intervention $M --intervention-scale 1 --eval-every 50 --out $OUT
rc=$?; [ $rc -ne 0 ] && { echo "TRAIN_FAILED rc=$rc"; rm -rf "$OUT.lock"; exit $rc; }
$PY analysis_v4/compute_metrics.py $OUT --out-root $PROJ/results/v4/result_A
D=$PROJ/results/v4/result_A/$(basename $OUT); mkdir -p $D
cp $OUT/log.jsonl $OUT/args.json $OUT/eval_*.json $D/ 2>/dev/null
rm -rf $OUT/capture; ls -d $OUT/ckpt_* 2>/dev/null | grep -v ckpt_000300 | xargs -r rm -rf
echo "V4B1_DONE $OBJ $M s${SEEDS[$i]}"
