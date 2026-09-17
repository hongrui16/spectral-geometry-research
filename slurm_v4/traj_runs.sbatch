#!/bin/bash
# A3(a): off-task / task KL trajectory. Dense SFT lr* (1e-6) and RLVR lr* (2e-7), 2 seeds, keep every 50-step ckpt,
# then cumulative KL (both references) on each kept ckpt. 4 runs.
#   sbatch --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=0-3 slurm_v4/traj_runs.sbatch
#SBATCH --job-name=spec-traj
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
OBJS=(sft sft rlvr rlvr); SEEDS=(0 1 0 1); i=$SLURM_ARRAY_TASK_ID; OBJ=${OBJS[$i]}
LR=1e-6; ROLL=""; [ $OBJ = rlvr ] && { LR=2e-7; ROLL="--rollouts 8"; }
OUT=$RUNS/v4_${OBJ}_full_traj_lr0.1_s${SEEDS[$i]}
if ! mkdir "$OUT.lock" 2>/dev/null; then echo "SKIP: $OUT claimed"; exit 0; fi
echo "=== traj $OBJ lr=$LR seed=${SEEDS[$i]} -> $OUT ==="
$PY scripts_v4/train.py --objective $OBJ --steps 300 --save-every 20 --lr $LR --prompts-per-step 8 $ROLL --max-new-tokens 384 \
  --seed ${SEEDS[$i]} --eval-every 50 --keep-eval-ckpts --out $OUT
rc=$?; [ $rc -ne 0 ] && { echo "TRAIN_FAILED rc=$rc"; rm -rf "$OUT.lock"; exit $rc; }
CKS=$(ls -d $OUT/ckpt_* | tr '\n' ' ')
for REF in offtask task; do $PY analysis_v4/cum_kl.py --ckpts $CKS --reference $REF --n-prompts 64 --out $PROJ/results/v4/result_A/traj_${REF}/$(basename $OUT); done
D=$PROJ/results/v4/result_A/$(basename $OUT); mkdir -p $D; cp $OUT/log.jsonl $OUT/args.json $OUT/eval_*.json $D/ 2>/dev/null
rm -rf $OUT/capture; echo "TRAJ_DONE $OBJ s${SEEDS[$i]}"
