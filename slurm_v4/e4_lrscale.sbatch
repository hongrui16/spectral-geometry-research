#!/bin/bash
# E-v4-4: per-matrix lr multipliers from the task/off-task attribution (by type, alpha 1 and 2), at lr*,
# SFT and RLVR, 2 seeds each = 8 runs. Compared against dense lr* (existing) on GSM8K, MMLU, and off-task/task cumulative KL.
#   sbatch --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=0-7 slurm_v4/e4_lrscale.sbatch
#SBATCH --job-name=spec-e4ls
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v4/common.sh
OBJS=(sft sft sft sft rlvr rlvr rlvr rlvr); ALPHAS=(1 1 2 2 1 1 2 2); SEEDS=(0 1 0 1 0 1 0 1)
i=$SLURM_ARRAY_TASK_ID; OBJ=${OBJS[$i]}; A=${ALPHAS[$i]}
LR=1e-6; ROLL=""; [ $OBJ = rlvr ] && { LR=2e-7; ROLL="--rollouts 8"; }
SC=$PROJ/results/v4/result_A/lr_scales/${OBJ}_type_a${A}.json
OUT=$RUNS/v4_${OBJ}_lrscale_a${A}_lr0.1_s${SEEDS[$i]}
if ! mkdir "$OUT.lock" 2>/dev/null; then echo "SKIP: $OUT claimed"; exit 0; fi
echo "=== E-v4-4 $OBJ alpha=$A lr=$LR seed=${SEEDS[$i]} scales=$SC -> $OUT ==="
$PY scripts_v4/train.py --objective $OBJ --steps 300 --save-every 20 --lr $LR --lr-scale-file $SC \
  --prompts-per-step 8 $ROLL --max-new-tokens 384 --seed ${SEEDS[$i]} --eval-every 50 --out $OUT
rc=$?; [ $rc -ne 0 ] && { echo "TRAIN_FAILED rc=$rc"; rm -rf "$OUT.lock"; exit $rc; }
for REF in offtask task; do $PY analysis_v4/cum_kl.py --ckpts $OUT/ckpt_000300 --reference $REF --n-prompts 64 --out $PROJ/results/v4/result_A/cum_kl_${REF}_e4/$(basename $OUT); done
D=$PROJ/results/v4/result_A/$(basename $OUT); mkdir -p $D; cp $OUT/log.jsonl $OUT/args.json $OUT/eval_*.json $D/ 2>/dev/null
rm -rf $OUT/capture; ls -d $OUT/ckpt_* | grep -v ckpt_000300 | xargs -r rm -rf; echo "E4LS_DONE $OBJ a$A s${SEEDS[$i]}"
