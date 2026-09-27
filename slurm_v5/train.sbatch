#!/bin/bash
# v5 generic training array. Each non-comment line of $CFG is "<run_name> <train.py args...>";
# array index i runs line i (0-based). Always under the 4-job cap:
#   bash slurm_v5/submit_limited.sh "sbatch --array=0-N%4 --export=ALL,CFG=slurm_v5/configs/<x>.txt slurm_v5/train.sbatch"
# After training: bf16 visibility of the final ckpt (-> <run>/bf16_visible.json) and a copy of the
# run's small files into results/v5/result_A/<run>/.
#SBATCH --job-name=v5-train
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v5/common.sh
LINE=$(grep -v '^\s*#' $CFG | grep -v '^\s*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -z "$LINE" ] && { echo "no config line $SLURM_ARRAY_TASK_ID in $CFG"; exit 1; }
NAME=$(echo $LINE | cut -d' ' -f1); ARGS=$(echo $LINE | cut -d' ' -f2-)
OUT=$RUNS/$NAME
echo "=== $NAME ($REV) on $(hostname): $ARGS ==="
if [ -f $OUT/eval_gsm8k.json ]; then echo "already done: $OUT"; exit 0; fi
rm -rf $OUT
$PY scripts_v5/train.py $ARGS --out $OUT || exit 1
LAST=$(ls -d $OUT/ckpt_* | sort | tail -1)
$PY analysis_v5/bf16_visible.py --ckpt $LAST --write-run > /dev/null
DST=$PROJ/results/v5/result_A/$NAME; mkdir -p $DST
cp $OUT/log.jsonl $OUT/args.json $OUT/eval_*.json $OUT/bf16_visible.json $DST/ 2>/dev/null
echo "rev=$REV" > $DST/REV
echo "TRAIN_DONE $NAME"
