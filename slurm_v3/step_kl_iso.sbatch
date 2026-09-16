#!/bin/bash
# E-v3-0c extension (2026-09-16): per-step functional step (KL) of exact_iso / frame_matched vs full,
# spectrum_matched, random_ext at equal Frobenius norm, same prompt stream/seed, 8 steps, both objectives.
# Motivation: RLVR H-2 result (exact_iso learns fastest) must be checked against the effective-step (M1) explanation.
# Stage 1 (array 0-9): 8-step capture runs, smoke settings (lr x1, seed 0, save-every 4).
# Stage 2 (STAGE=kl):  step_kl.py at steps 8 and 4 for the SFT set and the RLVR set.
# Submit:
#   J=$(sbatch --parsable --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=0-9 slurm_v3/step_kl_iso.sbatch)
#   sbatch --dependency=afterany:$J --export=ALL,STAGE=kl --partition=contrib-gpuq --gres=gpu:A100.80gb:1 slurm_v3/step_kl_iso.sbatch
#SBATCH --job-name=spec-klcap
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out
source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v3/common.sh
INTS=(none frame_matched exact_iso spectrum_matched random_ext)
if [ "${STAGE:-cap}" = kl ]; then
  for OBJ in sft rlvr; do for STEP in 8 4; do
    RUNS_LIST=""; for I in "${INTS[@]}"; do RUNS_LIST="$RUNS_LIST $RUNS/klcap_v3_${OBJ}_${I}"; done
    $PY analysis_v3/step_kl.py --step $STEP --out $PROJ/results/v3/result_A/step_kl_iso_${OBJ}_step${STEP} --runs $RUNS_LIST
  done; done
  echo STEP_KL_ISO_DONE; exit 0
fi
i=$SLURM_ARRAY_TASK_ID; OBJ=$([ $i -lt 5 ] && echo sft || echo rlvr); INT=${INTS[$((i % 5))]}
LR=1e-5; MNT=128; PPS=4; [ $OBJ = rlvr ] && { LR=2e-6; MNT=384; PPS=8; }
OUT=$RUNS/klcap_v3_${OBJ}_${INT}
if ! mkdir "$OUT.lock" 2>/dev/null; then echo "SKIP: $OUT claimed"; exit 0; fi
echo "=== klcap $OBJ $INT lr=$LR -> $OUT ==="
$PY scripts_v3/train.py --objective $OBJ --steps 8 --save-every 4 --save-ckpt-every 1000 --lr $LR \
  --prompts-per-step $PPS --rollouts 8 --max-new-tokens $MNT --seed 0 \
  --intervention $INT --intervention-scale 1 --eval-every 1000 --out $OUT
rc=$?; [ $rc -ne 0 ] && { echo "TRAIN_FAILED rc=$rc"; rm -rf "$OUT.lock"; exit $rc; }
ls $OUT/capture | head; echo "KLCAP_DONE $OBJ $INT"
