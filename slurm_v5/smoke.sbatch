#!/bin/bash
# v5 stage-0 GPU smoke (docs/unified_paper_document_v5.md part 3 §0).
#   0 base evals (0.8B): GSM8K full test (amp + bf16 numerics) + retention suite (amp + bf16)
#   1 length/accuracy probe: 0.8B at T=1, k=4, 256 train questions, 1536-token budget
#   2 same for the Qwen3.5-2B teacher at T=0.7
#   3 SFT loop smoke (gsm8k_clean, 10 steps, tiny evals)
#   4 RLVR loop smoke (10 steps, 768-token rollouts, tiny evals)
# Submit: bash slurm_v5/submit_limited.sh "sbatch --array=0-4%4 slurm_v5/smoke.sbatch"  (max 4 jobs at once)
#SBATCH --job-name=v5-smoke
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%A_%a.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm_v5/common.sh
RES=$PROJ/results/v5/result_A
echo "=== v5 smoke task $SLURM_ARRAY_TASK_ID ($REV) on $(hostname) ==="
set -x
case $SLURM_ARRAY_TASK_ID in
  0) B=$RES/base; mkdir -p $B
     $PY scripts_v5/eval_gsm8k.py --model Qwen/Qwen3.5-0.8B --dump-samples 20 --out $B/eval_gsm8k.json
     $PY scripts_v5/eval_gsm8k.py --model Qwen/Qwen3.5-0.8B --numerics bf16 --out $B/eval_gsm8k_bf16.json
     $PY scripts_v5/eval_retention.py --model Qwen/Qwen3.5-0.8B --out $B/eval_retention.json
     $PY scripts_v5/eval_retention.py --model Qwen/Qwen3.5-0.8B --numerics bf16 --tasks mmlu,arc,hellaswag,triviaqa --out $B/eval_retention_bf16.json ;;
  1) $PY scripts_v5/make_sft_data.py --model Qwen/Qwen3.5-0.8B --k 4 --temperature 1.0 --limit 256 \
       --max-new-tokens 1536 --out $RUNS/data/probe_self08b.jsonl
     cp $RUNS/data/probe_self08b.jsonl.stats.json $RES/probe_self08b_stats.json ;;
  2) $PY scripts_v5/make_sft_data.py --model Qwen/Qwen3.5-2B --k 4 --temperature 0.7 --limit 256 \
       --max-new-tokens 1536 --out $RUNS/data/probe_teacher2b.jsonl
     cp $RUNS/data/probe_teacher2b.jsonl.stats.json $RES/probe_teacher2b_stats.json ;;
  3) OUT=$RUNS/smoke_v5_sft; rm -rf $OUT
     $PY scripts_v5/train.py --objective sft --sft-data gsm8k_clean --lr 1e-6 --steps 10 --save-every 5 \
       --save-ckpt-every 1000 --prompts-per-step 4 --seed 0 --eval-every 10 --eval-limit-gsm8k 32 \
       --eval-retention-extra "--limit mmlu=64 --limit arc=64 --limit hellaswag=32 --limit triviaqa=32 --limit mmlu_gen=16" \
       --out $OUT
     $PY analysis_v5/bf16_visible.py --ckpt $OUT/ckpt_000010 --write-run
     tail -3 $OUT/log.jsonl ;;
  4) OUT=$RUNS/smoke_v5_rlvr; rm -rf $OUT
     $PY scripts_v5/train.py --objective rlvr --lr 2e-6 --steps 10 --save-every 5 --save-ckpt-every 1000 \
       --prompts-per-step 4 --rollouts 4 --max-new-tokens 768 --seed 0 --eval-every 10 --eval-limit-gsm8k 32 \
       --eval-retention-extra "--limit mmlu=64 --limit arc=64 --limit hellaswag=32 --limit triviaqa=32 --limit mmlu_gen=16" \
       --out $OUT
     $PY analysis_v5/bf16_visible.py --ckpt $OUT/ckpt_000010 --write-run
     grep -v '"eval"' $OUT/log.jsonl | tail -3 ;;
esac
echo "SMOKE_V5_DONE $SLURM_ARRAY_TASK_ID"
