# 作者B 全部任务清单(逐条命令版)

按优先级排序。环境搭建见 `PHASE2_RUNBOOK.md` §3(一次性)。
**额外下载(E1/E4 的 OPD 格需要 teacher)**:
```bash
~/envs_spectral/bin/hf download Qwen/Qwen3.5-2B
```
通用环境变量(每个 shell 都要):
```bash
export HF_HOME=/你的大盘/hf_home HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUNS=/你的输出根目录
PY=~/envs_spectral/bin/python
```
**铁律**:除 `--out`/`CUDA_VISIBLE_DEVICES` 外不要改任何参数;RLVR 的 lr 必须照抄
(默认 1e-5 会策略崩塌);每 run 交付 `capture/ log.jsonl args.json eval_gsm8k.json`。

---

## P0:Phase 2(10 run)
见 `PHASE2_RUNBOOK.md` §4,命令齐全,照抄即可。

## P1:E1 —— 16 run(0.8B,单卡/run)

seed 0 的 4 个新格:
```bash
$PY scripts/train.py --objective rlvr --optimizer ssd      --lr 2e-6 --muon-lr 2e-5 --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 --out $RUNS/e1_rlvr_ssd_s0
$PY scripts/train.py --objective rlvr --optimizer ssd-muon --lr 2e-6 --muon-lr 2e-5 --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 --out $RUNS/e1_rlvr_ssd-muon_s0
$PY scripts/train.py --objective sft  --optimizer ssd --muon-lr 2e-4 --steps 500 --save-every 25 --prompts-per-step 8 --max-new-tokens 384 --seed 0 --out $RUNS/e1_sft_ssd_s0
$PY scripts/train.py --objective opd  --optimizer ssd --muon-lr 2e-4 --teacher Qwen/Qwen3.5-2B --steps 500 --save-every 25 --prompts-per-step 8 --max-new-tokens 384 --seed 0 --out $RUNS/e1_opd_ssd_s0
```
补 seed 12 个(在下列模板上只改 `--seed {1,2}` 与 `--out` 后缀 `_s1/_s2`):
```bash
# rlvr × {adamw, muon, ssd} × seed {1,2} —— 6 个:
$PY scripts/train.py --objective rlvr --optimizer adamw --lr 2e-6                 --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 1 --out $RUNS/e1_rlvr_adamw_s1
$PY scripts/train.py --objective rlvr --optimizer muon  --lr 2e-6 --muon-lr 2e-5  --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 1 --out $RUNS/e1_rlvr_muon_s1
$PY scripts/train.py --objective rlvr --optimizer ssd   --lr 2e-6 --muon-lr 2e-5  --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 1 --out $RUNS/e1_rlvr_ssd_s1
# sft × {adamw, muon, ssd} × seed 1 —— 3 个(muon/ssd 加 --muon-lr 2e-4):
$PY scripts/train.py --objective sft --optimizer adamw --steps 500 --save-every 25 --prompts-per-step 8 --max-new-tokens 384 --seed 1 --out $RUNS/e1_sft_adamw_s1
# opd × {adamw, muon, ssd} × seed 1 —— 3 个(都加 --teacher Qwen/Qwen3.5-2B):
$PY scripts/train.py --objective opd --optimizer adamw --teacher Qwen/Qwen3.5-2B --steps 500 --save-every 25 --prompts-per-step 8 --max-new-tokens 384 --seed 1 --out $RUNS/e1_opd_adamw_s1
```
每 run 评测:
```bash
$PY scripts/eval_gsm8k.py --model $RUNS/<run>/ckpt_000500 --limit 500 --out $RUNS/<run>/eval_gsm8k.json
```

## P2:E2 —— SSD 消融 4 run(0.8B rlvr,300 步,seed 0)
在 e1_rlvr_ssd_s0 的命令上改 `--steps 300` 并各改一处:
```bash
--optimizer ssd-muon        → --out $RUNS/e2_signvariant
--ssd-k 64                  → --out $RUNS/e2_k64
--ssd-no-align              → --out $RUNS/e2_noalign
--ssd-tail-coef 0           → --out $RUNS/e2_notail
```
各配 eval(ckpt_000300)。

## P3:E3 —— 分层 SSD 4 run(0.8B rlvr,300 步,seed 0)
```bash
$PY scripts/train.py --objective rlvr --optimizer ssd-routed --ssd-layer-range 0-5   --lr 2e-6 --muon-lr 2e-5 --steps 300 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 --out $RUNS/e3_ssd_layers0-5
# 另外三个:--ssd-layer-range 6-11 / 12-17 / 18-23,--out 对应改
```
各配 eval(ckpt_000300)。

## P4:E4 —— M2 自适应 α 3 run(0.8B,300 步,seed 0)
对照不用新跑:full=phase2_{rlvr,sft}_none,ISO 型=phase2_{rlvr,sft}_frame_only(复用 P0)。
```bash
$PY scripts/train.py --objective rlvr --lr 2e-6 --intervention adaptive_alpha --steps 300 --save-every 20 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 --out $RUNS/e4_rlvr_alpha
$PY scripts/train.py --objective sft  --intervention adaptive_alpha --steps 300 --save-every 20 --prompts-per-step 8 --max-new-tokens 384 --seed 0 --out $RUNS/e4_sft_alpha
$PY scripts/train.py --objective opd  --teacher Qwen/Qwen3.5-2B --intervention adaptive_alpha --steps 300 --save-every 20 --prompts-per-step 8 --max-new-tokens 384 --seed 0 --out $RUNS/e4_opd_alpha
```
各配 eval(ckpt_000300)+ MMLU:`$PY scripts/eval_mmlu.py --model $RUNS/<run>/ckpt_000300 --limit 1000 --out $RUNS/<run>/eval_mmlu.json`

## P5:4B(5 run)
```bash
# 先下载:hf download Qwen/Qwen3.5-4B;OPD teacher:hf download Qwen/Qwen3.5-9B
# 基线一次:
$PY scripts/eval_gsm8k.py --model Qwen/Qwen3.5-4B --limit 200 --out $RUNS/eval_base_4b.json
# Phase 1 三格(单卡 80GB/run,--seqs-per-microbatch 1 --save-every 25):
$PY scripts/train.py --objective sft  --model Qwen/Qwen3.5-4B --steps 500 --save-every 25 --prompts-per-step 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/p1_4b_sft_adamw
$PY scripts/train.py --objective opd  --model Qwen/Qwen3.5-4B --teacher Qwen/Qwen3.5-9B --steps 500 --save-every 25 --prompts-per-step 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/p1_4b_opd_adamw
$PY scripts/train.py --objective rlvr --model Qwen/Qwen3.5-4B --lr 2e-6 --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/p1_4b_rlvr_adamw
# E1 RLVR 行补两格:
$PY scripts/train.py --objective rlvr --model Qwen/Qwen3.5-4B --optimizer muon --lr 2e-6 --muon-lr 2e-5 --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/e1_4b_rlvr_muon
$PY scripts/train.py --objective rlvr --model Qwen/Qwen3.5-4B --optimizer ssd  --lr 2e-6 --muon-lr 2e-5 --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/e1_4b_rlvr_ssd
```

## P6:9B(3 run,4B 收尾后;多卡 `--device-map auto`)
见 `SCALING.md` 的 9B 配方;三格 = P5 的 Phase 1 三格把 `--model` 换成
Qwen/Qwen3.5-9B、teacher 换成 Qwen/Qwen3.5-27B(先 hf download),
加 `--device-map auto`,CUDA_VISIBLE_DEVICES 给 3-4 张卡。

---

## 交付方式
每完成一批,打包:
```bash
cd $RUNS && for d in <本批目录>; do tar cf $d.tar $d; done
```
传输任选其一:① scp 到集群 `hopper.orc.gmu.edu:/scratch/rhong5/from_B/`(账号找作者A 开);
② 任何网盘给下载链接;③ 如果你的机器能被 ssh,直接给作者A 一个可读路径。
**每批开跑 30 分钟内**把各 run 的 `tail -5 log.jsonl` 发给作者A 做健康核对。
