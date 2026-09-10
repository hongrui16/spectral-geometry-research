> **v1 归档文档(2026-09-06 → 09-09,已冻结,只改了文件名引用)。** 当前版本见 `docs/unified_paper_document_v2.md`、`docs/TASKS_A_v2.md`、`docs/TASKS_B_v2.md`。本文中的 `scripts/ analysis/ specgeom/ slurm/` 指 **v1 代码文件夹**(git tag `v1`);v2 代码在 `*_v2/` 文件夹。本文提到的 run 结果归档在 `results/v1/result_A/`(作者A)与 `results/v1/result_B/`(作者B)。

# 作者B 任务清单(唯一入口,逐条命令版)

按 P0→P6 优先级执行。实验的科学设计与产出见 `docs/EXPERIMENTS_v1.md`(总账,不用照它跑)。

## 0. 环境(一次性,约 20 分钟)
```bash
git clone git@github.com:hongrui16/spectral-geometry-research.git && cd spectral-geometry-research
python3.10 -m venv ~/envs_spectral
~/envs_spectral/bin/pip install -U pip
~/envs_spectral/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
~/envs_spectral/bin/pip install -U transformers trl datasets accelerate math-verify matplotlib pandas tensorboard
# transformers 必须 >= 5.x(qwen3_5 架构)
export HF_HOME=/你的大盘/hf_home
~/envs_spectral/bin/hf download Qwen/Qwen3.5-0.8B
~/envs_spectral/bin/hf download Qwen/Qwen3.5-2B     # E1/E4 的 OPD teacher
~/envs_spectral/bin/python -c "from datasets import load_dataset; load_dataset('openai/gsm8k','main')"
```
每个跑训练的 shell:
```bash
export HF_HOME=/你的大盘/hf_home HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUNS=/你的输出根目录
PY=~/envs_spectral/bin/python
```
**铁律**:除 `--out`/`CUDA_VISIBLE_DEVICES` 外不要改任何参数;RLVR 的 lr 必须照抄
(默认 1e-5 会策略崩塌);OOM 时唯一允许自调的参数是 `--seqs-per-microbatch 1`(要注明)。

---

## P0:Phase 2 干预 —— 10 run(0.8B,单卡/run,全可并行,无 teacher)

RLVR 5 个(把 `--intervention` 与 `--out` 按下表替换;`none` = 不加 --intervention):
```bash
$PY scripts/train.py --objective rlvr --lr 2e-6 --steps 300 --save-every 20 \
  --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 \
  --intervention spectrum_only --out $RUNS/phase2_rlvr_spectrum_only
```
| --intervention | 附加参数 | --out |
|---|---|---|
| (不加) | | phase2_rlvr_none |
| spectrum_only | | phase2_rlvr_spectrum_only |
| frame_only | | phase2_rlvr_frame_only |
| snr_topq | --intervention-q 0.1 | phase2_rlvr_snr_topq |
| mag_topq | --intervention-q 0.1 | phase2_rlvr_mag_topq |

SFT 5 个(同上五种,模板去掉 `--lr/--rollouts`,目录 phase2_sft_*):
```bash
$PY scripts/train.py --objective sft --steps 300 --save-every 20 \
  --prompts-per-step 8 --max-new-tokens 384 --seed 0 \
  --intervention spectrum_only --out $RUNS/phase2_sft_spectrum_only
```
每 run 评测(checkpoint 用 **ckpt_000300**):
```bash
$PY scripts/eval_gsm8k.py --model $RUNS/<run>/ckpt_000300 --limit 500 --out $RUNS/<run>/eval_gsm8k.json
```
**健康自检(开跑 30 分钟内看 log.jsonl)**:
- rlvr_none/frame_only:reward_mean 从 ~0.15 上行,`skip` 行 <10%
- rlvr_spectrum_only:reward 不涨甚至缓跌——**这是理论预测,不是 bug,跑满别杀**
- sft:loss 0.4-0.7,不发散;干预 run 比对照慢 20-40% 属正常(SVD 开销)
- 出现 Traceback/OOM/NaN:发日志给作者A,不要自行改参数重跑

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
每 run 评测:`$PY scripts/eval_gsm8k.py --model $RUNS/<run>/ckpt_000500 --limit 500 --out $RUNS/<run>/eval_gsm8k.json`

## P2:E2 —— SSD 消融 4 run(0.8B rlvr,300 步,seed 0)
在 e1_rlvr_ssd_s0 命令上改 `--steps 300` 并各改一处:
```bash
--optimizer ssd-muon        → --out $RUNS/e2_signvariant
--ssd-k 64                  → --out $RUNS/e2_k64
--ssd-no-align              → --out $RUNS/e2_noalign
--ssd-tail-coef 0           → --out $RUNS/e2_notail
```
各配 eval(ckpt_000300)。

## P3:E3 —— 分层 SSD 4 run(0.8B rlvr,300 步,seed 0)
```bash
$PY scripts/train.py --objective rlvr --optimizer ssd-routed --ssd-layer-range 0-5 --lr 2e-6 --muon-lr 2e-5 --steps 300 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 --out $RUNS/e3_ssd_layers0-5
# 另外三个:--ssd-layer-range 6-11 / 12-17 / 18-23,--out 对应改
```
各配 eval(ckpt_000300)。

## P4:E4 —— M2 自适应 α 3 run(0.8B,300 步,seed 0)
对照复用 P0(full=phase2_*_none,ISO 型=phase2_*_frame_only),只新跑:
```bash
$PY scripts/train.py --objective rlvr --lr 2e-6 --intervention adaptive_alpha --steps 300 --save-every 20 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 --out $RUNS/e4_rlvr_alpha
$PY scripts/train.py --objective sft  --intervention adaptive_alpha --steps 300 --save-every 20 --prompts-per-step 8 --max-new-tokens 384 --seed 0 --out $RUNS/e4_sft_alpha
$PY scripts/train.py --objective opd  --teacher Qwen/Qwen3.5-2B --intervention adaptive_alpha --steps 300 --save-every 20 --prompts-per-step 8 --max-new-tokens 384 --seed 0 --out $RUNS/e4_opd_alpha
```
各配 eval(ckpt_000300)+ MMLU:`$PY scripts/eval_mmlu.py --model $RUNS/<run>/ckpt_000300 --limit 1000 --out $RUNS/<run>/eval_mmlu.json`

## P5:4B —— 5 run(单卡 80GB/run)
```bash
# 先下载:hf download Qwen/Qwen3.5-4B 与 teacher:hf download Qwen/Qwen3.5-9B
$PY scripts/eval_gsm8k.py --model Qwen/Qwen3.5-4B --limit 200 --out $RUNS/eval_base_4b.json
$PY scripts/train.py --objective sft  --model Qwen/Qwen3.5-4B --steps 500 --save-every 25 --prompts-per-step 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/p1_4b_sft_adamw
$PY scripts/train.py --objective opd  --model Qwen/Qwen3.5-4B --teacher Qwen/Qwen3.5-9B --steps 500 --save-every 25 --prompts-per-step 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/p1_4b_opd_adamw
$PY scripts/train.py --objective rlvr --model Qwen/Qwen3.5-4B --lr 2e-6 --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/p1_4b_rlvr_adamw
$PY scripts/train.py --objective rlvr --model Qwen/Qwen3.5-4B --optimizer muon --lr 2e-6 --muon-lr 2e-5 --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/e1_4b_rlvr_muon
$PY scripts/train.py --objective rlvr --model Qwen/Qwen3.5-4B --optimizer ssd  --lr 2e-6 --muon-lr 2e-5 --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seqs-per-microbatch 1 --seed 0 --out $RUNS/e1_4b_rlvr_ssd
```

## P6:9B —— 3 run(4B 收尾后;多卡 `--device-map auto`,配方见 docs/SCALING_v1.md)
P5 的三个 Phase 1 命令把 `--model` 换成 Qwen/Qwen3.5-9B、teacher 换成
Qwen/Qwen3.5-27B(先 hf download),加 `--device-map auto`,给 3-4 张卡。

---

## 交付方式
每 run 完成后先跑 `$PY analysis/compute_metrics.py $RUNS/<run>`(生成 metrics.csv),
然后把每 run 的 4 个小文件发给作者A:`log.jsonl、args.json、eval*.json、metrics.csv`。
大文件(capture/、checkpoint)留在本机不删,不要发。
每批开跑 30 分钟内先发一次 `tail -5 log.jsonl` 给作者A 核对健康度。
