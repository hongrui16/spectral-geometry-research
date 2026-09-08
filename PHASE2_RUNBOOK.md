# Phase 2 Runbook(作者B 执行)

自包含文档:按此文档从零开始即可完成 Phase 2 全部 10 个 run 并交付。
预计总墙钟(卡够全并行):**半天**。有问题随时把日志发给作者A。

---

## 1. 这组实验在验证什么

论文核心分解(`unified_paper_document.md` §3):权重 W = UΣV^T,任何一步更新 H 都可以
分成"改奇异值 σ(spectrum,增益)"与"转奇异向量 U,V(frame,路由)"两部分。
Phase 2 通过**在训练循环里改写每步实际更新**做因果检验:

| 干预 | 数学操作(对每个 decoder 矩阵) | 检验 |
|---|---|---|
| `spectrum_only` | H → U·diag(diag(U^T H V))·V^T(只保留对角/谱分量) | H5 |
| `frame_only` | H → H − spectrum_only(H)(只保留旋转分量) | H5 |
| `snr_topq` | 谱系数 C=U^T H V,只保留 SNR 排前 q% 的项(SNR 由梯度的跨步 EMA 估计:m²/v) | H6 |
| `mag_topq` | 同上,但按幅值 |C| 排前 q%(Muon/Pion 的隐含准则) | H6 |

**预测**(可证伪):RLVR 下 spectrum_only 学不动(改增益建不了新关联,Prop.2)、
frame_only ≈ 完整训练;同 q 下 snr_topq > mag_topq,且 RLVR 差距最大(§5)。
SFT 下两组差距都应显著小于 RLVR。

实现:`specgeom/intervene_engine.py`。U,V 基每 10 步重算一次(更新小,冻结基是一阶
精度近似);干预作用于全部 decoder 权重矩阵(~170 个),不只是被追踪的 27 个。

## 2. 资源需求

- **模型**:只用 `Qwen/Qwen3.5-0.8B`(Phase 2 没有 OPD,不需要 teacher)。
- **每 run 单卡**:40GB 显存可跑,80GB 更稳(`snr_topq` 的 EMA 状态多占 ~5GB)。
- **10 个 run 相互独立**,卡够就全并行。
- 磁盘:每 run 的 capture 约 12-15GB(15 个保存步 × ~820MB),10 run 预留 ~200GB。

| run 类型 | 训练时长(A100 参考) | 评测 | 合计 |
|---|---|---|---|
| RLVR 类 ×5 | ~3.5-4h(干预 run 比对照慢 20-40%,SVD 开销,正常) | ~1h | ~5h |
| SFT 类 ×5 | ~1-1.5h | ~1h | ~2.5h |

## 3. 环境(一次性,约 20 分钟)

```bash
git clone git@github.com:hongrui16/spectral-geometry-research.git && cd spectral-geometry-research

python3.10 -m venv ~/envs_spectral          # 任何 python>=3.10
~/envs_spectral/bin/pip install -U pip
~/envs_spectral/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
~/envs_spectral/bin/pip install -U transformers trl datasets accelerate math-verify matplotlib pandas tensorboard
# transformers 必须 >= 5.x(qwen3_5 架构);装完可 python -c "import transformers; print(transformers.__version__)" 确认

export HF_HOME=/你的大盘/hf_home            # 写进 ~/.bashrc 或每次 export
~/envs_spectral/bin/hf download Qwen/Qwen3.5-0.8B
~/envs_spectral/bin/python -c "from datasets import load_dataset; load_dataset('openai/gsm8k','main')"
```

跑训练时统一加:
```bash
export HF_HOME=/你的大盘/hf_home
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUNS=/你的输出根目录
```

## 4. 10 个 run 的完整命令

**规则:除 `--intervention*` 与 `--out` 外,不要改动任何参数。**
seed 固定 0;RLVR 必须 `--lr 2e-6`(默认 1e-5 已实测 40 步内策略崩塌);SFT 用默认 lr。

RLVR 模板(5 个,替换 `INT` 与目录名):
```bash
# INT ∈ {none 不加 --intervention 参数, spectrum_only, frame_only, snr_topq, mag_topq}
CUDA_VISIBLE_DEVICES=0 ~/envs_spectral/bin/python scripts/train.py \
  --objective rlvr --lr 2e-6 --steps 300 --save-every 20 \
  --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 \
  --intervention spectrum_only \
  --out $RUNS/phase2_rlvr_spectrum_only
```
5 个目录名:`phase2_rlvr_none`(无 --intervention)、`phase2_rlvr_spectrum_only`、
`phase2_rlvr_frame_only`、`phase2_rlvr_snr_topq`、`phase2_rlvr_mag_topq`。
topq 两个额外加 `--intervention-q 0.1`。

SFT 模板(5 个,同样五种,去掉 `--lr/--rollouts`):
```bash
CUDA_VISIBLE_DEVICES=1 ~/envs_spectral/bin/python scripts/train.py \
  --objective sft --steps 300 --save-every 20 \
  --prompts-per-step 8 --max-new-tokens 384 --seed 0 \
  --intervention spectrum_only \
  --out $RUNS/phase2_sft_spectrum_only
```
目录名:`phase2_sft_none / _spectrum_only / _frame_only / _snr_topq / _mag_topq`。

每个 run 结束后评测(用同一张卡即可,~1h):
```bash
~/envs_spectral/bin/python scripts/eval_gsm8k.py \
  --model $RUNS/phase2_rlvr_spectrum_only/ckpt_000300 --limit 500 \
  --out $RUNS/phase2_rlvr_spectrum_only/eval_gsm8k.json
```
(checkpoint 会存 `ckpt_000250` 和 `ckpt_000300` 两个,评测用 **000300**。)

## 5. 健康自检(每个 run 开跑 30 分钟内做一次)

看 `$RUNS/<run>/log.jsonl` 最后几行,每行形如:
```json
{"step": 42, "loss": -0.03, "secs": 33.4, "reward_mean": 0.28, "n_groups_kept": 5}
```
- **rlvr_none / frame_only**:`reward_mean` 应从 ~0.15 缓慢上行;`n_groups_kept` 多在 3-6;
  `{"skip": "no rows"}` 行占比 < 10%。
- **rlvr_spectrum_only**:reward 不涨、甚至缓跌,skip 变多——**这是 H5 的预测行为,不是 bug**,
  不要中途杀掉,跑满 300 步(它的"失败"正是论文要的数据点)。
- **sft 各 run**:loss 应稳定在 0.4-0.7 区间,不应发散(>2 或 NaN 说明有问题)。
- 干预 run 单步时间比对照多 20-40% 属正常(每步全矩阵投影 + 每 10 步 SVD 刷新)。
- 任何 Traceback / OOM / NaN:截图 + log.jsonl 发作者A,不要自行改参数重跑。

常见问题:
- **OOM**:加 `--seqs-per-microbatch 1`(仅此参数允许自行调整,调了要在交付说明里注明)。
- **HF 报连不上网**:确认 `HF_HUB_OFFLINE=1` 且模型已预下载在 `HF_HOME`。
- **首步特别慢**:模型 fp32 加载 + 编译预热,正常。

## 6. 交付物
每 run 完成后跑:`$PY analysis/compute_metrics.py $RUNS/<run>` 然后
`$PY scripts/summary.py $RUNS/<run>`,**把打印出的那行文字发给作者A**。
capture/ 留在本机不删;不要改 analysis/ 代码。

## 7. 时间线

- 开跑:**现在即可**,不必等 0.8B Phase 1 收尾(实验独立成立;若作者A 今晚的 H1 判定
  推翻预期,会另行通知,已跑的对照/干预数据照样可用)。
- 期望交付:开跑后 24 小时内(含评测)。
- 优先级:RLVR 5 个 > SFT 5 个(RLVR 是论文主叙事;卡不够先跑 RLVR 组)。
