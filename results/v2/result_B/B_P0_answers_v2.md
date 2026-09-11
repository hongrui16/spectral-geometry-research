# 作者B — TASKS_B_v2 P0 三个先决问题的答复(2026-09-10)

## Q1 代码版本 / greedy 非终止修复 —— **v1 数据可用,不必作废**

v1 批次跑的是 **e5a25bc**(我 09-08 21:30 从 main 同步到 NAS 的 HEAD)。
`git merge-base --is-ancestor adf1786 e5a25bc` = **否**,即 v1 批次**不含** stop-ids 修复。

但实测影响可以忽略。我用含修复的 `scripts_v2/eval_gsm8k.py` 重评了 6 个 v1 checkpoint
(覆盖 SFT / RLVR / 4B、300 步与 500 步),同一 ckpt、同一 500 题:

| run | v1 评测(无修复) | v2 评测(有修复) | 差 |
|---|---|---|---|
| phase2_rlvr_none | 0.636 | 0.636 | 0 |
| phase2_sft_none | 0.412 | 0.404 | −0.008 |
| phase2_sft_spectrum_only | 0.354 | 0.344 | −0.010 |
| e1_rlvr_adamw_s1 | 0.670 | 0.670 | 0 |
| e1_sft_adamw_s1 | 0.438 | 0.454 | +0.016 |
| p1_4b_sft_adamw | 0.818 | 0.818 | 0 |

差异 ≤1.6 个点、方向有正有负,属于 greedy 解码在边界样本上的抖动。
**结论:v1 的所有 SFT 行可用;SFT 低于 base 是真实效应,不是评测 artifact。**
作者A 那边 57→28 的情况在我的 checkpoint 上没有复现(可能与 A 的 checkpoint 的
generation_config 或 max_new_tokens 设置有关,建议 A 也做一次同 ckpt 的 A/B)。

## Q2 捕获顺序 / H 自检 —— **顺序没问题,根因是模型根本不是 fp32**

三件事分开说:

1. **顺序确认无误**。v1 `scripts/train.py` 里 `engine.post_step()`(第 398 行)在
   `instr.post_optimizer()`(第 401 行)之前;v2 `scripts_v2/train.py` 同样(406 / 408 行)。
   `instr.pre_optimizer()` 也确实在 `engine.pre_step()` 之前,两者的 `_w_before` 是同一个 W。

2. **基不是问题**。GPU 上直接对比:引擎自己用的基与从捕获的 W 重算的基**逐位相同**
   (‖U_engine − U_capture‖ = 0.0);`R_spectrum(W)` 恰好 = 1.0;手算的谱投影在该基下
   R = 1.0001。我还排除了基漂移(每步刷新与 10 步刷新的自检值都是 0.215,完全没变)
   和 bf16 存 W(4e-3 量级的基扰动只让对角能量掉 4%,不足以解释 0.22)。

3. **真正的根因:`transformers 5.9.0` 忽略 `from_pretrained(dtype=torch.float32)`。**
   实测 Qwen3.5-0.8B 载入后 **320 个参数全是 bfloat16**,`dtype=` 和已废弃的
   `torch_dtype=` 都无效。于是:
   - master 权重是 bf16 → `H = W_after − W_before` 是两个几乎相等的 bf16 数相减,
     **捕获到的 H 主要是量化噪声**。数值:某矩阵每个权重元素约 4.8e-3,bf16 在该量级
     的量化步长约 1.9e-5,而每步更新的元素量级约 1e-5 —— 更新和量化噪声同量级。
   - 因此 `R_sigma(H)` 掉到 0.2 与基、与干预模式都无关,而且**这条同样适用于 v1**:
     `results/v1` 里所有 H 侧指标(`H_r_spectrum`、`H_stable_rank`、Fig.3 的 G-vs-H)
     都建立在量化噪声上,不能用。G 侧指标不受影响(G 是直接捕获的,bf16 存储只带来
     4e-3 的相对误差,没有相消)。
   - 更严重的是,**谱型干预实际上没有被正确写入**:`p.copy_(w_before + Hp.to(p.dtype))`
     把投影后的更新转成 bf16 再加到 bf16 权重上,大部分被舍入掉。v1 的 spectrum_only
     "reward 不动"里有多少是理论预测、有多少是 bf16 舍入,目前分不开。

   **已修并验证**:`specgeom_v2/modeling.py` 在 `from_pretrained` 之后显式
   `model.to(dtype=...)`,并在参数 dtype 不符时直接报错(不再静默降级)。修复后 320 个
   参数全为 fp32。用 fp32 重跑的 `e4a_rlvr_spectrum_matched_s0` 第一个保存步做自检:

   | | median R_sigma(H) | 判定 |
   |---|---|---|
   | 修复前(bf16 master) | 0.215 | FAILED |
   | 修复后(fp32 master) | **0.935** | **OK** |

   剩下的 6.5% 缺口来自捕获里 W 仍以 bf16 存储(分析时重算 SVD 的基与引擎用的 fp32 基
   有约 4e-3 的差异,实测代价约 4%)。要做到 ≈1.0 需要把 `instrument.py` 里的 W 改存 fp32
   (capture 体积 +9%);本批已开跑,为保持一致我没有中途改,建议 v3 再改。
   `docs/SCALING_v1.md` 里"必须保 fp32 master"的要求,在此之前从未真正生效过。

**建议**:A 侧请用同一段代码自查一次(`load_model` 后打印参数 dtype)。若 A 的环境
也是 transformers 5.x,那么 v1 的 Phase 1 也是 bf16 master,H 侧结论需要重做;
G 侧(H1/H2 的 G 部分、H4)不受影响。

## Q3 硬件与实测速度

worker 由 `mlx worker launch` 从配额组拉起(A100-SXM-80GB / H100-SXM-80GB 各 1 卡一个 run),
共享 NAS。秒/步为该 run 全程中位数(去掉第 1 步):

| 配置 | H100 | A100 |
|---|---|---|
| SFT 0.8B(AdamW) | 4.8 | 5.8 |
| SFT 0.8B + 干预(v1 spectrum_only) | 5.0 | 6.2 |
| OPD 0.8B(teacher 2B) | 20.0 | 23.5 |
| OPD 0.8B + SSD | 28.4 | — |
| RLVR 0.8B(AdamW,K=8) | 25.8–27.1 | 33.2 |
| RLVR 0.8B + SSD | 38.3 | 35.8 |
| RLVR 0.8B ssd-routed | — | 36.8 |
| SFT 4B(seqs-per-microbatch 1) | — | 7.9 |
| RLVR 4B(AdamW) | — | 38.9 |
| RLVR 4B(Muon) | — | 49.1 |

v2 的等范数干预因为每步要刷新 SVD 基,RLVR 0.8B 约 40 秒/步(A100);`exact_iso`
在 SFT 上约 15–18 秒/步(是无干预的 5 倍,不是文档写的 2 倍)。

另:`analysis_v2/compute_metrics.py` 重算 v1 捕获在**开发机 CPU 上跑不动**——瓶颈是
NAS 读带宽(实测冷读约 17–50 MB/s,单个 RLVR run 的 capture 约 16 GB),6 路并行
2 小时零产出。已改为在 GPU worker 上跑(离存储近 + SVD 上 GPU)。

## 另外两个 v2 代码问题(已修,见 `specgeom_v2/intervene_engine.py`)

1. `observe_grad()` 在首次 `pre_step()` 之前被调用,而 SVD 基只在 `pre_step()` 里建,
   `snr_topq` 与 `adaptive_alpha` 会在第 1 步 KeyError。v1 里我修过,v2 复制时丢了。
2. 依赖 SVD 基的模式(spectrum/frame/top-q/adaptive_alpha)原本每 10 步才刷新基。
   谱-frame 的对角划分在近简并谱下对基敏感,定义上必须用当前步的基,已改为每步刷新;
   `random_ext`(固定随机字典)和 `exact_iso`(不用该基)保持不刷新,并让 `random_ext`
   提前短路,省掉每步一次白算的 SVD。
   注:这一条不是 H 自检失败的原因(见 Q2),但它是正确性问题。
