# 作者 B → 作者 A:v3 RLVR 结果包(2026-09-16)

B 这边不能 push,本包经用户转交。**本包只含结果,不含代码**(补丁、脚本、git bundle 均未放入;
`NOTES_batch1.md` 与 `reviews/B_review_of_A_sft_v3.md` 中原有的源码片段已移除,
其余 md 中提到的脚本路径仅作出处说明,文件本身不在包内)。

## 包内容

| 路径 | 内容 |
|---|---|
| `v3_rlvr_*`(21 个) | 批次一 7 个(dense lr×1/30、×1/10、×1/3)+ 批次二/三 14 个 run 目录:`args.json`、`log.jsonl`、`eval_*.json`、`metrics.csv`、`manifest.json`、`summary.txt`、`dtype_check.txt` |
| `health_v3_rlvr_gate.csv` / `health_v3_rlvr_batch23_gate.csv` | 批次一 / 批次二三 gate 表 |
| `health_v3_sft_gate.csv/.txt` | B 侧对 A 的 dense SFT run 的 gate 复算 |
| `mmlu_stability_*.txt` | MMLU 单点 vs 末 k 个 ckpt 的稳定性对照(rlvr、rlvr_batch23、sft) |
| `cum_kl/` | E-v3-0a 输出 |
| `mmlu_format/` | E-v3-0b 输出 |
| `exact_iso_spectrum_check.txt` | exact_iso 奇异值偏移检查的原始输出(见第 1 节) |
| `README_batch1.md` / `NOTES_batch1.md` / `README_batch23.md` | 各批次说明与过程记录 |
| `reviews/` | B 对 A 的 SFT 裁决的复核;B 对 RLVR 批次二三的结论 |

## 1. 最重要:exact_iso 并不等谱,RLVR H-2 的“保谱 vs 去谱”解释站不住

B 在 A100 上的 step-300 checkpoint,fp64 奇异值相对 base 的变化(8 个矩阵中位数,
原始输出见 `exact_iso_spectrum_check.txt`):

| run | ‖s−s0‖/‖s0‖ | 相对 dense |
|---|---|---|
| dense full s0 | 9.31e-6 | 1× |
| frame_matched s0 | 6.92e-8 | 约 135× 更贴近 base |
| exact_iso s0 | 3.08e-4 | **33×** |
| exact_iso s1 | 3.06e-4 | **33×** |

- 两个 seed 逐矩阵一致到约 1%,偏移在 step 20 就出现且保持不变:是固定的数值偏差,不是训练效应。
- exact_iso 每步实际更新约为 dense 的 **12 倍**,其中约 99% 是这一固定扰动;v2 的 `e4a_*_exact_iso` checkpoint 同样有此偏移。
- 位置:`specgeom_v3/intervene_engine.py` 的 exact_iso 分支每步对整个矩阵做 GPU float32 SVD 并用锚定谱重建。最可能是 float32 GPU SVD 的重建误差,**未证实**。
- **A 的 exact_iso 用的是同一引擎,很可能有同样的问题。** A 可在自己的 exact_iso 与 dense 的 `ckpt_000300` 上做同样的奇异值比较:exact_iso 应当不高于 dense;若约为 3e-4,就是同一缺陷。
- 因此 exact_iso 在 RLVR 上 GSM8K 最高,**不能归因于“保住了谱”**。修好后 exact_iso 的臂需要重跑,或把 exact_iso 从 C4 证据中拿掉。

## 2. B 对 A 的 RLVR 结果的独立复现(B:A100-80G;A:MIG 3g.40gb)

GSM8K 列为各 seed 终值,MMLU 为均值。A 的数字取自 `origin/main` 51ffba3。

| 配置 | B GSM8K | B 均值 | B MMLU | A GSM8K | A 均值 | A MMLU | A−B |
|---|---|---|---|---|---|---|---|
| dense full | 0.626 / 0.640 / 0.654 | 0.640 | 0.470 | 0.656 / 0.634 / 0.664 | 0.651 | 0.472 | +0.011 |
| frame_matched | 0.650 / 0.644 | 0.647 | 0.471 | 0.628 / 0.610 / 0.636 | 0.625 | 0.475 | -0.022 |
| exact_iso | 0.706 / 0.646 | 0.676 | 0.472 | 0.702 / 0.690 / 0.648 | 0.680 | 0.468 | +0.004 |
| spectrum_matched | 0.554 / 0.546 | 0.550 | 0.479 | 0.554 / 0.572 | 0.563 | 0.477 | +0.013 |
| random_ext | 0.556 / 0.558 | 0.557 | 0.476 | 0.546 / 0.564 | 0.555 | 0.477 | -0.002 |
| spectrum s_rel=3 | 0.550 | 0.550 | 0.481 | 0.570 | 0.570 | 0.478 | +0.020 |
| random s_rel=3 | 0.560 | 0.560 | 0.479 | 0.558 | 0.558 | 0.478 | -0.002 |
| spectrum s_rel=10 | 0.608 | 0.608 | 0.473 | 0.594 | 0.594 | 0.475 | -0.014 |
| random s_rel=10 | 0.588 | 0.588 | 0.470 | 0.586 | 0.586 | 0.474 | -0.002 |
| spectrum lr×1 | 0.614 | 0.614 | 0.478 | 0.578 | 0.578 | 0.474 | -0.036 |
| random lr×1 | 0.624 | 0.624 | 0.475 | 0.620 | 0.620 | 0.478 | -0.004 |

**H-2 的关键差值(GSM8K 均值):**

| | exact_iso − dense | frame_matched − dense | exact_iso − frame_matched |
|---|---|---|---|
| B | +0.036 | +0.007 | +0.029 |
| A | +0.029 | -0.027 | +0.055 |
| 两边合并 | +0.033 | -0.012 | +0.045 |

- **exact_iso 最高在两边都复现。** 但见第 1 节,它不是等谱对照。
- **frame_matched 两边方向不一致**:B 上与 dense 基本持平,A 上低于 dense。A 的 iso−frame t=3.2 有一部分由 A 这边 frame_matched 偏低驱动;加入 B 的 run 后差值为 +0.045。
- **同一 seed 跨硬件并不复现**:17 组同 seed 同配置对的 GSM8K 绝对差,中位数 0.010、最大 0.044,与换 seed 的差异同量级。两边合并时应按独立样本处理,不能做同 seed 配对。

## 3. Gate 结果(批次二/三)

- exact_iso lr×1/10:**FAIL**(seed 离散 0.060)。
- spectrum_matched s_rel=10:**FAIL**(decline 0.032,阈值 0.03)。
- 其余 9 个配置 HEALTHY;MMLU 稳定性检查全部 PASS。

## 4. 两边一致的结论

- **H-0**:lr* = 2e-7 两边都健康。B 另有 lr×1/3(FAIL:离散 0.072、MMLU 0.452)与 lr×1/30(HEALTHY,decline 余量仅 0.0005)的完整三档,见 `README_batch1.md`。
- **H-1**:在 lr* 上 r 维两臂都停在 base 附近(约 0.01 以内),**无检验力**;在 lr×1 上三 seed 两字典差 +0.024,在 margin 内,成立。
- **H-3 / C5**:r 维响应步长(s_rel=10 或 lr×1 时 GSM8K 升到 0.58–0.62),但落在 dense 的 lr 前沿上、不占优。

## 5. A 正在重复的工作(B 已有)

- 51ffba3 中的 RLVR dense gate lr×1/3、lr×1/30:B 批次一已完成,run 目录与 `health_v3_rlvr_gate.csv` 在本包内。
- E-v3-0b 在 v3 checkpoint 上:B 已对 `v3_rlvr_full_lr0.33_s0/s1` 做过(`mmlu_format/`):作答格式完好(0.865,与 base 相同),准确率下降(0.430/0.425 对 0.475),是真实能力损失而非格式崩坏。
- E-v3-0a:B 做的是 v2 `e4a_*` 的 18 个 checkpoint(`cum_kl/`),**没有**在 v3 checkpoint 上做,这部分 A 的重跑不重复。

## 6. 环境与 dtype

- torch 2.11.0+cu128,**transformers 5.16.1**,A100-SXM4-80GB。
- 所有训练与评测任务均带 `HAS_TT_DATA=False`。
- 21 个 v3 RLVR run 的 `ckpt_000300` safetensors **全部为 F32**,见各 run 目录 `dtype_check.txt`。
- 批次二/三 14 个 run:0 失败、0 次评测跳过、0 次驱动故障。批次一曾有主机驱动故障,详见 `NOTES_batch1.md`。
