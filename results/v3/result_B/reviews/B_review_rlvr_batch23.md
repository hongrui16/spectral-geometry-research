# B 对 RLVR 批次二+三的判读与 `exact_iso` 引擎缺陷(2026-09-16)

**方法**:4 路独立检查(交付审计 / H-2 / 干预幅度 / H-1),对每条结论与每个问题做对抗验证(18 个 agent);`exact_iso` 的谱漂移由 B 本人在 CPU 上用 fp64 `svdvals` 独立重算。以下只列经验证后仍成立的结论,出处逐条标注。

## 1. `exact_iso` 引擎有数值缺陷(两名验证者未能推翻 + B 亲手复算)

`ckpt_000300` 相对 base 的奇异值变化 ‖s−s0‖/‖s0‖,8 个矩阵(L3/L7/L12/L20 的 mlp 与 linear_attn)的中位数:

| run | 中位数 | 相对 full |
|---|---|---|
| full_s0 | 9.31e-6 | 1× |
| frame_matched_s0 | 6.92e-8 | 约 135× 更接近 base |
| exact_iso_s0 | 3.08e-4 | **33.1×** |
| exact_iso_s1 | 3.06e-4 | **32.9×** |

两个 seed 逐矩阵一致到约 1%,偏移在 step 20 就已出现且保持不变:这是固定的、与 seed 无关的数值偏差,不是训练效应。验证者另测得每步实际更新约为 dense 的 12 倍,其中约 99% 的能量是这一固定扰动;v2 的 `e4a_rlvr_exact_iso_s0` 与 `e4a_sft_exact_iso_s0` checkpoint 有同样的偏移。

代码(`specgeom_v3/intervene_engine.py`):第 54–56 行构造时用 `svdvals(p.float())` 取锚定谱(GPU、float32);第 98–103 行每步 `svd(W_after)`(GPU、float32)后用 `Ua @ diag(S_anchor) @ Vah` **重建整个矩阵**。最可能的原因是 float32 GPU SVD 的重建误差,**但未证实**——验证者在 CPU 上用 fp32 模拟同样的重置是准确的。候选修正(未验证):SVD 用 float64;或只施加增量 `W_after + Ua·diag(S_anchor − S_after)·Vah`,避免从 float32 因子重建整个矩阵。

**影响 A 的 SFT H-2**:SFT 那边同样依赖 exact_iso 臂“落在同一点”,而该臂同样不等谱。

## 2. RLVR H-2(lr* = 2e-7)

| 臂 | GSM8K | 均值 | MMLU | 均值 |
|---|---|---|---|---|
| full(批次一) | 0.626 / 0.640 / 0.654 | 0.6400 | 0.468 / 0.467 / 0.474 | 0.4697 |
| frame_matched | 0.650 / 0.644 | 0.6470 | 0.471 / 0.471 | 0.4710 |
| exact_iso | 0.706 / 0.646 | 0.6760 | 0.466 / 0.478 | 0.4720 |

- exact_iso − full:GSM8K **+0.0360**,MMLU +0.0023。GSM8K 超过 0.03 margin,按预注册规则字面判 **H-2 不成立**(两名验证者未能推翻)。
- 但:这是点估计越线(t≈1.79,主要由 seed 0 贡献),不是可测效应;2–3 个 seed 下 `|t|<2` 这一条永远不起作用,规则退化为纯点估计,“成立”不代表等价、“不成立”也不代表有差异。
- **越线的恰好是有缺陷的那个臂。**
- frame_matched − full:GSM8K +0.0070,MMLU +0.0013,不可分。frame_matched 是有效对照:其奇异值保持在 base 附近(见上表),“近乎空干预”的说法已被对抗验证驳回。

**建议**:要么修好 exact_iso 后重跑它的四个臂(RLVR ×2、SFT ×2),要么把 exact_iso 从 C4 证据中拿掉、只用 frame_matched 论证 H-2。由 A 决定。

## 3. H-1:在 lr* 上无检验力,在 lr×1 上成立

- lr*:spectrum_matched GSM8K 0.554 / 0.546(均 0.550)/ MMLU 0.480 / 0.478;random_ext 0.556 / 0.558(均 0.557)/ MMLU 0.475 / 0.478。两者差 GSM8K -0.0070、MMLU +0.0025。
  但两个 r 维臂都在 base(0.546 / 0.483)的约 0.01 以内——**几乎没有训练动**。在这里比较“不可分”没有检验力,不应写成“H-1 在 lr* 复现”(经对抗验证)。

- lr×1(2e-6;seed 0/1 为 v2 `e4a_rlvr_*`,seed 2 为 v3 `lr1_s2`):spectrum_matched GSM8K 0.630 / 0.638 / 0.614(均 0.6273)/ MMLU 均 0.4753;random_ext 0.606 / 0.580 / 0.624(均 0.6033)/ MMLU 均 0.4750。
  差 GSM8K **+0.0240**、MMLU +0.0003,均在 margin 内。这里 r 维臂确实有推进(GSM8K 升到 0.58–0.64)。

## 4. r 维响应步长(与 A 对 SFT 的“不响应”表述相反)

- spectrum_matched:s_rel=1 @lr* 0.554/0.480;s_rel=3 0.550/0.481;s_rel=10 0.608/0.473;lr×1 s2 0.614/0.478
- random_ext:s_rel=1 @lr* 0.556/0.475;s_rel=3 0.560/0.479;s_rel=10 0.588/0.470;lr×1 s2 0.624/0.475

RLVR 上 GSM8K 随步长明显上升,MMLU 基本不动。单 seed,仅供参照。

## 5. 提出后被对抗验证驳回的

- “exact_iso 未通过 seed 离散判据”作为独立问题:数字可复现,但该判据用于选 lr,不是逐臂准入测试。
- “流水线不可能发现 exact_iso 问题”:日志已自动记录了暴露问题的数字,只是没有自动判据。
- “frame_matched 近乎空干预、证据分量低”:checkpoint 显示它确实保住了谱。
- “lr×1 两 seed 数据已满足 H-1 拒绝条件”:第三个 seed 落地后被推翻。

