# results_B_v2 — 作者B 的 TASKS_B_v2 交付(2026-09-11)

范围:`docs/TASKS_B_v2.md` 的 P0–P5 全部完成;P6(SSD 学习率扫描,可选)未跑。
每个 run 一个目录,小文件:`log.jsonl、args.json、eval_gsm8k.json、eval_mmlu.json、
metrics.csv、manifest.json、summary.txt`。capture 与 checkpoint 留在 B 的 NAS
(`/mnt/bn/genai-nebula/kaifan.fk/spectral/runs/<run>/`),未删,P3 的 capture 全部保留。

评测口径:GSM8K test 前 500 题 greedy pass@1;MMLU 1000 题 zero-shot letter-logit。
同口径基线:**0.8B = 0.546 / 0.483**,4B = 0.862 / 0.698。

## 环境、代码版本、硬件

- **库**:`/home/tiger/venv/py311`,torch 2.6.0+cu124、transformers 5.9.0,外加
  `load_model` 里的显式 fp32 转换与 dtype 断言。transformers 5.9.0 会忽略
  `from_pretrained(dtype=torch.float32)`,不加转换就是 bf16 master(见 P0 答复 Q2)。
- **fp32 已逐个核实**:18 个 e4a run 的 `ckpt_000300` safetensors 头全部是 **F32**(2.8 GB);
  对照 v1 的 `phase2_rlvr_none` 是 BF16(1.4 GB)。
- **与 A 的版本(transformers 5.16.1)**:已建对齐环境
  `/mnt/bn/genai-nebula/kaifan.fk/spectral/venv_v2`(torch 2.11.0+cu128),worker 上可用。
  两版本前向 A/B:权重逐位相同,top-1 一致率 100%,KL 3.8e-4(与 A5 测的精度地板同量级),
  判定**无需重跑**;为保证 P3 批内一致,整批留在原环境。
- **代码**:16 个 P3/P4 run 用的 `specgeom_v2/ scripts_v2/ analysis_v2/` 与 commit **1a1e2f9**
  的内容一致(2026-09-10 04:30 UTC 同步到 NAS,之后才提交);2 个补跑的 fp32 full 对照
  用的是合并后的 **c4582b9**,只差 `modeling.py`,两者都是"显式转 fp32 + 断言",功能等价。
  注意 `worker.log` 里的 `commit=` 为空,因为 NAS 上的代码副本不带 `.git`。
- **硬件**:每 run 1 张 A100-SXM4-80GB 或 H100 80GB HBM3(`mlx worker`,共享 NAS)。

### 秒/步(v2 run,全程中位数)

| 配置 | A100 | H100 |
|---|---|---|
| RLVR full | 27.4 / 30.5 | — |
| RLVR spectrum_matched | 45.1 / 46.4 | — |
| RLVR frame_matched | 42.8 | 32.8 |
| RLVR random_ext | 33.0 | 25.8 |
| RLVR exact_iso | 41.5 | — |
| SFT full | 5.5 | 5.1 |
| SFT spectrum_matched | — | 13.7 / 14.0 |
| SFT frame_matched | — | 13.5 / 13.7 |
| SFT random_ext | — | 4.8 / 5.0 |
| SFT exact_iso | 18.4 | — |

spectrum/frame_matched 慢是因为每步要重算全部 186 个矩阵的 SVD 基;random_ext 用固定随机
字典、不刷新基,速度与 full 相同。v1 各配置的秒/步见 `B_P0_answers_v2.md` Q3。

## 与清单的偏离

1. **P3 的 seed 0 full 对照是新跑的**(`e4a_rlvr_full_s0`、`e4a_sft_full_s0`,fp32),没有复用
   v1 的 `phase2_*_none`。v1 是 bf16 master,舍入丢掉了大部分更新,与 fp32 不可比:
   RLVR full 在 v1 为 0.636 / 0.477,在 fp32 为 0.412–0.494 / 0.232–0.237。
2. **`specgeom_v2/intervene_engine.py` 有两处修改**(已提交):`observe_grad()` 首步的基初始化
   (否则 snr_topq/adaptive_alpha 首步 KeyError);依赖 SVD 基的模式改为每步刷新基。
   因此这里的 spectrum/frame_matched 是在当前步的精确基里投影的。
3. **P0.4 的 smoke 跑的是 4 步而不是 20 步**。四个模式都正常:scale_mean 分别约 54
   (spectrum_matched)、1.0(frame_matched)、52(random_ext),exact_iso 正常。
   H 自检在正式 run 上核验:`e4a_rlvr_spectrum_matched_s0` median R_sigma(H) = **0.935,OK**。
   剩余 6.5% 来自捕获里 W 以 bf16 存储,建议 v3 改存 fp32。
4. **P2 在 GPU worker 上重算**,不是开发机 CPU(开发机读 NAS 太慢)。
5. **`p1_4b_rlvr_adamw` 没有 v2 指标**(不在 P2 清单),目录里放了 v1 格式的 `metrics_v1.csv`。
   `e1_opd_adamw_s1` 只按 P1 要求补了 MMLU,没有 metrics/manifest。
6. **P5**:`e1_4b_rlvr_muon` 训练早已完成,按"已在跑的可以跑完并交付"补了评测;
   `e1_4b_rlvr_ssd` 停在第 378 步,未交付;`p1_4b_opd_adamw` 按处置表未跑。
7. **P0 Q1 的重评**:6 个 v1 checkpoint 用含 stop-ids 修复的脚本重评,结果在各目录的
   `eval_gsm8k_v2.json`(差 ≤1.6 点,v1 数据可用)。

## 结果

### P1 补评测(MMLU;这三个都是 v1 的 bf16 master run)

| run | MMLU |
|---|---|
| e1_opd_adamw_s1(OPD full,500 步) | 0.247 |
| phase2_rlvr_none | 0.477 |
| phase2_rlvr_frame_only | 0.482 |

OPD full 自身就把 MMLU 打到接近随机,`e4_opd_alpha` 的 0.262 与之持平 —— α 没有减少遗忘。

### P3 + P4 等范数干预(0.8B,300 步,fp32;seed 0 / seed 1)

RLVR:

| 干预 | GSM8K | MMLU |
|---|---|---|
| full | 0.412 / 0.494 | 0.237 / 0.232 |
| spectrum_matched | **0.630 / 0.638** | **0.473 / 0.475** |
| frame_matched | 0.584 / 0.472 | 0.248 / 0.228 |
| random_ext | 0.606 / 0.580 | 0.474 / 0.476 |
| exact_iso | 0.600 / — | 0.255 / — |

SFT:

| 干预 | GSM8K | MMLU |
|---|---|---|
| full | 0.298 / 0.304 | 0.239 / 0.239 |
| spectrum_matched | **0.382 / 0.392** | **0.471 / 0.484** |
| frame_matched | 0.278 / 0.288 | 0.235 / 0.236 |
| random_ext | 0.378 / 0.408 | 0.467 / 0.466 |
| exact_iso | 0.290 / — | 0.240 / — |

读法(给 A 写 §9.4 参考,判定以 A 为准):

- **分界线是秩,不是"谱 vs frame"**。同一 Frobenius 范数下,限制在 r 个秩一方向上的更新
  (spectrum_matched、random_ext)保住 MMLU(约 0.47,base 0.483)且 GSM8K 更高;全秩更新
  (full、frame_matched、exact_iso)把 MMLU 打到随机水平(0.23–0.26)。
- **random_ext 与 spectrum_matched 基本打平**(RLVR 0.606/0.580 对 0.630/0.638,SFT 0.378/0.408
  对 0.382/0.392),即随机的 r 维字典与权重的奇异基效果接近。这对"谱方向本身特殊"是反例,
  更像"低秩约束本身在起作用"。
- **frame_matched ≈ full**(scale ≈ 1.0,谱分量只占更新能量的约 1/2500),两者行为一致在意料之中。
- fp32 下 300 步的 full 更新在两个 objective 上都低于 base,v1 里"RLVR full 提升到 0.636"
  很大程度上来自 bf16 舍入带来的隐式小步长。

### P5(4B,500 步;v1 代码,bf16 master)

| run | GSM8K |
|---|---|
| base | 0.862 |
| p1_4b_sft_adamw | 0.818 |
| p1_4b_rlvr_adamw | 0.932 |
| e1_4b_rlvr_muon | 0.918 |

### P2

24 个 v1 run 全部用 `analysis_v2/compute_metrics.py` 重算,新列与 `manifest.json`
在各自目录;v1 的 metrics.csv 未覆盖。注意:v1 run 是 bf16 master,**H 侧列不可用**,
G 侧列不受影响。
