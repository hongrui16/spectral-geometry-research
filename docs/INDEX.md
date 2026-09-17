# 文档索引

仓库里所有文档、交付说明与登记处的入口。版本规则见 `README.md`:v1 已冻结,
v2 为当前版本;旧版本文件只改引用不改内容。

## v4(2026-09-17 起;同日暂停,见企划书第三部分)

| 文件 | 内容 | 作者 |
|---|---|---|
| `docs/unified_paper_document_v4.md` | v4 企划书(白话):第一部分现阶段工作总结(分析、优化方法、实验结果);第二部分改进方案:评估、三路调研结论、"遗忘是走了多远不是往哪走"的因果检验 / 8 步遗忘探针 / 按功能步长做 SFT,规模与停止条件;第三部分暂停决定(不足以投主会的理由、能站住的结论、暂停后处理) | A |
| `docs/survey_v4_literature.md` | 三份文献调研原始报告(SVD 类微调方法;RL vs SFT 更新几何与遗忘;遗忘控制方法与改动压缩) | A |
| `docs/TASKS_A_v4.md` / `docs/TASKS_B_v4.md` | v4 任务清单 | A 写 |
| `results/v4/result_A/`、`results/v4/result_B/` | v4 交付落地处 | A / B |

## v3(2026-09-11 → 09-17,实验完成,冻结)

| 文档 | 内容 | 维护人 |
|---|---|---|
| `docs/unified_paper_document_v3.md` | v3 企划书草案(2026-09-11 第三稿):开头为 v2 状况五行总结;主线 = 奇异基不是特权坐标系(C1–C3 现有数据可靠);发现 dense 在 lr×1 为崩溃态(§2.5:reward 达峰回落、有效 group 减少、seed 发散),故 C4/C5 须先找健康 lr\* 再裁决;实验按三个小批次(闸门 → 主结果 → 前沿)组织,每批一天;论文有底线/上限两个版本 | A |
| `docs/TASKS_A_v3.md` | 作者A 任务清单 v3:A0 代码(已完成)、A1 smoke、A2 批次一裁决(lr\*)、A3 E-v3-0c、A4 前沿图、A5 写作 | A |
| `docs/TASKS_B_v3.md` | 作者B 任务清单 v3:批次一(E-v3-0a/0b 诊断 + 12 个 dense lr 扫描 run,逐条命令)、批次二/三预告(命令待 A 裁决后填) | A 写,B 执行 |
| `results/v3/result_A/`、`results/v3/result_B/` | v3 交付落地处(与 v1/v2 同构) | A / B |

## v2(2026-09-09 → 09-11,已冻结)

| 文档 | 内容 | 维护人 |
|---|---|---|
| `docs/unified_paper_document_v2.md` | 企划书:理论 + 实验设计 + 结果登记(§9.4)+ 优先级(§14)。唯一叙事来源 | A |
| `docs/TASKS_A_v2.md` | 作者A 任务清单:A0 代码、A1 重算、A2 smoke、A3 B 交付处理、A4 写作、A5 E0 校验、A6 阶段评估 | A |
| `docs/TASKS_B_v2.md` | 作者B 任务清单:P0 先决问题、P1 补评测、P2 重算指标、P3 等范数 H5(14 run)、P4 exact_iso、P5 4B、P6 可选 | A 写,B 执行 |
| `paper/results_draft.md` | 数值登记处;"v2 登记处"小节收 A1/A5 数字与 B 待填项 | A |
| `results/v2/result_B/B_P0_answers_v2.md` | **B 对 P0 三问的答复(2026-09-10)**。要点:(1) v1 评测未含 stop-ids 修复,但 6 个 ckpt 重评差 ≤1.6 点,SFT 低于 base 是真实效应;(2) B 机器 transformers 5.9.0 忽略 `dtype=fp32`,v1 批次 master 权重实为 bf16 → B 侧 v1 全部 H 侧指标与 spectrum_only/frame_only 干预写回都受 bf16 舍入污染,G 侧不受影响;已在 `specgeom_v2/modeling.py` 加显式 `.to(dtype)` 与 dtype 断言,修复后 H 自检 0.215→0.935;(3) 各类 run 的 秒/步;另修 `intervene_engine.py` 两处(首步 KeyError、SVD 基改每步刷新)。A 侧已核实自身环境(transformers 5.16.1)加载即为 fp32,A 侧 Phase 1 / smoke / KL 探针不受影响 | B |
| `results/v2/result_A/` | A 的交付:`phase1_*`(v2 指标重算)、`smoke_v2_sft_*`(四模式 smoke)、`kl_probe_0.8b_{bf16,fp32}`(A5) | A |
| `results/v2/result_B/README.md` | **B 的 v2 交付说明(2026-09-11)**:P0–P5 全部完成(P6 未跑),48 个 run 目录;P3+P4 等范数主表(fp32):mn 维更新(full/frame_matched/exact_iso)两范式 MMLU 降到随机、r 维更新(spectrum_matched/random_ext)保住 MMLU 且 GSM8K 更高,spectrum ≈ random。数字已登记到 `paper/results_draft.md`;解读见 v3 草案 §2 | B |
| `results/v2/result_B/` | B 的交付落地处(P1–P6 的 `metrics.csv / manifest.json / eval*.json / log.jsonl / args.json`) | B |
| `figs/v2/` | Fig.B α 收敛、Fig.C 信号/噪声、Fig.D v1 耦合 vs split-half、Fig.E KL 探针;Fig.A 等 P3 | A |

## v1(2026-09-06 → 09-09,已冻结;代码 = git tag `v1`)

| 文档 | 内容 |
|---|---|
| `docs/unified_paper_document_v1.md` | v1 企划书(credit-assignment 决定谱几何 + SSD,H1–H7) |
| `docs/EXPERIMENTS_v1.md` | v1 实验总账(run matrix) |
| `docs/TASKS_B_v1.md` | v1 作者B 任务清单 P0–P6,已全部交付 |
| `docs/SCALING_v1.md` | 更大 Qwen3.5 checkpoint 的运行指南 |
| `docs/archive_idea_evolution.md` | 想法演进记录(从最初问题到方案) |
| `results/v1/result_A/`、`results/v1/result_B/README.md` | v1 结果与 B 的交付说明 |
| `figs/v1/` | v1 图 |

## 交付文档的放置规则
- B 发来的说明性文档(答复、README、交付备注)放 `results/v<版本>/result_B/`,文件名带 `_v<版本>`,并在本索引登记一行摘要。
- 数字一律进 `paper/results_draft.md`,文档只写结论与指向。
