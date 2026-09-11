# 文档索引

仓库里所有文档、交付说明与登记处的入口。版本规则见 `README.md`:v1 已冻结,
v2 为当前版本;旧版本文件只改引用不改内容。

## v2(当前,2026-09-09 起)

| 文档 | 内容 | 维护人 |
|---|---|---|
| `docs/unified_paper_document_v2.md` | 企划书:理论 + 实验设计 + 结果登记(§9.4)+ 优先级(§14)。唯一叙事来源 | A |
| `docs/TASKS_A_v2.md` | 作者A 任务清单:A0 代码、A1 重算、A2 smoke、A3 B 交付处理、A4 写作、A5 E0 校验、A6 阶段评估 | A |
| `docs/TASKS_B_v2.md` | 作者B 任务清单:P0 先决问题、P1 补评测、P2 重算指标、P3 等范数 H5(14 run)、P4 exact_iso、P5 4B、P6 可选 | A 写,B 执行 |
| `paper/results_draft.md` | 数值登记处;"v2 登记处"小节收 A1/A5 数字与 B 待填项 | A |
| `results/v2/result_B/B_P0_answers_v2.md` | **B 对 P0 三问的答复(2026-09-10)**。要点:(1) v1 评测未含 stop-ids 修复,但 6 个 ckpt 重评差 ≤1.6 点,SFT 低于 base 是真实效应;(2) B 机器 transformers 5.9.0 忽略 `dtype=fp32`,v1 批次 master 权重实为 bf16 → B 侧 v1 全部 H 侧指标与 spectrum_only/frame_only 干预写回都受 bf16 舍入污染,G 侧不受影响;已在 `specgeom_v2/modeling.py` 加显式 `.to(dtype)` 与 dtype 断言,修复后 H 自检 0.215→0.935;(3) 各类 run 的 秒/步;另修 `intervene_engine.py` 两处(首步 KeyError、SVD 基改每步刷新)。A 侧已核实自身环境(transformers 5.16.1)加载即为 fp32,A 侧 Phase 1 / smoke / KL 探针不受影响 | B |
| `results/v2/result_A/` | A 的交付:`phase1_*`(v2 指标重算)、`smoke_v2_sft_*`(四模式 smoke)、`kl_probe_0.8b_{bf16,fp32}`(A5) | A |
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
