# 作者A 任务清单 v4(2026-09-17 生效)

企划书 `docs/unified_paper_document_v4.md`(第一部分 v3 总结;第二部分 §2 改进方案:遗忘是"走了多远"不是"往哪走"、§3 规模、§4 停止条件)。代码 `*_v4/`(从 v3 复制,内部引用已改;v3 冻结)。结果 `results/v4/result_A`。数字登记仍在 `paper/results_draft.md`(新开"v4 登记处")。文档一律白话。

## A0 代码 ✅ 2026-09-17
- `specgeom_v4/ analysis_v4/ scripts_v4/ slurm_v4/` 建好,含 exact_iso 修复版(gesvda + 等范数)。CPU 单元测试通过。
- `analysis_v4/kl_probe.py` 新增 `build_offtask_reference`(MMLU validation 题干 + 选项,与评测用的 test 分开);`cum_kl.py` / `step_kl.py` 加 `--reference {task,offtask}`。
- `slurm_v4/offtask_kl.sbatch`(§2.1-1)、`slurm_v4/batch1_lr1_arms.sbatch`(§2.1-2)已提交。

## A1 §2.1-1 任务外 KL(零训练,已提交 offtask_kl.sbatch)
- 对全部 v3 终点 ckpt(含 exact_iso 修复版)算任务外累计 KL;对 10 组 8 步捕获算任务外单步 KL(步 8、4)。
- 交付:`results/v4/result_A/cum_kl_offtask/`、`step_kl_offtask_{sft,rlvr}_step{8,4}/`。
- 裁决口径(企划书 §2.1):五种方向臂在等范数下,任务外累计 KL 差不超过 20%,MMLU 差不超过 0.02;学习率一档之差的遗忘差远大于此。

## A2 §2.1-2 lr×1 上的保谱与去谱(已提交 batch1_lr1_arms,8 run,MIG)
- SFT 1e-5、RLVR 2e-6,frame_matched / exact_iso(修复版)× 2 seed。与已有 lr×1 上的 dense(v2 e4a / v3 dense lr0.33 不是同 lr,故 lr×1 dense 需补:见 A2b)、spectrum、random 组成 lr×1 全套。
- **A2b**:lr×1 dense 参照。v2 e4a 的 dense 是 bf16 时代?——不是,v2 e4a 为 fp32 但为崩溃态;仍用作 lr×1 dense 参照(reward 回落已知),另在 v4 补 SFT dense lr×1 × 2 seed(RLVR lr×1 dense 用 e4a_rlvr_full_s0/s1 + v3 lr1 s2 数据)。

## A3 §2.1-3(修订,2026-09-17,据 E-v4-1):RL 与 SFT 的差别在多步叠加,不在单步
- E-v4-1:单步每单位范数的任务 KL 与任务外 KL 两范式相同,300 步后任务外累计 KL 差 13 倍。原"每步功能步长匹配"实验取消。
- 改做两件事:(a) **任务外 KL 轨迹**:dense SFT lr\* 与 RLVR lr\* 各 2 seed 重跑并保留每 50 步 ckpt(`--keep-eval-ckpts`),算每个 ckpt 的任务外/任务累计 KL,看 SFT 的任务外累积从哪一步开始与 RLVR 分开(`slurm_v4/traj_runs.sbatch`);(b) **逐矩阵归因**:对终点 ckpt 逐个矩阵回退到 base,测任务外 KL 与任务 KL 的变化,得到每个矩阵的"任务外承重 / 任务承重"比(`analysis_v4/matrix_attrib.py`),比较 SFT 与 RLVR 的分布,找 SFT 里任务外承重集中的矩阵。
- 裁决:若 SFT 的任务外累积集中在少数矩阵且这些矩阵在 RLVR 下不动,则得到一个可操作的方法假设(§2.3 改为"按矩阵的任务外承重压步长");若均匀分布,§2.3 退回全局功能步长控制。
- 遗忘读数改为:任务外 KL + 全量 MMLU(E-v4-2,`slurm_v4/mmlu_full.sbatch`,全部终点 ckpt)。

## A4 §2.2 遗忘探针
- 对每个已有配置(约 60 个 run 的起点配置)跑 8 步捕获 + 任务外单步 KL,算 KL/‖H‖²;与最终 MMLU 降幅作图。写 `analysis_v4/forgetting_probe.py`。
- 判据:R² ≥ 0.7 且跨范式、跨干预同一条曲线;< 0.5 探针删除。
- 新设定前瞻:0.8B + MATH(需先下载 MATH 到 HF 缓存,离线)两范式各扫三档,先预测后训练(12 run,A);2B + GSM8K RLVR 三档由 B 跑。

## A5 §2.3 方法:按功能步长做 SFT(第 2 周)
- `scripts_v4/train.py` 加 `--offtask-kl-budget`:训练前与每 50 步用探针测任务外单步 KL,按预算调学习率(全局版);按矩阵版作消融。
- 对照:调好 lr 的 SFT(已有)、通用数据 KL 惩罚(`--kl-beta` 改为任务外参考集)、FINCH(2605.20005,按 loss 调 lr,需实现)、Anchored Learning(2605.04468,需实现或最接近的等价)。
- 判据:同任务分数下 MMLU 不低于最强对照且开销不高于它。打不过如实写。

## A6 写作
- 第一部分的短文(negative + protocol)现在就写;§2.1/2.2 结果出来后按企划书 §2 的主张改写为正文。

## 时间线
- 第 1 周:A1、A2、A3、A4(现有数据部分);MATH 下载。
- 第 2 周:A5;A4 新设定。
- 第 3 周:裁决与写作。停止条件见企划书 §4。
