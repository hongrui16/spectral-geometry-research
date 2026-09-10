# 作者A 任务清单 v2

对应企划书 `docs/unified_paper_document_v2.md`,结果交付到 `results/v2/result_A/`。
原则(2026-09-09 定):**Hopper 资源紧张,A 只做代码、CPU 计算与极小的 GPU smoke;所有批量 GPU 实验归作者B(`docs/TASKS_B_v2.md`)。**

## 版本管理

- tag `v1` = 产生 `results/v1` 的代码;`docs/unified_paper_document_v1.md`、`docs/TASKS_B_v1.md`、`docs/EXPERIMENTS_v1.md` 为其文档。
- main = 本轮开发(只有 main 一个分支)。v2 功能一律放新文件,不改 v1 文件:
  `specgeom/interventions_v2.py`、`specgeom/intervene_engine_v2.py`、`analysis/compute_metrics_v2.py`、`scripts/summary_v2.py`、`analysis/unit_test_cpu_v2.py`。
  `scripts/train.py` 只加了向后兼容的分发:v1 的 `--intervention` 值仍走 v1 引擎,四个新值走 v2 引擎。
- 每次提交前跑 `analysis/unit_test_cpu.py`(v1)与 `analysis/unit_test_cpu_v2.py`(v2)都必须 PASS。

## A0:代码(已完成 2026-09-09,已在 main)

| 项 | 文件 | 状态 |
|---|---|---|
| 等范数干预 spectrum_matched / frame_matched / random_ext / exact_iso | `specgeom/interventions_v2.py`、`specgeom/intervene_engine_v2.py` | ✅ CPU 测试通过 |
| train.py 分发、`--intervention-seed`、`scale_mean/scale_max` 日志 | `scripts/train.py` | ✅ |
| 交叉样本 SNR、`R_sigma_cross`、split-half Spearman、H 自检、manifest | `analysis/compute_metrics_v2.py` | ✅ CPU 测试通过 |
| 交付摘要含新列 | `scripts/summary_v2.py` | ✅ |
| slurm:v2 指标重算(CPU)、v1 大文件清理(依赖前者成功)、v2 模式 GPU smoke | `slurm/recompute_v2.sbatch`、`slurm/purge_v1_bulk.sbatch`、`slurm/smoke_v2.sbatch` | ✅ 已提交 / 待提交 |

## A1:用 v2 指标重算 scratch 上的 v1 捕获(CPU,`normal` 分区)

0.8B 六格(seed 0,各 50 保存步 × 27 矩阵,含 8 微批次 G_b)、h3power、Llama-1B/3B RLVR 的捕获仍在 `/scratch/rhong5/spectral_runs/`。
这批数据可以直接给出 §9.2 H4 行的替换值(交叉样本 SNR)和 H-A 的第一手证据(信号能量是否可分辨),**不需要 B 重新捕获**。
```bash
sbatch --array=0-8 slurm/recompute_v2.sbatch          # -> results/v2/result_A/<run>/
sbatch --dependency=afterok:<jobid> slurm/purge_v1_bulk.sbatch   # 成功后删 capture/ 与 ckpt_*
```
完成标准:九个 run 的 `manifest.json` 存在;登记每个 run 的 `R_sigma_cross`、`sig_energy_total` 与 `noise_energy` 中位数。
初步观察(login 节点单矩阵抽查,SFT step 10 的 L7 q_proj):`sig_energy_total ≈ -3e-4`,`noise_energy ≈ 0.47`,即 4 vs 4 微批次下**平均梯度的信号能量不可分辨**。若全表如此,H-A 成立且 v1 的 R_enrich 全部是噪声主导,这要写进 §9.2。

## A2:v2 模式 GPU smoke(唯一的 GPU 项,4 × 20 步 SFT,单 A100.40gb,约 15 分钟/个)

```bash
sbatch --array=0-3 slurm/smoke_v2.sbatch
```
通过标准:四个模式都跑完 20 步无 NaN;`scale_mean` 在预期范围(spectrum/random 约 30–80,frame 约 1);`compute_metrics_v2` 的 H 自检对 spectrum_matched 报 OK。**通过后通知 B 开 P3。**

## A3:B 交付的处理

1. 收到 P0 回答后:若 eval 脚本缺 adf1786,B 需用新脚本重评所有 SFT checkpoint(只评测,不训练);把结论写进 v2 §9.4 caveat 3。
2. 收到 P2 的 v2 metrics 后:用 H 自检结果关闭或升级 §9.4 caveat 5 与 §12.4 的待查项。
3. 收到 P3 后:主表 = {full, spectrum_matched, frame_matched, random_ext} × {RLVR, SFT} × 2 seeds 的 GSM8K/MMLU,登记到 `paper/results_draft.md`,并把 v2 §9.4 的 H5 表述改写为等范数版本。判定规则(预先锁定):
   - spectrum_matched 与 full 之差 > 2×SE(≈0.044)且 random_ext 不如 spectrum_matched → "谱方向可学但不如 frame"或"谱方向与随机方向同等",按数据写;
   - spectrum_matched ≈ random_ext ≈ 明显低于 full/frame → "在等步长下开放谱方向没有可测的额外价值",这是 §0.1 问题的直接答案;
   - 任何情形都同时报告 MMLU 变化。

## A4:分析与写作

- 新出图脚本 `analysis/plot_v2.py`(待写):Fig.A 等范数 H5 主图(两范式 × 四模式,带 CI);Fig.B 三范式 α 收敛曲线(e4_*);Fig.C 交叉样本信号/噪声能量随步数(A1 输出);Fig.D v1 H4 耦合指标 vs 交叉指标对比。
- `paper/results_draft.md` 增加"v2 登记处"小节;v1 登记处保留不改。
- v2 文档:§9.2 H4 行改为交叉指标数值;§9.4 按 P3 结果改写;§10.1 当前状态更新;§14 优先级完成情况打勾。

## A5:E0 数值校验中与现有数据直接相关的部分(CPU)

矩形投影与维数基线(已在 §4.5)、split-half 估计的无偏性(unit_test_cpu_v2 已覆盖)、有限步 KL 的二次标度(需一次 0.8B 前向,可放在 A2 的 smoke 之后用同一张卡做,若资源不允许则推迟)。

## 时间线
- 09-10:A1 提交(已提交)、A2 提交并通知 B
- 09-11 到 09-14:A3.1–A3.2、A4 出图脚本
- 09-15 到 09-20:A3.3、写作
- 09-21 到 09-26:定稿
