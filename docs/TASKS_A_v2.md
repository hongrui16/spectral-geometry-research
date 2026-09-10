# 作者A 任务清单 v2

对应企划书 `docs/unified_paper_document_v2.md`,结果交付到 `results/v2/result_A/`。
原则(2026-09-09 定):**Hopper 资源紧张,A 只做代码、CPU 计算与极小的 GPU smoke;所有批量 GPU 实验归作者B(`docs/TASKS_B_v2.md`)。**

## 版本管理

- tag `v1` = 产生 `results/v1` 的代码;`docs/unified_paper_document_v1.md`、`docs/TASKS_B_v1.md`、`docs/EXPERIMENTS_v1.md` 为其文档。
- 只有 main 一个分支。代码按文件夹加版本后缀:v1 = `specgeom/ analysis/ scripts/ slurm/`(与 tag `v1` 逐字节一致,不再改动);v2 = `specgeom_v2/ analysis_v2/ scripts_v2/ slurm_v2/`(从 v1 复制后叠加 v2 改动,自包含,互不引用)。
- v2 相对 v1 的实质改动:`specgeom_v2/interventions.py`、`specgeom_v2/intervene_engine.py`(四个等范数模式)、`scripts_v2/train.py`(新 `--intervention` 值、`--intervention-seed`、scale 日志)、`analysis_v2/compute_metrics.py`(交叉样本指标、H 自检、manifest)、`scripts_v2/summary.py`、`analysis_v2/unit_test_cpu_v2.py`、`slurm_v2/*`。
- 每次提交前跑 `analysis_v2/unit_test_cpu.py` 与 `analysis_v2/unit_test_cpu_v2.py` 都必须 PASS。

## A0:代码(已完成 2026-09-09,已在 main)

| 项 | 文件 | 状态 |
|---|---|---|
| 等范数干预 spectrum_matched / frame_matched / random_ext / exact_iso | `specgeom_v2/interventions.py`、`specgeom_v2/intervene_engine.py` | ✅ CPU 测试通过 |
| 新 intervention 值、`--intervention-seed`、`scale_mean/scale_max` 日志 | `scripts_v2/train.py` | ✅ |
| 交叉样本 SNR、`R_sigma_cross`、split-half Spearman、H 自检、manifest | `analysis_v2/compute_metrics.py` | ✅ CPU 测试通过 |
| 交付摘要含新列 | `scripts_v2/summary.py` | ✅ |
| slurm:v2 指标重算(CPU)、v1 大文件清理(依赖前者成功)、v2 模式 GPU smoke | `slurm_v2/recompute.sbatch`、`slurm_v2/purge_v1_bulk.sbatch`、`slurm_v2/smoke.sbatch` | ✅ 重算与清理已完成(09-10);smoke 在 gpuq 排队 |
| v2 出图脚本(Fig.A 等范数 H5、Fig.B α 收敛、Fig.C 信号/噪声能量、Fig.D v1 耦合 vs split-half) | `analysis_v2/plot_v2.py` → `figs/v2/` | ✅ B/C/D 已出图;A 等 P3 数据 |

## A1:用 v2 指标重算 scratch 上的 v1 捕获(CPU,`normal` 分区)—— ✅ 完成 2026-09-10

结果:`results/v2/result_A/phase1_{sft,opd,rlvr}_{adamw,muon}/`,数字登记在 `paper/results_draft.md` v2 登记处 A1,判定写入 v2 §9.2 与 §10.1。
h3power 与 Llama RLVR 的 capture 早已删除(只剩 rho_hist),无法重算。0.8B 六格的 capture/ckpt 已按 `purge_v1_bulk.sbatch` 删除(释放 360G),两个 RLVR run 的 rho 张量留在 scratch 的 `rho_hist_v2.pt`。

0.8B 六格(seed 0,各 50 保存步 × 27 矩阵,含 8 微批次 G_b)、h3power、Llama-1B/3B RLVR 的捕获仍在 `/scratch/rhong5/spectral_runs/`。
这批数据可以直接给出 §9.2 H4 行的替换值(交叉样本 SNR)和 H-A 的第一手证据(信号能量是否可分辨),**不需要 B 重新捕获**。
```bash
sbatch --array=0-8 slurm_v2/recompute.sbatch          # -> results/v2/result_A/<run>/
sbatch --dependency=afterok:<jobid> slurm_v2/purge_v1_bulk.sbatch   # 成功后删 capture/ 与 ckpt_*
```
完成标准:九个 run 的 `manifest.json` 存在;登记每个 run 的 `R_sigma_cross`、`sig_energy_total` 与 `noise_energy` 中位数。
初步观察(login 节点单矩阵抽查,SFT step 10 的 L7 q_proj):`sig_energy_total ≈ -3e-4`,`noise_energy ≈ 0.47`,即 4 vs 4 微批次下**平均梯度的信号能量不可分辨**。若全表如此,H-A 成立且 v1 的 R_enrich 全部是噪声主导,这要写进 §9.2。

## A2:v2 模式 GPU smoke —— ✅ 通过 2026-09-10(MIG 3g.40gb 与 contrib-gpuq A100.80gb 各跑一遍,结果一致;`results/v2/result_A/smoke_v2_sft_*`)

| 模式 | 20 步 SFT loss(step 1 → 20) | 秒/步 | scale_mean / scale_max | 捕获 H 的 R_Σ(H) 中位数 |
|---|---|---|---|---|
| spectrum_matched | 1.121 → 0.653 | 6.7 | 50 / 91 | 0.967(自检 OK,基线的 3.4e3 倍) |
| frame_matched | 1.121 → 0.537 | 6.8 | 1.0 / 1.0 | 1e-5(≈0,应为 0) |
| random_ext | 1.121 → 0.649 | 7.3 | 52 / 82 | 1.07× 基线(随机字典与 W 基无关,应为基线) |
| exact_iso | 1.121 → 0.540 | 19.4(每步 SVD) | — | 24× 基线(重置奇异值的谱修正项) |

四个模式无 NaN,scale 在预期范围,三个投影模式的 H 自检与理论值一致。**这同时证明 A 侧代码路径下捕获的 H 就是投影后的更新**,§12.4 待查项缩小为 B 侧问题。
早期迹象(仅 20 步,不作结论):等范数下 spectrum_matched 与 random_ext 的 loss 下降幅度相同(0.65),都明显慢于 frame_matched 与 full(0.54)。P3 会回答这是否持续。

```bash
sbatch --array=0-3 slurm_v2/smoke.sbatch
```
通过标准:四个模式都跑完 20 步无 NaN;`scale_mean` 在预期范围(spectrum/random 约 30–80,frame 约 1);`analysis_v2/compute_metrics.py` 的 H 自检对 spectrum_matched 报 OK。**通过后通知 B 开 P3。**

## A3:B 交付的处理

1. 收到 P0 回答后:若 eval 脚本缺 adf1786,B 需用新脚本重评所有 SFT checkpoint(只评测,不训练);把结论写进 v2 §9.4 caveat 3。
2. 收到 P2 的 v2 metrics 后:用 H 自检结果关闭或升级 §9.4 caveat 5 与 §12.4 的待查项。
3. 收到 P3 后:主表 = {full, spectrum_matched, frame_matched, random_ext} × {RLVR, SFT} × 2 seeds 的 GSM8K/MMLU,登记到 `paper/results_draft.md`,并把 v2 §9.4 的 H5 表述改写为等范数版本。判定规则(预先锁定):
   - spectrum_matched 与 full 之差 > 2×SE(≈0.044)且 random_ext 不如 spectrum_matched → "谱方向可学但不如 frame"或"谱方向与随机方向同等",按数据写;
   - spectrum_matched ≈ random_ext ≈ 明显低于 full/frame → "在等步长下开放谱方向没有可测的额外价值",这是 §0.1 问题的直接答案;
   - 任何情形都同时报告 MMLU 变化。

## A4:分析与写作

- ✅ `analysis_v2/plot_v2.py`:Fig.B α 收敛、Fig.C 信号/噪声能量、Fig.D v1 耦合 vs split-half 已出图(`figs/v2/`);Fig.A 等范数 H5 主图在 P3 数据到达后自动生成(脚本已写好,读 `results/v2/result_B/e4a_*`)。
- ✅ `paper/results_draft.md` 已加"v2 登记处"(A1 数字 + B 待填项)。
- ✅ v2 文档 §9.2 H4 行改为交叉指标并新增信号/噪声行;§10.1 当前状态更新(H-A 成立);§14 第 3 项打勾。⏳ §9.4 等 P3 结果改写。

## A5:E0 数值校验中与现有数据直接相关的部分(CPU + 一次 GPU 前向)

- ✅ 矩形投影与维数基线(§4.5);✅ split-half 估计的无偏性(unit_test_cpu_v2)。
- 有限步 KL 标度(`analysis_v2/kl_probe.py`,`slurm_v2/kl_probe.sbatch`):27 个跟踪矩阵 × 4 个单位范数方向(随机谱、纯缩放、frame 旋转、高斯)× 步长档位,32 条 GSM8K test 参考序列、3991 个答案 token。
  - ❌ 第一次(bf16 autocast 前向,`results/v2/result_A/kl_probe_0.8b_bf16`)无效:相对扰动 ≤1e-2 低于 bf16 分辨率,纯缩放方向 KL 恰为 0,其余方向 KL 停在 2.4e-4 nats/token 的舍入底噪,对步长的对数斜率 0.1 而非 2。**结论:任何基于 KL 的 E2 测量必须用 fp32 前向;训练循环的 bf16 前向不能用于功能代价估计。** 这条写进 v2 §12.4。
  - ⏳ 第二次(fp32 前向、TF32 关闭、步长 1e-4 到 3e-2,`results/v2/result_A/kl_probe_0.8b_fp32`)作业 9787846 运行中。

## A6:阶段评估(2026-09-10,基于 A1/A2/A5 与 B 的 v1 交付)

**v2 的核心问题与 C1 测量框架成立且被加强;C2 需降级;C3 本次投稿去掉。论文最可能走 §14 的阴性结果路径。**

- 成立:C1(分半重算证明 v1 全部 G 侧指标为噪声,B=8 下信号 <1% 噪声,H4 统计量为耦合产物);§0.1 的问题有三条独立证据指向同一答案(v1 H5 两范式 frame_only≈full;α 门三范式收敛到 0.05;20 步等范数 smoke 中 spectrum_matched 与 random_ext 的 loss 相同且慢于 frame);§8.8 对 SSD 的降级被 E1/E2/E3 证实。
- 站不住:C2 依赖梯度均值 c,而 c 在 B=8 下不可分辨,需 B≥80;若 P3 证实谱方向≈随机方向,C2 是在为零增益建模 → 建议降为附录工具。C3 无实现无数据,16 天内不可能 → 删除。§10.1 的核心假设应改为 H-A + "等范数下谱方向增益不高于同维随机方向"(裁决规则见 A3.3)。
- 风险:(1) B 的 SFT 行全部低于 base,先要 P0.1 的回答;(2) fp32 KL 探针若显示谱/frame/高斯方向功能代价也相同,则"谱 vs frame"在 bulk 上既无学习价值也无功能差异,论文立足点退到"权重 SVD 基对后训练不是特殊坐标系",需改标题与摘要。
- 待你决定后执行的 v2 改动:§0.2 贡献表(C1 保留、C2 附录、C3 删除)、§1.1 摘要按阴性路径重写、§10.1 核心假设替换、§14 只保留 P3 与 SFT 评测确认为必做。

## 时间线
- 09-10:A1 提交(已提交)、A2 提交并通知 B
- 09-11 到 09-14:A3.1–A3.2、A4 出图脚本
- 09-15 到 09-20:A3.3、写作
- 09-21 到 09-26:定稿
