# 实验总账(Run Matrix)

更新:2026-09-06。状态标记:✅ 完成 / 🟢 运行中 / ⬜ 排队或未开始 / 🔒 等依赖。
所有 run 输出在 `/scratch/rhong5/spectral_runs/<run_name>`,捕获数据同目录 `capture/`。

## 模型分工

| 模型 | 范围 | 谁跑 | 硬件 |
|---|---|---|---|
| Qwen3.5-0.8B | 全量主线(下表全部) | 本集群 | MIG 3g.40gb |
| Qwen3.5-4B | Phase 1 (AdamW×3) + E1 RLVR 行 | 协作者 A | 单卡 80GB,开箱即用(见 SCALING.md) |
| Qwen3.5-9B | Phase 1 (AdamW×3) 复现 Fig.2/4 | 协作者 B | 单卡 80GB + 影子捕获改造(待做)或 2×80GB |
| Llama-3.2-3B-Instruct | Phase 1 (AdamW×3) 家族轴,复现 Fig.2/4 | 本集群 | MIG;等 gated 审批(已申请) |
| Qwen3.5-2B | 仅作 0.8B 的 OPD teacher(bf16 推理) | — | — |
| 27B / 35B-A3B(MoE) | 不跑,future work | — | — |

OPD teacher 配对:0.8B←2B、4B←9B、9B←27B(teacher 全部 bf16 推理)。

## 0.8B 主线(本集群)

### Phase 1 — 三范式测量,H1-H4(任务 #3)
500 步,save-every 10,P=8,K=8,seed 0。

| Run | 状态 | 备注 |
|---|---|---|
| phase1_sft_adamw | 🟢 step 200+ (~7.8s/步) | |
| phase1_opd_adamw | 🟢 刚启动 | teacher=2B |
| phase1_rlvr_adamw | 🟢 (~32s/步,reward 8-23%) | 关键路径 |
| phase1_rlvr_muon | ⬜ 排队 | H2 |
| phase1_sft_muon | ⬜ 未提交(array 4) | H2 |
| phase1_opd_muon | ⬜ 未提交(array 5) | H2 |

产出:Fig.2 (R_spectrum 三范式)、Fig.3 (G vs H)、Fig.4 (ρ^Σ/ρ^frame)、Fig.5 (|C| vs SNR)。
基线:0.8B GSM8K greedy pass@1 = **57.5%** (200 题) ✅

### 决策点
H1 成立(R_spectrum: SFT > OPD > RLVR)→ Phase 2;不成立 → 主线转 H2+§5(见文档 §附)。

### Phase 2 — 干预实验,H5-H6(任务 #4)🔒 依赖 Phase 1
300 步 + GSM8K-500 评测,AdamW 底座,`slurm/phase2.sbatch` array 0-7:
{rlvr,sft} × {spectrum_only, frame_only} (H5);{rlvr,sft} × {snr_topq, mag_topq} q=0.1 (H6)。
产出:Fig.6、Fig.7。

### E1 — optimizer × 范式主表(任务 #5)
`slurm/e1_optimizers.sbatch`。AdamW/Muon 六格复用 Phase 1(同配置同 seed);新跑:
| Run | 状态 |
|---|---|
| e1_{sft,opd,rlvr}_ssd(SSD-Wiener) | 🔒 SSD 先过小规模试跑 |
| e1_rlvr_ssd-muon(变体) | 🔒 |
| 每 run 附 GSM8K-500 评测 | |
补 seed:RLVR 三 optimizer × seed {1,2} 优先;SFT/OPD × seed 1 次之。
产出:Tab.1、Fig.8 (RLVR 训练曲线:Muon 退化 vs SSD)。

### E2 — SSD 消融 🔒 依赖 E1
rlvr 300 步 ×4:f≡1(退化 Muon)、硬阈值 vs Wiener、k∈{64,256}、去 mode 对齐。

### E3 — 预测关(Part I → M1)🔒 依赖 Phase 1 + E1
按 4 组层(embed 侧/中/深/输出侧)单独启用 SSD,rlvr 300 步 ×4;
横轴用 Phase 1 测得的 1−Spearman(|C|,SNR),纵轴 SSD 相对 Muon 增益。产出:Fig.9。

### E4 — M2 自适应 α(任务 #5 内)
{sft,opd,rlvr} × adaptive_alpha(`--intervention adaptive_alpha`),300-500 步 + 评测;
对照 full(α=1)与 ISO 型(spectrum 冻结)。产出:Fig.10 (α_t 轨迹 + 性能/遗忘)。

### 砍掉/降级(20 天范围)
- M3/E6(contribution regularizer)、E7(continued pretraining sanity):时间富余才做
- 遗忘评测:MMLU 子集一项,IFEval/HumanEval 砍
- 3 seeds → 主表 2-3 seeds,消融 1 seed

## 时间预算(实测步速)
单 run:SFT ~1h,OPD ~3-4h,RLVR ~4.5h(MIG 3g.40gb,500 步)。
0.8B 全部 ≈ 105-120 GPU·h;3 切片并行 ≈ 2.5-3 天。
4B 全套(协作者)≈ 90 GPU·h;9B Phase1 ≈ 60 GPU·h;Llama-3B Phase1 ≈ 30 GPU·h。

## 待办(工程)
- [ ] SSD 小规模 GPU 试跑(50 步 rlvr)后解锁 E1/E2/E3
- [ ] 9B 影子 fp32 捕获 + 8-bit AdamW 改造(半天,推仓库后协作者 B 接手)
- [ ] Llama-3.2-3B 冒烟(等 gated 审批)
- [ ] MMLU 子集遗忘评测脚本
- [ ] E3 分层 SSD 路由 flag
