# 实验总账(Run Matrix)

**硬 deadline:全部实验 14 天内跑完(至 2026-09-20)。**
- D0-2 Phase 1 → D1-4 SSD+Phase 2 → D3-9 E1/E2/E3/E4+seed → D9-14 家族轴+缓冲+图表冻结
- 家族轴:Llama-3.2-3B 审批若 D1 内不下来,立即切 SmolLM3-3B(Apache,免审批)
- 9B:仅当协作者 4B 在 D8 前收尾,否则砍
- seed 策略:E1 RLVR 行 3 seeds,主表其余 2,消融 1

更新:2026-09-06。状态标记:✅ 完成 / 🟢 运行中 / ⬜ 排队或未开始 / 🔒 等依赖。
所有 run 输出在 `/scratch/rhong5/spectral_runs/<run_name>`,捕获数据同目录 `capture/`。

## 人员分工(按当前 GPU 情况)

**你(本集群,MIG 切片;作业由 Claude 编排提交)**
- 0.8B 全部:Phase 1 六格、Phase 2 ×8、E1 SSD 行、E2/E3/E4、补 seed、全部评测
- Llama-3.2-3B 家族轴 Phase 1(等 gated 审批;MIG 跑得动,~30 GPU·h)
- 全部分析出图(analysis/ 管线,含协作者传回的数据)
- 只有你本人能做的两件事:① GPU 取舍——fit*/ss* 手部建模作业与论文作业共享 16 卡配额,
  要加速就暂停几个;② gated 模型审批(Llama 已申请)

**协作者(自己的 GPU,假设 ≥1×A100/H100 80GB;不够 80GB 提前说)**
- 【立刻开工】4B:Phase 1 AdamW×3(sft/opd/rlvr,teacher=9B)+ E1 RLVR 行
  ({adamw, muon, ssd})+ 4B base 的 GSM8K greedy 基线一次
- 【4B 收尾后】9B:Phase 1 AdamW×3(等仓库里的影子捕获改造,Claude 负责,预计 1-2 天内推上)
- 交付物:每个 run 的 `capture/` 目录 + `log.jsonl` + `args.json` + 评测 json,
  打包传回集群(或给可访问路径);**不要自己改分析代码**,口径统一走本仓库 analysis/

**Claude(工程,不占 GPU 决策)**
- SSD 50 步试跑验证 → 解锁 E1/E2/E3;9B 影子 fp32 捕获改造;Llama 冒烟;
  E3 分层路由 flag;MMLU 遗忘评测脚本;EXPERIMENTS.md 状态维护 + push

## 模型分工

| 模型 | 范围 | 谁跑 | 硬件 |
|---|---|---|---|
| Qwen3.5-0.8B | 全量主线(下表全部) | 本集群 | MIG 3g.40gb |
| Qwen3.5-4B | 【优先】Phase 1 (AdamW×3) + E1 RLVR 行 | 协作者(唯一) | 单卡 80GB,开箱即用(见 SCALING.md) |
| Qwen3.5-9B | 【余力目标】Phase 1 (AdamW×3) 复现 Fig.2/4 | 同一协作者,4B 收尾后 | 需影子捕获改造(待做);若放弃,尺度轴 0.8→4 + Llama 家族轴仍然成立 |
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
