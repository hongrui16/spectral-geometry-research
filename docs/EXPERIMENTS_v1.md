# 实验总账(Run Matrix)

**硬 deadline:全部实验 14 天内跑完(至 2026-09-20)。**
- 9B:仅当作者B 4B 在 D8 前收尾,否则砍
- seed 策略:E1 RLVR 行 3 seeds,主表其余 2,消融 1
- 家族轴不再扩尺寸(8B+ 会加剧 GSM8K 饱和,无科学增益;尺度泛化由 Qwen 4B/9B 承担)

更新:2026-09-08(D2)。状态标记:✅ 完成 / 🟢 运行中 / ⬜ 排队或未开始 / 🔒 等依赖。
所有 run 输出在 `/scratch/rhong5/spectral_runs/<run_name>`,捕获数据同目录 `capture/`。

**D2 快照**:作者A 侧接近收官——0.8B 六格 ✅ + 四假设判定 ✅;H3 终判 ✅;SSD ✅;
家族轴 5/6 格 ✅(3B SFT/RLVR + 1B SFT/OPD;在途:1B RLVR ⬜ 排队、3B OPD 🟢);
**关键路径 = 作者B(Phase 2 ×10、E1 ×16、E2 ×4、E3 ×4、4B/9B),D2 仍零开工——
论文的 Fig.6-9 与主表全部依赖这批**。数值登记:`paper/results_draft.md`。

## 人员分工

**原则:只有"单卡 + 任务数量少"的任务归作者A;批量实验和多卡任务全部归作者B。**

**作者A(单卡、少量、快迭代)**
- Phase 1 0.8B 六格(已在跑/完成)+ H1-H4 判定
- SSD 50 步单卡试跑验证(验证通过后 E1/E2/E3 由作者B 批量执行)
- Llama-3.2-3B 家族轴 Phase 1 AdamW×3(单卡,3 个 run)
- 全部分析出图(analysis/ 管线,含作者B 传回的数据)、论文写作、本文档维护
- gated 模型审批等只有 A 能做的账号操作

**作者B(批量 + 多卡,全部可并行)**
- **唯一执行文档:`TASKS_B.md`(P0-P6,含环境、命令、健康自检、交付)**
- 4B:Phase 1 AdamW×3 + E1 RLVR 行({adamw, muon, ssd})+ 4B 基线评测
- 9B:Phase 1 AdamW×3(`--device-map auto`,见 docs/SCALING_v1.md)
- E1 SSD 行 ×4、E2 消融 ×4、E3 分层 ×4、E4 ×3、补 seed 批量(SSD 相关项等
  作者A 试跑验证通过后开跑;各项的具体命令届时由作者A 更新到本文档)
- 交付物:每 run 跑 `scripts/summary.py`,把打印的一行文字发给作者A(B 无法回传文件)

## 模型分工

| 模型 | 范围 | 谁跑 | 备注 |
|---|---|---|---|
| Qwen3.5-0.8B | 全量主线(下表全部) | 作者A | |
| Qwen3.5-4B | Phase 1 (AdamW×3) + E1 RLVR 行 | 作者B | 开箱即用,可并行(见 docs/SCALING_v1.md) |
| Qwen3.5-9B | Phase 1 (AdamW×3) 复现 Fig.2/4 | 作者B | `--device-map auto`,配置见 docs/SCALING_v1.md |
| Llama-3.2-3B-Instruct | 家族轴(补充材料,带饱和窗口分析) | 作者A | SFT/RLVR ✅ 判定已出;OPD 排队 |
| Llama-3.2-1B-Instruct | 家族轴主证据(无饱和,AdamW×3) | 作者A | 三格已提交 MIG(9693553),teacher=3B |
| Qwen3.5-2B | 仅作 0.8B 的 OPD teacher(bf16 推理) | — | |
| 27B / 35B-A3B(MoE) | 不跑,future work | — | |

OPD teacher 配对:0.8B←2B、4B←9B、9B←27B(teacher 全部 bf16 推理)。

## 0.8B 主线(作者A)

### Phase 1 — 三范式测量,H1-H4(任务 #3)✅ 0.8B 六格全部完成(D0)
500 步,save-every 10,P=8,K=8,seed 0。
lr:SFT/OPD 1e-5;RLVR 2e-6(AdamW)/2e-5(Muon)——1e-5 实测 40 步内策略崩塌。

| Run | 状态 |
|---|---|
| phase1_{sft,opd,rlvr}_{adamw,muon} 六格(0.8B) | ✅ 全部完成 + metrics |
| phase1_llama3b_sft / rlvr | ✅(判定:H4 更强,H1 于未饱和窗口复现) |
| phase1_llama3b_opd | 🟢 running(teacher=8B) |
| phase1_llama1b_sft / opd | ✅ 1.067/0.846、1.073/0.818(与 0.8B 稠密端结论一致) |
| phase1_llama1b_rlvr | ⬜ 排队(无饱和家族轴关键格) |
| h3power_rlvr(加强采样) | ✅ 终判:σ/frame 对称,H-RLVR 拒绝 |

H2 三对齐全:G 侧 optimizer 不变性在 SFT/OPD/RLVR 全部成立(RLVR 对:0.000290 vs
0.000287);Muon 的 H 谱富集在 RLVR 上反而消失(H/G≈1.0 vs SFT/OPD 的 1.1)。
跨架构:Llama(标准注意力)富集 >1,Qwen 偏低是 linear_attn(0.674)拖累——
线性注意力抑制谱对角是架构效应。

产出:Fig.2 (R_spectrum 三范式)、Fig.3 (G vs H)、Fig.4 (ρ^Σ/ρ^frame)、Fig.5 (|C| vs SNR)。
基线:0.8B GSM8K greedy pass@1 = **57.5%** (200 题) ✅

### 决策点 — 已判定(D0 晚)
0.8B AdamW 三格结果:
- **H1 方向成立、幅度温和**:富集度 SFT 0.886 > OPD 0.869 > RLVR 0.839(排序一致,跨度 ~5%)
- **H2 强成立**:G 侧几何对 optimizer 不变(SFT/OPD 两对均 <7% 差异)
- **H3 未获支持**:K=8 下 |ρ^Σ| ≈ |ρ^frame|(0.291 vs 0.292),池化后仍对称——符合 Prop.7,
  H-RLVR 的序列级破缺不可见。终判(D1 加强采样 8×统计力):σ/frame 完全对称,H-RLVR 拒绝,归因唯一化到 SNR 通道
- **H4 方向成立、幅度小**:Spearman(|C|,SNR) RLVR 0.848 < SFT 0.880
- **新发现**:linear_attn 梯度对角富集仅 0.65(低于随机),MLP ≈1.0,三范式一致

**主线调整**:机制叙事以 §5 SNR 理论 + C3 为核心(文档 §附 预案);Phase 2 因果干预
(作者B,TASKS_B.md P0)升级为全文关键证据;H3 结果作为"可证伪量按设计工作"呈现。

### Phase 2 — 干预实验,H5-H6(任务 #4)|主责:**作者B**
10 个 run(8 干预 + 2 对照),300 步 + GSM8K-500 评测。命令:`TASKS_B.md` P0。
产出:Fig.6、Fig.7(作者A 出图)。

### E1 — optimizer × 范式主表(任务 #5)|✅ SSD 试跑通过,作者B 可开跑
AdamW/Muon 六格复用 Phase 1(seed 0);作者B 新跑以下 16 个(0.8B,单卡/run,全可并行):

**seed 0 SSD 格(4 个)**,500 步模板(SFT/OPD 去掉 --lr/--rollouts,OPD 加 --objective opd):
```bash
python scripts/train.py --objective rlvr --optimizer ssd --lr 2e-6 --muon-lr 2e-5 \
  --steps 500 --save-every 25 --prompts-per-step 8 --rollouts 8 \
  --max-new-tokens 384 --seed 0 --out $RUNS/e1_rlvr_ssd_s0
# 同样跑:e1_sft_ssd_s0、e1_opd_ssd_s0、e1_rlvr_ssd-muon_s0(--optimizer ssd-muon)
```
**补 seed(12 个)**:rlvr×{adamw,muon,ssd}×seed{1,2}(6 个);{sft,opd}×{adamw,muon,ssd}×seed 1(6 个)。
参数与 Phase 1/上面完全一致,只改 --seed 与 --out 后缀(_s1/_s2)。
每 run 结束跑评测:`python scripts/eval_gsm8k.py --model $RUNS/<run>/ckpt_000500 --limit 500 --out $RUNS/<run>/eval_gsm8k.json`
产出:Tab.1、Fig.8。

### E2 — SSD 消融(作者B,rlvr 300 步 ×4,seed 0)
```bash
# 基准即 e1_rlvr_ssd_s0;消融各改一个 flag:
--optimizer ssd-muon                 # sign 变体(硬 vs Wiener)
--ssd-k 64                           # 子空间维度
--ssd-no-align                       # 去 mode 对齐
--ssd-tail-coef 0                    # 去尾部更新
```
--steps 300,其余同 E1 模板,--out $RUNS/e2_<变体名>。

### E3 — 预测关(Part I → M1)|作者B,flag 已实现
按 4 组层单独启用 SSD(其余层 Muon),rlvr 300 步 ×4 + 对照全 Muon(复用 E1):
```bash
# 0.8B 共 24 层,4 组:0-5 / 6-11 / 12-17 / 18-23
python scripts/train.py --objective rlvr --optimizer ssd-routed --ssd-layer-range 0-5 \
  --lr 2e-6 --muon-lr 2e-5 --steps 300 --save-every 25 --prompts-per-step 8 \
  --rollouts 8 --max-new-tokens 384 --seed 0 --out $RUNS/e3_ssd_layers0-5
# 其余三组同理:6-11 / 12-17 / 18-23,各配 eval(--limit 500)
```
横轴用 Phase 1 测得的按层 1−Spearman(|C|,SNR),纵轴该组启用 SSD 的增益。产出:Fig.9。

### E4 — M2 自适应 α(任务 #5 内)
{sft,opd,rlvr} × adaptive_alpha,300 步 + 评测;对照复用 Phase 2 的 none/frame_only。
命令:`TASKS_B.md` P4。产出:Fig.10 (α_t 轨迹 + 性能/遗忘)。

### 砍掉/降级(20 天范围)
- M3/E6(contribution regularizer)、E7(continued pretraining sanity):时间富余才做
- 遗忘评测:MMLU 子集一项,IFEval/HumanEval 砍
- 3 seeds → 主表 2-3 seeds,消融 1 seed

## 时间预算(实测步速)
单 run:SFT ~1h,OPD ~3-4h,RLVR ~4.5h(500 步,实测)。
0.8B 全部 ≈ 105-120 GPU·h;3 切片并行 ≈ 2.5-3 天。
4B 全套(作者B)≈ 90 GPU·h;9B Phase1 ≈ 60 GPU·h;Llama-3B Phase1 ≈ 30 GPU·h。

## 待办(工程)— D1 全部清零
- [x] SSD GPU 试跑(50 步 rlvr 通过,reward 正常爬升)
- [x] 9B 多卡支持(`--device-map auto`)
- [x] Llama-3.2-3B 冒烟(GPU 节点通过;登录节点 OOM-kill 是内存限额)
- [x] MMLU 遗忘评测脚本(`scripts/eval_mmlu.py`)
- [x] E3 分层 SSD 路由(`--optimizer ssd-routed --ssd-layer-range`)
- [x] H3 加强采样 run + 终判(拒绝 H-RLVR,归因 SNR 通道)
