# 实验结果草稿(随数据滚动更新;最终转 LaTeX)

## 数值登记处(单一可信来源)

### Setup
- 主模型 Qwen3.5-0.8B(混合架构:linear_attn 多数层 + full attn 于 L7/15/19;24 层)
- GSM8K,同 prompt 流(seed 0),500 步,P=8;RLVR: GRPO K=8,correctness reward
- lr:SFT/OPD 1e-5,RLVR 2e-6(AdamW)/2e-5(Muon);fp32 主权重 + bf16 autocast
- 基线:0.8B greedy pass@1 = 57.5%(200 题);Llama-3.2-3B 家族轴对照

### Phase 1(seed 0,中位数聚合 27 矩阵 × 50 保存步)

| 量 | SFT | OPD | RLVR | 说明 |
|---|---|---|---|---|
| R_enrich(G) AdamW | 0.886 | 0.869 | 0.839 | 对随机零假设 1/max(m,n) 归一 |
| Spearman(\|C\|,SNR) | 0.880 | 0.874 | 0.848 | H4 |
| G stable rank | 2.65 | 2.55 | — | 梯度高度低秩 |
| ⟨G,W⟩/(‖G‖‖W‖) | ≈0 | ≈0 | ≈0 | Prop.6 验证 |

H2(G 侧 optimizer 不变性,AdamW vs Muon 的 R_spectrum(G)):
- SFT:0.000302 vs 0.000312;OPD:0.000296 vs 0.000318;RLVR:0.000290 vs 0.000287
- Muon 的 H/G 谱富集比:SFT 1.13 / OPD 1.10 / **RLVR ≈1.0(消失)**

H3(K=8,4 矩阵 × 50 步):|ρ^Σ| med 0.291 ≈ |ρ^frame| med 0.292,零假设 0.278;
Fisher-z 池化后均降至池化零假设水平;符号无关池化:均值 0.316-0.336 vs 零假设 0.313,
超阈 mode 占比 3-12%(零假设 ~2%)。→ 有微弱持续信号,但 σ/frame **无差别**。
加强采样 run(P=4×K=16,4 捕获组,400 步,seed 3)终判:
|ρ^Σ| med 0.1314 ≈ |ρ^frame| med 0.1308,匹配零假设(K16,G2)0.1305;
尾部(>0.3)13.0% / 12.9% vs 零假设 11.4%。**σ/frame 完全对称,H-RLVR 拒绝,
Prop.7 对称性在真实模型成立;frame-dominance 归因唯一化到 SNR 通道(§5/C3)。**

家族轴(Llama-3.2-3B,标准注意力)SFT vs RLVR:
- 全程:R_enrich 1.073 vs 1.072(无差);Spearman 0.863 vs **0.785**(差距 0.078,
  是 Qwen 的 2.5 倍)
- **饱和混淆**:Llama reward 100 步内 58%→98%,后 400 步 RLVR 信号近零。
  限定未饱和窗口(step≤100):R_enrich 1.084 vs 1.060——**H1 排序在 RLVR
  实际生效阶段复现**;Spearman 0.858 vs 0.808
- 结论:H4(SNR 通道)跨家族复现且在标准注意力上更强;H1 排序在活跃 RLVR
  阶段跨家族复现(幅度均温和)。写作时注明饱和窗口的处理
- |ρ^Σ| med 0.325(192 captures)——σ/frame 对称性待同样检查
- 3B OPD ✅:R_enrich 1.082,spearman 0.809

家族轴主证据(Llama-3.2-1B,无饱和;teacher=3B):
- SFT:R_enrich 1.067,spearman 0.846
- OPD:R_enrich 1.073,spearman 0.818(稠密端与 SFT 基本持平,与 0.8B 结论一致)
- RLVR(无饱和 ✓,reward 全程 53-57%):R_enrich 1.057(三范式最低),spearman 0.806
- **家族轴终判**:H4 排序 SFT>OPD>RLVR 三模型全中(0.880/0.874/0.848、0.863/0.809/0.785、
  0.846/0.818/0.806);H1 修正为"稠密>RLVR"(SFT-OPD 子排序仅 Qwen 成立);
  H3 对称性在 Llama 复现(0.309 vs 0.307)
- 1B 与 3B 的 SFT 几乎同型(1.067 vs 1.073)——Llama 家族内尺度稳定

分家族(Qwen 0.8B,三范式一致):linear_attn 富集 ~0.65 **低于随机**,
mlp ~0.98,self_attn ~0.87-0.89。

### 叙事要点(当前证据支持的)
1. **梯度的谱-frame 几何由 objective 决定,optimizer 只调制**(H2,三对全部成立)——
   这是 C1 的直接证据,也是"分析 G 而非 W_T"方法论的回报。
2. **credit 密度的排序效应存在但温和**(H1 方向一致,~5% 跨度)——诚实呈现;
   RLVR 的差异更多体现在 SNR 结构(C3 主线)而非对角能量占比。
3. **per-mode 相关上 σ 与 frame 对称**(H3 证伪,呼应 Prop.7)——把 frame-dominance
   的机制归因收窄到方差/SNR 通道(§5),而非 reward-相关结构。可证伪量按设计工作。
4. **Muon 的谱富集在 RLVR 下消失**(H/G≈1.0)——与"RLVR 梯度谱信号弱"自洽,
   为 SSD 的 SNR 门控提供动机。
5. **线性注意力抑制谱对角**(架构效应,新观察)——独立小节或附录。

### 待填(依赖在途/作者B)
- Phase 2 H5/H6 表(作者B ×10)→ Fig.6/7:______
- E1 主表 + Fig.8(Muon 退化 vs SSD):______
- E2 消融表:______、E3 预测关散点:______、E4 α 轨迹:______
- 4B/9B 复现行:______;Llama RLVR/OPD:______
- 遗忘:MMLU(base vs 各法):______

---

## v2 数值登记处(2026-09-10 起;v1 登记处在上,保留不改)

来源:`results/v2/result_A/`(作者A)、`results/v2/result_B/`(作者B)。图:`figs/v2/`(`analysis_v2/plot_v2.py`)。

### A1:v1 捕获用 v2 指标重算(0.8B,seed 0,27 矩阵 × 50 保存步;8 微批次分成 4|4 两半)

| run | v1 谱富集 R_enrich | 交叉信号能量/噪声能量 | 信号>0 的行占比 | split-half Spearman(\|C\|_A,\|C\|_B) | v1 Spearman(\|C\|,SNR) | 交叉 Spearman(\|C\|_A,SNR_B) | \|ρ^Σ\| / \|ρ^frame\| |
|---|---|---|---|---|---|---|---|
| sft_adamw  | 0.885 | 0.0036 | 0.23 | 0.055 | 0.879 | 0.000 | — |
| sft_muon   | 0.886 | 0.0022 | 0.21 | 0.062 | 0.875 | 0.001 | — |
| opd_adamw  | 0.869 | 0.0073 | 0.27 | 0.055 | 0.875 | 0.002 | — |
| opd_muon   | 0.892 | 0.0038 | 0.24 | 0.057 | 0.872 | 0.001 | — |
| rlvr_adamw | 0.838 | 0.0004 | 0.17 | 0.062 | 0.848 | 0.000 | 0.285 / 0.290 |
| rlvr_muon  | 0.838 | 0.0004 | 0.17 | 0.055 | 0.844 | 0.001 | 0.295 / 0.298 |

- v2 脚本复算出的 v1 列与 v1 登记逐行一致(最大差 3e-6),重算无偏差。
- 前 100 步的交叉信号占比:SFT 0.008、OPD 0.011、RLVR -0.001;300 步后三者均为 0。
- 分家族(AdamW):linear_attn / mlp / self_attn 的信号占比 SFT 0.003/0.005/0.003,OPD 0.009/0.009/0.004,RLVR 0.000/0.001/-0.001。

**判定**:在 B=8 的样本量下,平均梯度中跨样本可复现的能量不到噪声的 1%,四分之三以上的矩阵-步组合分辨不出正信号;
v1 的 G 侧谱几何(R_enrich 贴近随机基线、家族差异、范式 ordering)全部处于噪声主导区,不能作为真实平均梯度谱占比的证据。
v1 的 H4 统计量(同一样本的 \|C\| 与 SNR)是耦合产物,换成独立两半后归零。RLVR 的 σ/frame 相关对称性复现。→ v2 H-A 在 0.8B 成立;E1 的样本量档位 B≥64 是必需的。

### A5:有限步 KL 探针(E0/E2;0.8B 冻结 checkpoint,fp32 前向,27 矩阵 × 4 单位范数方向 × 6 步长;32 条 GSM8K test 参考序列、3991 答案 token)

| 量 | 结果 |
|---|---|
| KL 对 η 的对数斜率,η∈[1e-3,1e-2] | 四个方向全部 2.01,矩阵间范围 [1.98, 2.15] —— 局部二次区成立 |
| 有效步长窗口 | η≥3e-4;η=1e-4 时 KL≈1e-11 为 fp32 底噪(符号不定)。bf16 autocast 前向在 η≤1e-2 全部失效(KL 停在 2.4e-4,纯缩放方向为 0) |
| 单位步长能量的 KL 代价,η=1e-3,按家族中位数(frame_rot / gauss / spec_rand / spec_scale) | linear_attn 8.5e-5 / 7.7e-5 / 7.1e-5 / 6.5e-5;mlp 2.6e-4 / 2.4e-4 / 3.2e-4 / 2.7e-4;self_attn 2.9e-4 / 3.6e-4 / 3.2e-4 / 1.2e-4 |
| 逐矩阵比值 spec_rand / frame_rot | 中位 0.95,范围 [0.26, 5.2]。q_proj 三层一致 0.27–0.31(谱步便宜),o_proj 三层一致 1.6–2.4(谱步贵),mlp 与 linear_attn 无一致方向 |
| 纯缩放方向 spec_scale / gauss | k_proj 三层 ≈0(−0.04 到 −0.14,k_norm 下缩放不变,Prop.6 型不变性的直接验证);其余 0.3–4.1 |

**判定**:(1) 有限步 KL 在 η∈[3e-4,3e-2] 内精确二次,§12.4 的局部二次假设成立;(2) 谱方向与 frame 方向的功能代价**按矩阵类型**有 3–4 倍的各向异性,但没有全局符号,家族中位数接近 1 —— v2 §6.3"无通用 frame 对称性定理"成立,同时说明 E4a 的等 Frobenius 匹配与等 KL 匹配在 q_proj/o_proj 上会差 3 倍,P3 结果需用本表做等 KL 换算后再下结论;(3) 训练循环的 bf16 前向不能用于任何 KL/Fisher 估计。

### B v2 交付登记(2026-09-11;来源 `results/v2/result_B/`,README 与 `B_P0_answers_v2.md`)

**环境。** B:torch 2.6.0+cu124、transformers 5.9.0(忽略 `dtype=fp32`,已加显式转换与断言);
16 个 P3/P4 run 代码 = 1a1e2f9 内容,2 个补跑 full 对照 = c4582b9。18 个 e4a run 的 ckpt 均核实为 F32。
与 A 环境(transformers 5.16.1)前向 A/B:权重逐位相同,top-1 一致 100%,KL 3.8e-4。

**P0。** Q1:v1 批次 = e5a25bc,不含 stop-ids 修复;6 个 v1 ckpt 重评差 ≤1.6 点、方向不定 → v1 SFT 行可用,
SFT 低于 base 为真实效应。Q2:捕获顺序正确;根因 = v1 批次 master 权重为 bf16(见 README §Environment);
修复后 `e4a_rlvr_spectrum_matched_s0` 的 H 自检 median R_sigma(H) = 0.935(修复前 0.215)。剩余 6.5% 来自捕获中 W 以 bf16 存储。
Q3 秒/步(A100/H100):RLVR full 27–31;spectrum/frame_matched 43–46(A100);random_ext 33/26;exact_iso 41.5;
SFT full 5.5/5.1;spectrum/frame_matched 13.5–14(H100);random_ext 4.8–5.0;exact_iso 18.4。

**P1(MMLU,均为 v1 bf16 run)。** e1_opd_adamw_s1(OPD full,500 步)0.247;phase2_rlvr_none 0.477;phase2_rlvr_frame_only 0.482。
→ OPD full 自身即把 MMLU 打到随机,§9.4 caveat 4 关闭(α 不是原因)。

**P2。** 24 个 v1 run 已用 v2 脚本重算;v1 run 为 bf16 master,**H 侧列不可用**,G 侧列可用。

**P3 + P4 等范数干预(0.8B,300 步,fp32,P=8;RLVR lr 2e-6、K=8;SFT lr 1e-5;GSM8K 500 题 / MMLU 1000 题;base 0.546 / 0.483)**

| objective | 干预 | 自由度 | scale(均值) | GSM8K s0 / s1 | MMLU s0 / s1 | GSM8K 均值 | MMLU 均值 |
|---|---|---|---|---|---|---|---|
| RLVR | full | mn | — | 0.412 / 0.494 | 0.237 / 0.232 | 0.453 | 0.234 |
| RLVR | frame_matched | mn | 1.00 | 0.584 / 0.472 | 0.248 / 0.228 | 0.528 | 0.238 |
| RLVR | exact_iso | mn | — | 0.600 / — | 0.255 / — | 0.600 | 0.255 |
| RLVR | spectrum_matched | r | 50.3 | 0.630 / 0.638 | 0.473 / 0.475 | **0.634** | **0.474** |
| RLVR | random_ext | r | 51.7 | 0.606 / 0.580 | 0.474 / 0.476 | 0.593 | 0.475 |
| SFT | full | mn | — | 0.298 / 0.304 | 0.239 / 0.239 | 0.301 | 0.239 |
| SFT | frame_matched | mn | 1.00 | 0.278 / 0.288 | 0.235 / 0.236 | 0.283 | 0.236 |
| SFT | exact_iso | mn | — | 0.290 / — | 0.240 / — | 0.290 | 0.240 |
| SFT | spectrum_matched | r | 50.4 | 0.382 / 0.392 | 0.471 / 0.484 | 0.387 | **0.478** |
| SFT | random_ext | r | 51.9 | 0.378 / 0.408 | 0.467 / 0.466 | **0.393** | 0.466 |

标准误:单次 GSM8K(500)0.022、MMLU(1000)0.016;两 seed 合并 0.016 / 0.011。
spectrum_matched − random_ext:RLVR GSM8K +0.041(≈2.6 合并 SE,单向、仅一范式),MMLU −0.001;SFT GSM8K −0.006,MMLU +0.012。
自由度分组差(r 维 vs mn 维):MMLU 两范式均 ≈ +0.24;GSM8K RLVR +0.16,SFT +0.09。

RLVR 训练 reward(50 步窗口均值):full s0 0.505→0.594→**0.628**→0.583→0.555→0.476;full s1 峰 0.594(101–150)→末 0.501;
frame_matched 峰 0.53/0.61 → 末 0.52/0.41;spectrum_matched 0.354→0.390→0.462→0.486→0.523→0.577(单调,末窗最高);
random_ext 0.372→…→0.563(单调)。exact_iso 单调至 0.590。
SFT 训练 loss 末窗:full 0.505/0.494,spectrum_matched 0.496/0.493,random_ext 0.495/0.492(无差异)。

**等范数协议的算术(A,2026-09-11)。** 设 P 为到 r 维字典的正交投影,Hp = s·P(H),‖Hp‖=‖H‖ ⇒ s=‖H‖/‖PH‖。
则 cos(Hp,H)=‖PH‖/‖H‖=1/s≈0.02,沿优化器步方向的推进 <Hp,H>/‖H‖² = 1/s ≈ 1/50。
即 spectrum_matched / random_ext 每步沿 H 的推进只有 full 的 2%,其余 98% 能量在与 H 正交、且限于 r 维字典内的方向。
等 Frobenius 步长 ≠ 等推进。这是 v3 必须修正的协议问题。

**P5(4B,500 步,v1 代码,bf16 master)。** base 0.862;p1_4b_sft_adamw 0.818;p1_4b_rlvr_adamw 0.932;e1_4b_rlvr_muon 0.918。
e1_4b_rlvr_ssd 停在 378 步未交付;p1_4b_opd_adamw 未跑。P6 未跑。

**v1 对照(bf16 master,隐式截断更新)。** phase2_rlvr_none 0.636 / 0.477;phase2_sft_none 0.412 / 0.366。
fp32 spectrum_matched(0.634 / 0.474)与 v1 bf16 full 几乎相同:v1 的 full 实际运行在"小有效步长"区间。

**dense lr×1 崩溃态诊断(A,2026-09-11,来自 P3 log.jsonl;RLVR,50 步窗)。**
有效 group 数(8 个中 reward 不全同)末三窗:full 4.08/3.62/3.53、frame_matched 4.06/4.28/4.44 与 4.30/4.12/3.78、exact_iso 3.96/4.16/4.36;
spectrum_matched 5.38/5.00/4.90 与 5.48/5.50/5.41、random_ext 5.26/5.40/4.82 与 5.28/5.14/4.82。
策略梯度 |loss| 首窗→末窗:dense 0.04–0.06 → 0.01–0.02;r 维 0.06 → 0.04–0.05。
两 seed 最终 GSM8K 差:full 0.082、frame_matched 0.112;spectrum_matched 0.008、random_ext 0.026。
→ lr×1 dense 为策略退化区,dense 行不能作比较锚点;v3 批次一先扫 lr 找健康 lr\*。

## v3 数值登记处(2026-09-12 起;v1/v2 登记处在上,保留不改)

设置沿用 P3:0.8B,fp32 master,300 步,P=8;RLVR lr×1 = 2e-6、K=8;SFT lr×1 = 1e-5;GSM8K 500 题 / MMLU 1000 题,base 0.546 / 0.483;
每 50 步中间评测。SE:GSM8K 0.022、MMLU 0.016(单 run)。

### A1:v3 smoke(job 9913754,contrib-gpuq)
- SFT spectrum_matched s_rel=3(8 步):scale_mean 50.3(max 91)、cos_mean 0.0235、step_norm_rel 3.0;H 自检 median R_sigma(H)=1.0 OK;捕获 G/W/H 全 fp32;eval_step000008_* 与 eval_*.json 生成。
- SFT random_ext s_rel=1:scale_mean 51.9、cos_mean 0.0209;同上全部通过。
- RLVR 两路(none+kl-beta 0.01、spectrum_matched):第一次用 max_new_tokens 128 全部截断(trunc_frac 1.0)、reward≈0、多步无有效 group → 末步评测被跳过;已改 train.py(跳过步也执行 ckpt/eval)并把 smoke 改 384 token 重跑。kl-beta 与 resp_len/trunc 日志路径本身正常。
- 复跑(job 9914443,384 token):RLVR none+kl-beta 0.01 末步 reward 0.06、trunc_frac 0.63、评测生成;RLVR spectrum_matched cos_mean 0.0235、scale_mean 50.4、H 自检 1.0、评测生成;SFT none(step_kl 参照)通过。**A1 通过。**

### E-v3-0a 累计位移 / 累计 KL(B,18 个 e4a ckpt)—— 待填
### E-v3-0b MMLU 格式检查(B,base + 4 ckpt)—— 待填
### E-v3-0c 单步 KL(A,job 9914606;SFT smoke 第 8 步,同 prompt 流/seed,27 个跟踪矩阵的 H 加到 base 上;32 条 GSM8K 参考序列,fp32,TF32 关)

| run | 干预 | ‖H‖_F(27 矩阵合计) | 单步 KL(nats/token) | KL/‖H‖² |
|---|---|---|---|---|
| smoke_v3_sft_none_s1 | none(full) | 3.24e-2 | **6.50e-3** | 6.20 |
| smoke_v3_sft_spectrum_matched_s3 | spectrum_matched,s_rel=3 | 1.22e-1 | **1.33e-4** | 0.0089 |
| smoke_v3_sft_random_ext_s1 | random_ext,s_rel=1 | 4.76e-2 | **5.10e-5** | 0.0225 |

→ 等 Frobenius 步长下,r 维投影更新的**功能步长(KL)比 full 小约 50–130 倍;按单位范数平方算小 300–700 倍**。
原因:full 的 Adam 步把能量集中在梯度方向(高曲率、高 KL/范数);投影到 r 维字典只保留 1/2500 能量再放大,
保留的主要是与梯度弱相关、低曲率的分量。A5 的"谱/frame 单位范数代价相同"只对随机方向成立,对实际更新不成立。
含义:(i)v2 P3 的 dense 崩溃 vs r 维健康首先是功能步长相差两个量级(M1);(ii)dense 的 lr 扫描是决定性实验,
预计 lr×1/10 左右的 dense 单步 KL 才与 r 维干预相当(KL ∝ lr²);(iii)v3 §3.2 的假设改写。
注意:单步、smoke 规模(4 prompts)、27 矩阵、H 加在 base 上。
第 4 步复核(job 9914627):full ‖H‖ 4.51e-2、KL **3.09e-2**、KL/‖H‖² 15.2;spectrum_matched(s_rel=3)‖H‖ 1.58e-1、KL **4.50e-4**、0.018;random_ext ‖H‖ 5.74e-2、KL **7.13e-5**、0.022。
→ 与第 8 步一致:功能步长小 70–430 倍,单位范数平方代价小约 700–850 倍。**E-v3-0c 结论稳定。**

### E-v3-1 dense lr 扫描 —— SFT 六个(A,job 9914659,A100.40gb;2026-09-12)

| lr 档 | lr | GSM8K s0/s1(均值) | MMLU s0/s1(均值) | 训练 loss 末 50 步 | MMLU 轨迹(50→300) | 判定 |
|---|---|---|---|---|---|---|
| ×1(v2 P3) | 1e-5 | 0.298/0.304(0.301) | 0.239/0.239 | 0.50 | — | FAIL(崩溃) |
| ×1/3 | 3.33e-6 | 0.416/0.424(0.420) | 0.289/0.316(0.302) | 0.46 | 0.42→0.35→0.26→0.29(s0);0.46→0.44→0.38→0.32(s1) | FAIL(MMLU 单调衰减) |
| **×1/10** | **1e-6** | 0.408/0.422(**0.415**) | 0.462/0.450(**0.456**) | **0.465** | 平稳 0.45–0.46 | **HEALTHY → SFT lr\* = 1e-6** |
| ×1/30 | 3.33e-7 | 0.378/0.402(0.390) | 0.464/0.457(0.460) | 0.498 | 平稳 | HEALTHY(学得更少) |

对照 r 维干预(lr×1,s_rel=1,等范数):spectrum_matched 0.387/0.478、random_ext 0.393/0.466,训练 loss 末窗 0.49–0.50。
→ **dense 在 lr×1/10 上与 r 维干预处于同一前沿**(GSM8K +0.02–0.03,MMLU −0.01–0.02,均在 1–2 SE 内),且训练 loss 更低(0.465 vs 0.49)。
E-v3-0c 的预测(lr\* ≈ 1/10)命中。SFT 侧:H-0 成立;H-4 成立(dense 崩溃可由步长解释);C5 在 SFT 上初判阴性。
MMLU 在 ×1/3 档随步数单调衰减、GSM8K 反而上升,是"学得越多忘得越多"的连续过程,不是格式突变(仍待 E-v3-0b 确认)。

### E-v3-2 / E-v3-3 批次二 —— SFT 八个(A,jobs 27170/27171,MIG 3g.40gb;2026-09-13)

lr\* = 1e-6,s_rel = 1,300 步,2 seed;300 步终点,seed 合并;SE:GSM8K 0.016、MMLU 0.011(合并 2 seed)。

| 更新 | 维度 | GSM8K s0/s1(均值) | MMLU s0/s1(均值) |
|---|---|---|---|
| full(批次一) | mn | 0.408/0.422(0.415) | 0.462/0.450(0.456) |
| frame_matched | mn | 0.388/0.412(0.400) | 0.457/0.450(0.454) |
| exact_iso | mn | 0.398/0.408(0.403) | 0.453/0.449(0.451) |
| spectrum_matched | r | 0.392/0.386(0.389) | 0.472/0.476(0.474) |
| random_ext | r | 0.354/0.396(0.375) | 0.474/0.476(0.475) |
| full lr×1/30(批次一) | mn | 0.390 | 0.461 |

两两差(margin GSM8K 0.03 / MMLU 0.02):
- full − frame_matched:+0.015 (t=0.7) / +0.003 (t=0.2);full − exact_iso:+0.012 (t=0.6) / +0.005 (t=0.3);frame − iso:−0.003 / +0.003。→ **H-2 成立(SFT):保谱/去谱在健康区不可分,C4 成立。** v2 的"mn 维崩到 0.24"全部是 lr×1 崩溃区效应。
- spectrum_matched − random_ext:+0.014 (t=0.6) / −0.001 (t=−0.1)。→ **H-1 在第二个 lr(lr\*)上复现(SFT,2 seed)**;lr×1 上(v2 e4a)−0.006 / +0.011,同样不可分。
- r 维在 lr\* 与在 lr×1 结果相同(spectrum 0.389/0.474 vs 0.387/0.477;random 0.375/0.475 vs 0.393/0.466):**r 维干预对 lr 的 10 倍变化不敏感**,与 E-v3-0c(投影更新的功能步长本来就小两个量级)一致。
- 前沿(H-3 初判):r 维两点 (0.38–0.39, 0.474) 与 dense lr\* (0.415, 0.456)、dense lr×1/30 (0.390, 0.461) 相比,GSM8K 低 0.03–0.04 (t≈1.2–1.8)、MMLU 高 0.018 (t≈1.2),均未过 |t|≥2;dense lr×1/30 与 spectrum_matched 几乎同点。r 维 MMLU 与 mn 维不重叠(+0.021,置换检验 p=0.0048,B 复核),但 GSM8K 同时更低,是权衡点不同而非占优。→ 无一方占优(非"前沿重合"),**C5 在 SFT 上维持阴性初判**;待 E-v3-5(s_rel 3/10)看 r 维能否沿前沿移动。
- 轨迹:所有六族 MMLU 在 50→300 步内平稳(0.45–0.48),GSM8K 无趋势;lr\* 上无崩溃迹象。

图:`figs/v3/fig1_frontier.pdf`、`fig1b_trajectories.pdf`、`frontier_points.csv`(`analysis_v3/frontier.py results/v3/result_A`)。

批次三 SFT 已提交(2026-09-13,jobs 38326/38327,`slurm_v3/batch3_sft.sbatch`):E-v3-4 r 维第三 seed(lr×1)×2 + E-v3-5 s_rel∈{3,10}×两字典(lr\*,seed 0)×4。

### E-v3-4 / E-v3-5 批次三 —— SFT 六个(A,jobs 38326/38327,MIG 3g.40gb;2026-09-13)

**E-v3-4 r 维第三 seed(lr×1 = 1e-5,s_rel=1)**:spectrum_matched s2 0.400/0.473,random_ext s2 0.386/0.473。
3 seed 合并:spectrum 0.391/0.476,random 0.391/0.469;差 +0.001 (t=0.0) / +0.007 (t=0.6)。→ **H-1 成立(SFT,3 seed,lr×1),并已在 lr\* 复现(2 seed)。C2 成立。**

**E-v3-5 s_rel 扫描(lr\* = 1e-6,seed 0,300 步终点)**:

| 字典 | s_rel=1(2 seed 均值) | s_rel=3 | s_rel=10 |
|---|---|---|---|
| spectrum_matched | 0.389 / 0.474 | 0.378 / 0.475 | 0.368 / 0.468 |
| random_ext | 0.375 / 0.475 | 0.378 / 0.462 | 0.366 / 0.462 |
| dense 参照 | lr×1/30:0.390 / 0.461;**lr\*:0.415 / 0.456**;lr×1/3:0.420 / 0.302 | | |

- 把逐步等范数步长放大 10 倍,r 维两字典都**没有沿前沿移动**:GSM8K 0.37–0.39 不升(s_rel=10 反而略降 0.02,1 SE 内),MMLU 0.46–0.48 不降,也不崩。
- 连同 lr 十倍(lr\* vs lr×1)不敏感:r 维干预在步长范数跨约 10 倍的范围内都停在同一点(s_rel 与 lr 作用在同一个量上:步长 = s_rel × 该 lr 下 dense 步长范数,s_rel=10@lr\* 与 s_rel=1@lr×1 步长范数仅差 5%,是同一工作点,B 复核;此前写的"100 倍"有误) (0.37–0.40, 0.46–0.48),该点与 dense lr×1/30 (0.390, 0.461) 重合;dense lr\* (0.415, 0.456) 在其右下方 1–2 SE,dense lr×1/3 则是"GSM8K 更高、MMLU 崩"的另一端。
- **H-3 裁决(SFT):前沿重合/交叉,r 维约束不在 dense 前沿右上方 → C5 阴性(SFT,终判)。** 写法:"逐步等范数下 r 维约束不优于调 lr;它只是把训练锁在 dense 的小步长点上,且对步长旋钮不响应"。
- 附带观察(进 §3.3 开放问题):r 维更新的任务指标对 lr 和 s_rel 都不响应,但训练 loss 是响应的且单调(末 20 步均值,seed 0:spectrum 0.598→0.554→0.519,random 0.596→0.551→0.519,s_rel 1→3→10;lr×1 s2 为 0.505;seed 间差 ≈0.02):优化在推进,GSM8K/MMLU 不动;与 E-v3-0c(功能步长小两个量级)一致,但"为何不随 s_rel 增大"未解释。

图/CSV 已更新:`figs/v3/fig1_frontier.pdf`、`fig1b_trajectories.pdf`、`frontier_points.csv`(现含 s_rel 与 lr1_s2 点)。

**SFT 侧三批总结(2026-09-13)**:H-0 ✅(lr\*=1e-6)、H-1 ✅(3 seed)、H-2 ✅(C4)、H-3 ✗(C5 阴性)。SFT 侧 v3 实验全部完成;RLVR 侧待 B 批次一。

### E-v3-1 dense lr 扫描 —— RLVR 六个(B 报告,2026-09-13;run 目录待交付,A 复核待做)—— RLVR lr\* = 2e-7(×1/10)

B 报告(MMLU 终值 / 两 seed GSM8K 差 / reward 回落):×1/30 0.479 / 0.000 / 0.0295;**×1/10 0.470 / 0.028 / 0.026 → HEALTHY,余量 +0.017**;×1/3 0.452 / 0.072 / 0.016 → MMLU 与离散双 FAIL。
B 另报:E-v3-0a 秩受限更新权重空间位移更大(0.89–0.94 vs 0.755)但函数空间位移小 10–30 倍;单步 KL 优势(276–845×)到 300 步累计只剩 3.5–4.7×(不可外推);E-v3-0b:v2 lr×1 dense 的 MMLU 崩塌是格式假象(argmax_in_letters 0.025 → H-4 成立),v3 lr×1/3 的下降是真实能力损失(格式 0.865 与 base 持平)。数字待 run 目录到齐后由 A 复核登记。

**A 复核(2026-09-16,A 自跑 dense lr\* ×2 seed,`slurm_v3/batch_rlvr.sbatch` 0–1,MIG 3g.40gb)**:full lr\*=2e-7 s0/s1 GSM8K 0.656/0.634(0.645)、MMLU 0.471/0.474(0.472);reward 峰 0.577、末 100 步 0.559、回落 0.018;两 seed GSM8K 差 0.022;trunc_frac 末 50 步 0.11 → **HEALTHY,与 B 报告一致(B:0.626–0.654 / 0.470),RLVR lr\* = 2e-7 确认,H-0 成立(RLVR)**。余量:MMLU +0.019、回落 +0.012。

### E-v3-2 / E-v3-3 / E-v3-4 / E-v3-5 —— RLVR 十六个(A 自跑,2026-09-16,`slurm_v3/batch_rlvr.sbatch`,MIG 3g.40gb;B 的 RLVR 交付始终未到,A 于 09-16 02:30 UTC 起自跑,约 5 h/run)

lr\* = 2e-7,300 步,`--eval-every 50`;GSM8K n=500、MMLU n=1000;SE 合并 2 seed:GSM8K 0.016、MMLU 0.011。base 0.546/0.483。所有 11 个配置按 H-0 三判据均 HEALTHY(`analysis_v3/health.py`)。表与两两检验由 `analysis_v3/verdict_rlvr.py results/v3/result_A` 生成。
运维记录:MIG 3g.40gb 上评测子进程与训练进程共卡,干预 run(含 dense 在 step 150)在 step 50/150 的 MMLU 评测 OOM(需 7.56 GiB,余 5–7.5 GiB);`train.py run_evals` 加评测前 `torch.cuda.empty_cache()` + 失败时降 batch 重试(f0a37a3),全部 16 个 run 在补丁后重跑,**无一触发降 batch 重试**,评测口径与 SFT 完全一致。

| 更新 | 维度 | lr | s_rel | GSM8K s0/s1(均值) | MMLU s0/s1(均值) | reward 末 100 | trunc 末 50 |
|---|---|---|---|---|---|---|---|
| full | mn | lr\* | – | 0.656/0.634(0.645) | 0.471/0.474(0.472) | 0.559 | 0.11 |
| frame_matched | mn | lr\* | 1 | 0.628/0.610(0.619) | 0.478/0.467(0.473) | 0.543 | 0.15 |
| exact_iso | mn | lr\* | 1 | 0.702/0.690(**0.696**) | 0.466/0.462(0.464) | **0.623** | 0.15 |
| spectrum_matched | r | lr\* | 1 | 0.554/0.572(0.563) | 0.474/0.480(0.477) | 0.341 | 0.47 |
| random_ext | r | lr\* | 1 | 0.546/0.564(0.555) | 0.476/0.478(0.477) | 0.340 | 0.48 |
| spectrum_matched | r | lr\* | 3 | 0.570(s0) | 0.478 | 0.400 | 0.33 |
| random_ext | r | lr\* | 3 | 0.558(s0) | 0.478 | 0.405 | 0.32 |
| spectrum_matched | r | lr\* | 10 | 0.594(s0) | 0.475 | 0.486 | 0.18 |
| random_ext | r | lr\* | 10 | 0.586(s0) | 0.474 | 0.480 | 0.23 |
| spectrum_matched | r | lr×1 | 1 | 0.578(s2) | 0.474 | 0.545 | 0.10 |
| random_ext | r | lr×1 | 1 | 0.620(s2) | 0.478 | 0.569 | 0.14 |

两两差(GSM8K / MMLU,t 为合并 SE):
- **H-2(E-v3-2),3 seed,exact_iso 臂已因引擎缺陷移除(见下)**:终点 GSM8K/MMLU:full 0.656/0.634/0.664(**0.651**)/0.472,frame_matched 0.628/0.610/0.636(**0.625**)/0.475。full − frame_matched +0.027 (t=1.5) / −0.003 (t=−0.2):GSM8K 差在 margin 0.03 内、|t|<2 → **H-2 成立(RLVR,以 frame_matched 论证):等范数保谱更新与 dense 不可分**。趋势上 frame_matched 略慢(reward 末 100 步 0.541 vs 0.570,三 seed 0.55/0.53/0.54 vs 0.54/0.58/0.59;单步 KL/‖H‖² 低 20–36%,E-v3-0c 扩展),但未过判据;写作时以"不可分、若有差异则保谱略慢"表述。[作废记录:exact_iso 0.702/0.690/0.648(0.680)/0.468,iso − frame +0.055 (t=3.2)、iso − full +0.029 (t=1.7),reward 末 100 步 0.62–0.63;该臂含每步 SVD 数值噪声,不作证据。]
- **H-1(E-v3-3)**:spectrum_matched − random_ext +0.008 (t=0.4) / 0.000 (t=0.0) → **H-1 在 RLVR lr\* 上成立(2 seed);C2 在两种范式上都成立**。lr×1 单 seed −0.042 (t=−1.4) / −0.004,单 seed 不裁决。
- **前沿(E-v3-3/5,C5)**:r 维两字典在 lr\* 上被 dense 严格占优:spectrum − full −0.082 (t=−3.8) / +0.005 (t=0.3),random − full −0.090 (t=−4.1) / +0.005 (t=0.3)。r 维在 lr\* 上 reward 末 100 步只有 0.34(dense 0.56),trunc_frac 0.47(base 水平)——**等范数 r 维更新在 RLVR 下几乎不训练**,与 E-v3-0c(功能步长小两个量级)一致。→ **C5 在 RLVR 上阴性,且比 SFT 更强(SFT 是"无一方占优",RLVR 是 dense 占优)**。
- **H-3(E-v3-5,s_rel 扫描,seed 0)**:与 SFT 不同,RLVR 的 r 维更新**对步长是响应的**:s_rel 1→3→10 GSM8K 0.56→0.57→0.59(+0.031, t=1.1,两字典一致),reward 末 100 步 0.34→0.40→0.48,trunc 0.47→0.33→0.20;lr×1(≈ s_rel 10 的步长)random_ext 0.620 / reward 0.569。但直到 lr×1 也只追到 dense lr\* 的 GSM8K 以下(0.58–0.62 vs 0.645),MMLU 全程 0.474–0.480 与 dense 0.472 在 1 SE 内。→ **r 维前沿没有落在 dense 前沿右上方(预注册判据),H-3 阴性(RLVR);r 维约束的效果是"学得慢",不是"折中更好"**。
- 对 v2 P3 的校正:v2 的 RLVR r 维 lr×1(0.63/0.59)与 v3 lr×1 单 seed(0.58–0.62 / 0.47)量级一致;v2 的"mn 维崩到 0.24"已由 E-v3-0b 确认是格式假象 + lr×1 崩溃区。

**RLVR 侧总结(2026-09-16 晚,3 seed,exact_iso 臂移除后)**:H-0 ✅(lr\*=2e-7)、H-1 ✅、**H-2 ✅(frame_matched ≈ full,+0.027,t=1.5)**、H-3 ✗(C5 阴性,dense 占优)。
**两范式对照**:C1–C3 两侧成立;**C4 两侧成立(保谱 frame_matched ≈ dense;exact_iso 臂因引擎数值缺陷移除)**;C5 两侧阴性。r 维更新在 SFT 上对步长不响应、在 RLVR 上响应但落后 dense。v3 §4 的裁决表不需要改 C4;需要改的是 C4 的证据基础(只用 frame_matched)和 exact_iso 的处理;**由用户决定**;A 未改企划书。
图:`figs/v3/fig1_frontier.pdf`、`fig1b_trajectories.pdf`、`frontier_points.csv` 已含 RLVR 全部 11 个配置。

**E-v3-1 RLVR 闸门,A 自跑补齐(2026-09-16 晚,`batch_rlvr.sbatch` 19–22)**:lr×1/3 = 6.67e-7 s0/s1 GSM8K 0.676/0.728(0.702)、MMLU 0.471/0.452(0.462)、reward 回落 0.011;两 seed GSM8K 差 0.052 > 0.04 → **FAIL(离散)**,s1 的 MMLU 在第 150 步跌到 0.416、末点 0.452 压线。与 B 报告方向一致(B:差 0.072、MMLU 0.452)。**RLVR lr\* = 2e-7 维持**。lr×1/30 = 6.67e-8 A 两 seed(batch_rlvr 21–22)已完成,与 B 两 seed 合并为 4 run,见"B v3 交付登记"合并表更新: `v3_rlvr_full_lr0.033                      4                0.548/0.562 | 0.562/0.562  0.558  0.477`;gate lr x1/30 - lr*: GSM8K -0.087 (t_binom=-6.2, t_emp=-12.6)  MMLU +0.007 (t=+0.6) → 健康但学得少(与 B 一致),lr\* = 2e-7 不变。
副产品:dense lr×1/3 的 (0.702, 0.462) 与 exact_iso lr\* 的 (0.680, 0.468) 几乎同点,且两者的 seed 离散都超 0.04(0.052 / 0.054)。

### E-v3-0a 累计位移 / 累计 KL —— A 自跑,v3 全部 39 个终点 ckpt(job 315249,`analysis_v3/cum_kl.py --glob 'v3_*'`;`results/v3/result_A/cum_kl/`)

disp = 186 个矩阵 ‖W_300 − W_base‖_F 的平方和开方(绝对值);KL = base→ckpt 逐 token 前向 KL(A5 参考集,fp32)。

| 范式 | 更新 | lr / s_rel | disp(seed 列表) | KL(seed 列表) |
|---|---|---|---|---|
| RLVR | full | lr\* | 0.096 / 0.097 / 0.094 | 0.60 / 0.67 / 0.52 |
| RLVR | frame_matched | lr\* | 0.093 / 0.093 / 0.089 | 0.50 / 0.52 / 0.32 |
| RLVR | exact_iso | lr\* | **0.638 / 0.640 / 0.642** | 0.66 / 0.62 / 0.52 |
| RLVR | spectrum_matched | lr\* | 0.136 / 0.136 | **0.0046 / 0.0052** |
| RLVR | random_ext | lr\* | 0.131 / 0.136 | **0.0042 / 0.0046** |
| RLVR | spectrum / random | lr\*, s_rel 3 | 0.34 / 0.35 | 0.016 / 0.014 |
| RLVR | spectrum / random | lr\*, s_rel 10 | 0.95 / 0.98 | 0.113 / 0.033 |
| RLVR | spectrum / random | lr×1 s2 | 0.89 / 0.94 | 0.122 / 0.165 |
| SFT | full | lr\* | 0.490 / 0.490 | 0.87 / 0.82 |
| SFT | frame_matched | lr\* | 0.490 / 0.490 | 0.87 / 0.82 |
| SFT | exact_iso | lr\* | 0.796 / 0.792 | 0.84 / 0.80 |
| SFT | spectrum / random | lr\* | 1.74 / 2.05 | 0.50 / 0.51 |
| SFT | spectrum / random | lr\*, s_rel 10 | 6.38 / 7.36 | 0.68 / 0.67 |
| SFT | full | lr×1/30 · lr×1/3 | 0.26 · 1.26 | 0.87 · 0.85 |

- **r 维在 RLVR 下几乎不动**:lr\* 累计 KL 是 dense 的 <1%(0.005 vs 0.6),s_rel 10 或 lr×1 也只到 dense 的 5–25%;SFT 下 r 维累计 KL 达 dense 的 60%(0.50 vs 0.85)。与 B 在 v2 e4a 上的读法(r 维权重位移更大、函数位移小 10–30 倍)同向,v3 上 RLVR 的差距更大。这解释了 RLVR 前沿上 r 维被 dense 占优、SFT 上只是"换权衡点"。
- exact_iso 的累计权重位移是 dense 的 6.7 倍(RLVR)/ 1.6 倍(SFT),累计 KL 与 dense 相同 → 见"exact_iso 引擎缺陷":这是每步 SVD 数值噪声的随机游走,不是设计效应。
- SFT 累计 KL 对 lr 几乎不敏感(lr×1/30 到 ×1/3 都是 0.85–0.87),但位移 5 倍:SFT 的函数位移在这个区间饱和,MMLU 的差异不是"离 base 多远"能解释的。

### E-v3-0b MMLU 格式检查 —— A 自跑,base + 12 个 v3 ckpt(job 315249,`analysis_v3/mmlu_format.py`,200 题;`results/v3/result_A/mmlu_format/`)

argmax 落在选项字母上的比例(base 0.865):RLVR lr\* 五种更新 0.84–0.96,RLVR random lr×1 0.89,SFT lr\* 0.90–1.0,SFT lr×1/30 0.985,**SFT lr×1/3 0.30**。限定字母打分的 MMLU(base 0.475):健康 run 0.43–0.475,SFT lr×1/3 0.265。
→ 所有健康区 run **没有格式假象**;SFT lr×1/3 的 MMLU 0.30 同时是格式漂移(0.30)和真实损失(限定字母 0.265 ≈ 随机 0.25)。B 报的 v2 lr×1 dense 崩到 0.24 = 格式假象(0.025)、v3 RLVR lr×1/3 = 真实损失(格式 0.865),与此一致。exact_iso RLVR 的限定字母 MMLU 0.43 略低于 dense 0.46(200 题,SE≈0.035,不裁决)。

### E-v3-0c 扩展:exact_iso / frame_matched 的单步功能步长 —— A 自跑(`slurm_v3/step_kl_iso.sbatch`,10 个 8 步捕获 + step_kl;`results/v3/result_A/step_kl_iso_{sft,rlvr}_step{8,4}`)

同 prompt 流、seed 0、lr×1,27 个跟踪矩阵的实际施加更新 H(引擎之后捕获)加到 base 上,32 条参考序列 fp32 前向。KL 单位 nats/token,‖H‖ 为 27 矩阵总 Frobenius 范数。

| 范式 / 步 | none | frame_matched | exact_iso | spectrum_matched | random_ext |
|---|---|---|---|---|---|
| RLVR 步 4:‖H‖ / KL / KL/‖H‖² | 0.0090 / 1.19e-3 / 14.5 | 0.0092 / 9.79e-4 / 11.6 | 0.0108 / 7.75e-4 / 6.7 | 0.0091 / 3.2e-7 / 0.004 | 0.0091 / 2.7e-7 / 0.003 |
| RLVR 步 8 | 0.0065 / 3.91e-4 / 9.3 | 0.0066 / 2.59e-4 / 5.9 | 0.0088 / 2.39e-4 / 3.1 | 0.0066 / 1.5e-7 / 0.003 | 0.0066 / 1.0e-7 / 0.002 |
| SFT 步 4 | 0.0452 / 3.00e-2 / 14.7 | 0.0452 / 3.04e-2 / 14.9 | 0.0452 / 2.85e-2 / 13.9 | 0.0572 / 1.0e-4 / 0.032 | 0.0574 / 7.2e-5 / 0.022 |
| SFT 步 8 | 0.0325 / 8.92e-3 / 8.4 | 0.0325 / 8.40e-3 / 7.9 | 0.0329 / 9.05e-3 / 8.3 | 0.0473 / 7.4e-5 | 0.0477 / 5.2e-5 |

- **exact_iso 的单步功能步长不比 dense 大**:RLVR 下 KL 是 dense 的 0.65 / 0.61(步 4 / 8),单位范数平方的 KL 只有 dense 的 46% / 33%;SFT 下三者相同。**M1"有效步长更大"不能解释 exact_iso 在 RLVR 上学得更快**:它单步走得少、累计走得一样远(E-v3-0a KL 相同)、权重却走了 6.7 倍。
- r 维(spectrum / random)单步 KL 比 dense 小 3–4 个数量级(RLVR)/ 2–3 个数量级(SFT),复现 E-v3-0c 主结果,这次 s_rel=1、两字典与 none 同口径(回应 B 对 s_rel=3 那个点的保留)。
- RLVR 下 frame_matched 的 KL/‖H‖² 比 dense 低 20–36%:保谱把更新压进已有奇异方向,单步功能效率更低,与它 reward 最慢一致。

### exact_iso 引擎缺陷 —— 确认(A,2026-09-16 晚;`analysis_v3/spectrum_drift.py`、`analysis_v3/exact_iso_noise.py`;`results/v3/result_A/spectrum_drift.{csv,log}`、`exact_iso_noise.log`)

设计:`intervene_engine.py` 的 exact_iso 在每个 dense 步之后对 W_after 做 SVD,把奇异值重置为构造时的锚定谱 `S_anchor`,只让奇异向量变化(frame-only 更新),**没有等范数匹配**。
实测:
- 终点 ckpt 的奇异值**没有**钉在 base 上:186 个矩阵的中位相对谱漂移 exact_iso 2.8e-4 vs dense 2.6e-5(RLVR,两 seed 一致;SFT 2.8e-4 vs 4.6e-5)= 11 倍,与 B 的"33 倍"同向。
- 漂移从第 4 步起就是 2.8e-4 且不再增长(klcap 捕获 W:步 4 = 2.81e-4、步 8 = 2.76e-4),说明是**常数偏置**而非累积。
- 根因:GPU fp32 `torch.linalg.svd/svdvals`(cuSOLVER)在 4096×1024 上的精度只有 ~1e-4 相对(锚定谱 vs fp64 真值 1.36e-4;CPU LAPACK fp32 为 7e-7)。TF32 开或关都一样。用 base W 复现引擎的一步:dense 大小的步(‖H‖/‖W‖ = 6.5e-5)之后,谱重置引入的修正量是 **‖H‖ 的 4.4 倍**(第 1 步)、0.7 倍(第 10 步),W 立即偏离 base 2.9e-4,之后每步继续注入与 dense 步同量级的数值噪声。
- 后果:exact_iso 臂 = dense 更新 + 每步同量级的 SVD 数值噪声 + 常数谱偏置。这解释了 E-v3-0a 的 6.7 倍权重位移(噪声随机游走)和 E-v3-0c 扩展里它单步范数大 20–35% 但 KL 更小。它**不是**干净的"去谱"对照。
**裁决:B 的判断正确,exact_iso 臂从 C4 证据中移除(RLVR 与 SFT 两侧);C4 只用 frame_matched(等范数、保谱)对 full 论证。** exact_iso 在 RLVR 上 reward 更高这一现象改记为开放问题(等范数噪声注入是否有利于 RLVR 探索),不进入主线。修复方案(若要保留该臂):锚定谱与每步 SVD 改用 fp64(每步 186 个 fp64 SVD,RLVR 单 run 估计 +1–2 h),并加等范数匹配;是否重跑 4 个臂由用户决定。

### B v3 交付登记(2026-09-16 21:34 包,`results/v3/result_B/`;21 个 RLVR run 目录 + gate 表 + E-v3-0a(v2 e4a)+ E-v3-0b + exact_iso 谱检查 + 复核文档;B 环境 A100-80G,transformers 5.16.1,ckpt 全 F32)

- **B 独立确认 exact_iso 缺陷**:fp64 CPU svdvals,8 个矩阵中位谱漂移 exact_iso 3.08e-4 / 3.06e-4 vs dense 9.3e-6(33 倍),frame_matched 6.9e-8(比 dense 更贴近 base 135 倍);v2 e4a exact_iso 同样。与 A 的 GPU 检查(186 矩阵中位 2.8e-4 vs 2.6e-5)一致。**两边独立得到同一结论,exact_iso 臂作废。**
- **同 seed 跨硬件不复现**:17 组同 seed 同配置对的 GSM8K 绝对差中位 0.010、最大 0.044,与换 seed 同量级 → A(MIG)与 B(A100)的 run 按独立样本合并,不做同 seed 配对。
- B 的 gate:exact_iso FAIL(离散 0.060)、spectrum s_rel=10 FAIL(decline 0.032)、其余 HEALTHY;MMLU 稳定性检查全 PASS。
- **A+B 合并裁决(`analysis_v3/verdict_rlvr_pooled.py`,`results/v3/result_A/verdict_rlvr_pooled.txt`;GSM8K 均值,n = run 数)**:

| 配置 | n | GSM8K(A | B) | 均值 | MMLU |
|---|---|---|---|---|
| full lr\* | 6 | 0.656/0.634/0.664 \| 0.626/0.640/0.654 | 0.646 | 0.471 |
| frame_matched lr\* | 5 | 0.628/0.610/0.636 \| 0.650/0.644 | 0.634 | 0.473 |
| exact_iso lr\*(作废) | 5 | 0.702/0.690/0.648 \| 0.706/0.646 | 0.678 | 0.469 |
| spectrum_matched lr\* | 4 | 0.554/0.572 \| 0.554/0.546 | 0.556 | 0.478 |
| random_ext lr\* | 4 | 0.546/0.564 \| 0.556/0.558 | 0.556 | 0.477 |
| spectrum / random, s_rel 3 | 2+2 | spectrum 0.570 \| 0.550;random 0.558 \| 0.560 | 0.560 / 0.559 | 0.479 / 0.478 |
| spectrum / random, s_rel 10 | 2+2 | spectrum 0.594 \| 0.608;random 0.586 \| 0.588 | 0.601 / 0.587 | 0.474 / 0.472 |
| spectrum / random, lr×1 s2 | 2+2 | spectrum 0.578 \| 0.614;random 0.620 \| 0.624 | 0.596 / 0.622 | 0.476 / 0.476 |
| full lr×1/3 | 4 | 0.676/0.728 \| 0.660/0.732 | 0.699 | 0.457 |
| full lr×1/30 | 2(B) | – | 0.562/0.562 | 0.562 | 0.479 |

  - **H-2**:full − frame_matched **+0.012**(t_binom 0.9,t_emp 1.3)/ MMLU −0.003 → **成立(RLVR,合并 11 个 run)**。A 侧 frame 偏低、B 侧持平,合并后在 margin 内。
  - **H-1**:lr\* 上 spectrum − random **0.000**;lr×1 上 −0.026(t≈−1.3,4 run)→ 成立;B 提醒 lr\* 上两臂都在 base 附近,检验力弱,lr×1 才是有效检验。
  - **前沿 / C5**:r 维 lr\* 比 dense 低 0.089 / 0.090(t_emp −11 / −13),MMLU +0.006 → dense 严格占优,C5 阴性(RLVR,终版)。
  - **H-3**:s_rel 10 两字典 +0.03–0.045(2+2 run),lr×1 +0.04–0.07;r 维响应步长,但落在 dense lr 前沿上(B 按 dense 前沿线性插值,预期 MMLU 0.472–0.474,实测 0.472–0.478,1 SE 内)→ 阴性(终版)。
  - **gate**:lr×1/3 四 run GSM8K 0.660–0.732、MMLU 0.457 → 离散与 MMLU 双失败;lr×1/30 健康但余量 0.0005;**lr\* = 2e-7(终版)**。
- B 对 A 的 SFT lr\* 复算(`health_v3_sft_gate.*`)与 A 一致;E-v3-0b 在 v3 lr×1/3 上:格式 0.865 完好、限定字母准确率 0.425–0.430,真实损失。
- B 的 E-v3-0a 是 v2 e4a 的 18 个 ckpt(`cum_kl/`),与 A 的 v3 版本互补,不重复。

### exact_iso 修复版重跑 —— SFT 两个(A,2026-09-17,`slurm_v3/exact_iso_fixed.sbatch` 3–4;引擎修复 6f2c741:CUDA SVD 改 gesvda + 等范数匹配)

| 更新 | GSM8K s0/s1(均值) | MMLU s0/s1(均值) | 末 20 步 loss | scale_mean / cos_mean |
|---|---|---|---|---|
| exact_iso(修复版) | 0.410/0.402(0.406) | 0.460/0.450(0.455) | 0.479 / 0.461 | 1.000 / 1.000 |
| full(参照) | 0.408/0.422(0.415) | 0.462/0.450(0.456) | 0.478 | – |
| frame_matched(参照) | 0.388/0.412(0.400) | 0.457/0.450(0.454) | – | 1.0003 / 0.9997 |

- exact_iso − full:GSM8K −0.009 (t=−0.4) / MMLU −0.001 (t=−0.1);exact_iso − frame_matched:+0.006 / +0.001。→ **H-2 在 SFT 上完整成立:保谱、去谱、dense 三者不可分,C4(SFT)完整。**
- **机理注记**:修复版的 scale_mean = 1.000、cos_mean = 1.000,即"把 W 的奇异值钉回 base 谱"这一步对 dense 更新几乎没有改动——健康 lr 下的 dense 步本身就几乎不改变奇异值(与 Jin 等 2509.12235 "SFT 与 RL 下奇异值变化 ~0.005"一致)。所以 exact_iso 在健康区是一个近乎空的干预;"去谱 ≈ dense"成立的原因是 dense 本来就不改谱,而不是"改谱与否无所谓"。写作时按此表述。旧 exact_iso 的 6.7 倍位移完全是数值噪声。
- **RLVR 三个 seed(修复版,A,2026-09-17)**:GSM8K 0.618/0.636/0.608(**0.621**)、MMLU 0.473/0.471/0.477(0.474)、reward 末 100 步 0.53–0.55;scale_mean 0.986、cos_mean 1.014(RLVR 的 dense 步比 SFT 多改一点奇异值,钉谱后更新缩 1.4%)。exact_iso − full −0.031(t_binom −1.7,t_emp −2.5)/ MMLU +0.002;exact_iso − frame_matched −0.004。→ **RLVR 上去谱 = 保谱 ≈ dense 略低 0.03(压 margin,|t_binom|<2),与 frame_matched 同一水平;旧 exact_iso 的"高 0.05"完全是数值噪声效应。** H-2 在 RLVR 上的表述:保谱与去谱互相不可分,二者比 dense 慢一点(0.03,边界),C4 以"不改谱的更新不优于 dense"成立。

**余量登记(B 指出,A 确认)**:SFT lr\* = 1e-6 的 H-0 余量仅 +0.003(均值 0.456 vs 0.453;s1 终值 0.450;末三点均值读法 0.4525 → FAIL);RLVR lr\* 两种读法均 PASS。引用 SFT lr\* 必须连余量;SFT dense 参照同时报 lr×1/30(0.390/0.461,余量 +0.008)。SFT 三批的 H-1/H-2/H-3 裁决不受影响(均为同 lr 下的族间比较,且 lr×1/30 点与 r 维干预重合)。
**命名说明**:`smoke_v3_*_s<N>` 的 `_s<N>` 是 s_rel,不是 seed;五个 smoke 均 seed 0(E-v3-0c 的同 prompt 流前提成立)。

## v4 数值登记处(2026-09-17 起;v1–v3 登记处在上,保留不改)

### E-v4-1 任务外 KL(A,job 346399;参考集 = MMLU validation 题干 + 选项 64 条,与评测用的 test 分开;`analysis_v4/cum_kl.py --reference offtask`、`step_kl.py --reference offtask`;`results/v4/result_A/cum_kl_offtask/`、`step_kl_offtask_*`)

**累计(300 步终点)任务外 KL(nats/token)与任务 KL、MMLU(1000 题)**

| 范式 | 更新 | lr / s_rel | 任务外 KL | 任务 KL | 任务外/任务 | MMLU |
|---|---|---|---|---|---|---|
| SFT | full | ×1/30 | 0.015 | 0.87 | 0.02 | 0.461 |
| SFT | full | ×1/10 = lr\* | 0.026 | 0.85 | 0.03 | 0.456 |
| SFT | full | ×1/3 | 0.073 | 0.85 | 0.09 | 0.30(格式崩) |
| SFT | frame / exact_iso(修复版) | lr\* | 0.026 / 0.026 | 0.85 / – | 0.03 | 0.454 / 0.455 |
| SFT | spectrum / random | lr\* | 0.006 / 0.005 | 0.50 / 0.51 | 0.01 | 0.474 / 0.475 |
| SFT | spectrum / random | lr\*, s_rel 10 | 0.019 / 0.016 | 0.68 / 0.67 | 0.03 | 0.468 / 0.462 |
| SFT | spectrum / random | lr×1 s2 | 0.056 / 0.053 | 0.67 / 0.69 | 0.08 | 0.473 / 0.473 |
| RLVR | full | ×1/30 | 0.0004 | – | – | 0.477 |
| RLVR | full | lr\* | 0.002 | 0.60 | 0.003 | 0.472 |
| RLVR | full | ×1/3 | 0.009–0.012 | – | – | 0.462 |
| RLVR | frame / exact_iso(旧,作废) | lr\* | 0.002 / 0.002 | 0.45 / 0.60 | 0.004 | 0.475 / 0.468 |
| RLVR | spectrum / random | lr\* | 0.0001 / 0.0000 | 0.005 / 0.004 | 0.01 | 0.477 / 0.477 |
| RLVR | spectrum / random | lr×1 s2 | 0.003 / 0.001 | 0.12 / 0.17 | 0.02 | 0.474 / 0.478 |

**单步(8 步捕获,等范数)任务外 KL / ‖H‖²**:SFT none 0.023、frame 0.023、exact_iso 0.021、spectrum 0.0018、random 0.00025;RLVR none 0.020、frame 0.028、exact_iso 0.012、spectrum 0.0005、random 0.0003。任务 KL / ‖H‖²(E-v3-0c 扩展):SFT none 8.4、RLVR none 9.3。

**读法(A,2026-09-17)**

1. **RLVR 在同样的任务学习量下,任务外输出的偏离比 SFT 小一个数量级**:lr\* 上 0.002 vs 0.026,任务 KL 相当(0.60 vs 0.85),GSM8K 还更高(0.646 vs 0.415)。这是 RL's Razor(2509.04259)的现象在逐 run 层面的复现。
2. **单步的"选择性"两种范式一样,累计后差 10 倍。** 单步每单位范数的任务 KL(SFT 8.4 vs RLVR 9.3)与任务外 KL(0.023 vs 0.020)在两种范式下几乎相同;但 300 步后 SFT 的任务外累计 KL 是 RLVR 的 13 倍,任务累计 KL 只是 1.4 倍。**RLVR 的优势不在单步更新的方向,而在多步如何叠加**:RLVR 的步子在任务外方向上互相抵消,SFT 的在任务外方向上持续累积。这与"Retaining by Doing"(2510.18874,on-policy 数据决定保持)一致,并给出了它的机理形式。→ v4 §2.1-3 的"每步功能步长匹配"实验按原设计没有意义(单步本来就匹配);改为**任务外 KL 随步数的轨迹**与**逐矩阵归因**(见 TASKS_A_v4 A3 修订)。
3. **方向臂**:等范数 r 维更新的单步任务外 KL 比 dense 小 15–100 倍,累计小 5 倍(SFT)到 20 倍(RLVR);保谱 / 去谱与 dense 完全相同(单步 0.023 vs 0.023,累计 0.026 vs 0.026)。**"保谱少遗忘"在任务外 KL 口径上也不成立**;r 维臂的任务外 KL 小只是因为它的功能步长整体小(任务 KL 也小)。
4. **MMLU(1000 题)分辨不了健康区的遗忘差异。** 健康区 MMLU 在 0.45–0.48 之间,差 0.01–0.03,而 1000 题的抽样 SE 是 0.016;任务外 KL 差 10 倍(RLVR 0.002 vs SFT 0.026)对应的 MMLU 差只有 0.016。反例:SFT r 维 lr×1 任务外 KL 0.055 但 MMLU 0.473,dense lr×1/3 任务外 KL 0.073 而 MMLU 0.30(格式崩)。**结论:v4 的遗忘读数必须换成 (a) 任务外 KL 本身 + (b) 全量 MMLU(14042 题,SE 0.004,单个 ckpt 约 12 分钟)**,只用 1000 题的终点 MMLU 不能裁决 §2.1。全量 MMLU 已对全部 v3 终点 ckpt 提交(E-v4-2)。

### E-v4-3 逐矩阵归因(A,jobs 348072/349179;`analysis_v4/matrix_attrib.py`:终点 ckpt 逐个矩阵回退到 base,测任务外 KL 与任务 KL 下降的份额;`results/v4/result_A/matrix_attrib/`)

| ckpt | 任务外 KL | 任务 KL | 份额之和(任务外 / 任务) | 承担一半任务外份额的矩阵数 | 类型排序(任务外份额) |
|---|---|---|---|---|---|
| SFT dense lr\* s0 | 0.026 | 0.90 | 1.80 / 0.51 | 30 / 186 | down_proj 0.55 > up_proj 0.34 > out_proj 0.31 > o_proj 0.22 > in_proj_qkv 0.16 > gate_proj 0.09 |
| RLVR dense lr\* s0 | 0.002 | 0.63 | 1.92 / 0.70 | 27 / 186 | down_proj 0.46 > out_proj 0.36 > o_proj 0.34 > up_proj 0.31 > in_proj_qkv 0.19 > gate_proj 0.09 |

- **同一批矩阵在两种范式下承担任务外变化**:逐矩阵份额的秩相关 0.93,前 30 名重合 25 个;最大单个是 L23 的注意力输出投影 / down_proj(SFT 12.7%,RLVR 15.6%)。→ RLVR 不是"保护了某些矩阵",而是同样的矩阵动得少 13 倍。承重位置由结构决定,不由范式决定;**"按矩阵定位 SFT 多动的地方"这条方法假设不成立**。
- **任务承重 / 任务外承重的比值按类型稳定**,两种范式一致:gate_proj 最高(SFT 0.49,RLVR 0.88),up/down_proj 中(0.3–0.5),注意力与线性注意力的输出投影最低(o_proj 0.13 / 0.18,out_proj 0.19 / 0.22)。→ 这是一个可用的结构先验:输出投影上的更新"改任务外行为多、对任务贡献少"。**方法假设改为:按类型/矩阵的任务承重比给学习率倍率(输出投影压低、gate/up 抬高,总步长不变),看能否在同样任务分数下减少任务外 KL 与 MMLU 损失**(E-v4-4,`--lr-scale-file`)。
- 任务外份额之和 1.8–1.9(超可加)、任务份额之和 0.5–0.7(次可加):任务外偏离是矩阵之间相互放大的,任务学习是冗余的。这一点本身支持"少动几个关键矩阵就能大幅减少任务外偏离"。
- **补 3 个 ckpt(SFT lr×1/30、lr×1/3、SFT spectrum_matched lr\*)**:dense 的任务外承重分布对学习率与范式都不变——SFT 三档之间秩相关 0.97–0.99,SFT 与 RLVR 0.93–0.94,前 30 重合 25–27;类型比值同样稳定(o_proj 0.08–0.18,out_proj 0.17–0.22,gate 0.49–0.88)。**这是模型结构的属性,不是训练配方的属性**,所以按类型的倍率可以跨 lr、跨范式用同一份。r 维臂(spectrum_matched)的分布不同(与 dense 秩相关 0.6):任务外承重更集中(11 个矩阵占一半),份额之和 0.74 / 0.76(两边都近似可加,不再超可加),并且每种类型的任务承重比都高于 dense(up_proj 2.1、gate 1.6、out_proj 1.3 vs dense 0.4 / 0.5 / 0.2)——r 维更新"更挑",但总量太小。
