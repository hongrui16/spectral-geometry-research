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

### E-v3-1 dense lr 扫描 —— RLVR 六个(B)—— 待填;RLVR lr\* = ______

