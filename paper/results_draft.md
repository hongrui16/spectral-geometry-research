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

### 待填(作者B v2 交付)
- P0 三个回答:commit hash ______;H 自检 ______;秒/步 ______
- P1:OPD none MMLU ______;RLVR none MMLU ______;RLVR frame_only MMLU ______
- P2:v1 24 个 run 的 v2 指标(H 自检结论)______
- P3 等范数 H5 主表(GSM8K / MMLU,2 seeds):RLVR full/frame_m/spec_m/rand ______;SFT ______
- P4 exact_iso:RLVR ______;SFT ______
- P5 4B RLVR AdamW:R_enrich ______,GSM8K ______
