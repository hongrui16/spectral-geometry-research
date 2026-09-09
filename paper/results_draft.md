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
- 3B OPD 在跑:______

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
