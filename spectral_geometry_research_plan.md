# Gradient Spectral Geometry across SFT / OPD / RLVR

Working title: **When Should LLM Training Change the Spectrum? Gradient-Level Spectral Geometry of SFT, On-Policy Distillation, and RLVR**

---

## 0. 一句话定位

在**同一 checkpoint、同一 prompt 集、同一 token budget** 下，逐步测量 SFT / OPD / RLVR 三种训练范式的 raw gradient \(G_t\) 和 optimizer update \(H_t\) 在当前权重 \(W_t\) 的奇异基下的分布，回答三个现有工作没回答的问题：

1. RLVR 的等谱（isospectral）倾向来自 objective 本身，还是 optimizer 后处理？
2. 为什么 policy gradient 会产生 frame-dominant 的更新？
3. Gradient 各 singular mode 的信噪比（SNR）能否替代幅值，作为 optimizer 该放大 / 抑制哪些方向的依据？

前两个问题是 analysis，第三个是 optimizer 推论。Phase 1 + Phase 2 已构成一篇完整的 analysis paper；Phase 3 可选。

---

## 1. 背景与已有工作

### 1.1 三种训练范式

| | SFT | OPD | RLVR |
|---|---|---|---|
| 数据来源 | 外部固定数据 | student 自己采样 | student 自己采样 |
| 监督信号 | 每 token 一个 target | 每 token 一个 teacher 分布 | 每序列一个标量 reward |
| 信号密度 | 稠密 | 稠密 | 极稀疏 |
| 是否 on-policy | 否 | 是 | 是 |

**RLVR**：可验证 reward（数学答案、代码测试），GRPO 类算法，组内均值做 baseline：

$$
G_{\rm RLVR} = \mathbb E\big[A(x,y)\,\nabla_W \log\pi(y|x)\big],\qquad A_k = \frac{r_k-\bar r}{\mathrm{std}(r)}
$$

监督是 sequence-level 标量，摊到每个 token。

**OPD**：student rollout，teacher 给 token-level 分布，reverse KL：

$$
G_{\rm OPD} = \mathbb E_{y\sim\pi_s}\Big[\sum_t \nabla_W\log\pi_s(y_t|\cdot)\,\big(\log\pi_s(y_t|\cdot)-\log\pi_t(y_t|\cdot)\big)\Big]
$$

监督是每 token 稠密、有方向。

OPD 卡在中间：与 SFT 同为稠密监督但 on-policy；与 RLVR 同为 on-policy 但监督稠密。因此可以分离 **on-policy 采样** 和 **信号密度** 两个因素。

### 1.2 已有观察（按范式）

**RLVR（ISO 一线）**
- Weights：\(\sigma(W_T)\approx\sigma(W_0)\)，谱漂移极小；\(\Delta W\) 落在 off-diagonal（frame）方向；固定谱、只训 \(U,V\) 即可学到能力。
- Gradients：Pion 报告 raw PG 梯度谱尾部 SNR 低，Muon 全谱拉平后放大噪声。
- 另有 RLVR update sparsity / 低 stable rank 的观察。

**OPD（两篇 OPD geometry 工作）**
- Weights：\(\Delta W\) 避开 \(W_0\) 主子空间；numerically full-rank 但 spectrally concentrated；FFN-heavy；早期几十步锁定后续有效子空间（subspace locking）。
- Gradients：无逐步分析，主要是 checkpoint 级。

**SFT**
- 作为 baseline 出现在上述工作里，谱改动大于 RLVR，主子空间干扰更强。
- LoRA / GaLore 一线有"梯度低秩"观察，但未与 OPD / RLVR 对照。

**Optimizer（Muon / Pion / Spectra）**
- Muon：\(\sigma_i(G)\to 1\)，全谱 flatten。
- Pion：保 dominant modes、压 tail，隐含假设"大奇异值 = 信号，小 = 噪声"。
- Spectra：区分 dominant spikes 与 tail，pretraining 加速。

**NaNA（SVD as Fast Interpretability, ICML 2026）**
- 对 MLP \(W_{\rm up}, W_{\rm down}\) 做 SVD，每个 rank-1 项 \(\sigma_i u_i v_i^\top\) 视为 Detector-Effector Unit。
- 对 (prompt \(x\), target \(t^*\)) 定义 mode 贡献 \(c_i=(\sigma_i v_i^\top x)(t^{*\top}u_i)\)，按 \(c_i\) 选 top-k（SCA）。
- GPT-2 上 top-20 DEU 恢复 97–99% top-1；ablate 掉则崩溃。
- 高贡献 mode 集中在低 index，但 tail 仍有非平凡语义贡献。
- 它只做 static 权重、只做 MLP、只做推理；对我们的价值是**给 \(W\) 的每个 singular mode 一个 task-level 功能标签**。

### 1.3 空缺

- 三种范式从没在同一 checkpoint、同一数据、同一 budget 下被逐步对比过；现有对比设置互不兼容。
- 几乎全部是 \(W_0\to W_T\) 的 endpoint 分析，\(G_t\) 与 \(H_t\) 的谱从未被分开看。
- 没有人解释为什么 reward-driven gradient 的对角分量小。
- "signal vs noise" 按幅值区分的假设未被检验。

### 1.4 Weights 分析 vs Gradients 分析

**Weights 分析（看结果）**：对象是 \(W_0, W_T, \Delta W\)。能回答谱变了多少、\(\Delta W\) 的 rank、落在哪个子空间。局限：累积量抹掉了中间过程；分不清 objective 效应和 optimizer 效应；分不清"无信号"和"信号被噪声平均掉"。

**Gradients 分析（看过程）**：对象是每步 \(G_t\)（objective 直接给出的方向）和 \(H_t\)（optimizer 处理后实际加到权重上的量）。能回答：objective 的性质（与 optimizer 无关）；optimizer 效应（\(H_t-G_t\)）；noise 结构（多 batch 估 mode-wise SNR）；时间动态。局限：开销大；单步噪声大；基随 step 漂移需对齐。

本文三个核心问题全在 gradients 侧。Weights 侧的累积量作为 sanity check。

---

## 2. 核心形式化

对每个二维权重矩阵，每步做 SVD：

$$
W_t = U_t\Sigma_tV_t^\top
$$

把 gradient（或 update）投到当前奇异基：

$$
C_t = U_t^\top G_t V_t
$$

- 对角 \(D_t=\operatorname{diag}(C_t)\)：一阶奇异值变化，\(\dot\sigma_i=(C_t)_{ii}\)。**Spectral motion**。
- 非对角 \(O_t=C_t-D_t\)：奇异向量旋转。**Frame motion**。

$$
\Delta W_t \approx \underbrace{U_tD_tV_t^\top}_{\text{spectral}} + \underbrace{U_tO_tV_t^\top}_{\text{frame}} + \text{residual}
$$

工作假设：随着 pretrained representation 越成熟，优化的角色从 "construct spectrum" 转向 "reorient spectrum"：

$$
R_{\rm spectrum}:\ \text{SFT} > \text{OPD} > \text{RLVR}
$$

---

## 3. 实验 Setup

固定以下项，否则跨范式对比无意义：

- **Base model**：Qwen3-1.7B 或 Llama-3.2-3B 起步；验证后上 7–8B 一个。
- **Prompt 集**：数学（GSM8K / MATH 子集），三种范式共用同一批 prompt。
  - SFT：prompt + 标准解
  - OPD：student rollout + 同 family 大模型（Qwen3-8B / 32B）token-level 分布，reverse KL
  - RLVR：GRPO，correctness reward，每 prompt \(K=8\) 条 rollout
- **Token budget**：三者相同。
- **Optimizer**：AdamW 为主；Muon 作为第二 optimizer 对 SFT 和 RLVR 各跑一遍。
- **训练长度**：几百步。目标是逐步测量，不是训到收敛。
- **保存**：每 10 步存 \(G_t\)、\(H_t\)、\(W_t\)，只存约 16 个代表性矩阵：
  - 早 / 中 / 晚各取若干层
  - 每层：\(W_Q, W_K, W_V, W_O, W_{\rm gate}, W_{\rm up}, W_{\rm down}\) 中挑一组
- **多 batch 采样**：每个保存 step 额外用 4–8 组不同 minibatch / rollout 组计算 \(G^{(b)}\)，用于 SNR 估计。

---

## 4. 三类指标

### 4.1 几何指标（gradient 相对权重谱）

对 \(G_t\) 和 \(H_t\) 各算一遍，差值即 optimizer 效应。

1. **Spectral-motion ratio**
   $$R_{\rm spectrum}(G_t)=\frac{\|\operatorname{diag}(U_t^\top G_tV_t)\|_F^2}{\|G_t\|_F^2},\qquad R_{\rm frame}=1-R_{\rm spectrum}$$

2. **子空间能量分布**：\(G_t\) 的能量在 \(W_t\) 的 top-k / mid / tail 奇异子空间的占比。

3. **主向量对齐**：\(G_t\) 的 top singular vectors 与 \(W_t\) 的 top / mid / tail 子空间的 principal angles。

4. **\(G_t\) 自身谱**：stable rank \(\|G\|_F^2/\|G\|_2^2\)，top-k 能量占比，谱衰减速率。

5. **累积量（sanity check）**：\(\|\sigma(W_t)-\sigma(W_0)\|\)；\(U_t\) vs \(U_0\)、\(V_t\) vs \(V_0\) 的子空间旋转。验证逐步指标能预测 endpoint 结果。

### 4.2 统计指标（mode-wise SNR）

同一 step 的 \(B\) 组 gradient \(G^{(1)},\dots,G^{(B)}\)，对 \(W_t\) 的每个 mode \(i\)：

$$
\mathrm{SNR}_i=\frac{\big(\mathbb E_b[u_i^\top G^{(b)}v_i]\big)^2}{\mathrm{Var}_b[u_i^\top G^{(b)}v_i]}
$$

对 off-diagonal 块 \((i,j)\) 做同样的量，可按 \((i,j)\) 所属谱区（top-top / top-tail / tail-tail）聚合。

预期：RLVR 的 SNR 在少数 mode 上高、尾部极低（scalar reward + 零均值 advantage 导致大量抵消）；OPD 因稠密 teacher 信号，tail 也有一致方向；SFT 介于两者或与 OPD 接近。

### 4.3 功能指标（借 NaNA 的 SCA）

只对 MLP \(W_{\rm up}, W_{\rm down}\)。

- 在 held-out prompt 集上，对每个 (prompt, target) 算 \(c_i=(\sigma_i v_i^\top x)(t^{*\top}u_i)\)。
- 聚合：\(|c_i|\) 的均值，或 top-k 出现频率。
- RLVR 无单一 target token：用 rollout 中高 advantage 位置的 token。
- 把 \(W_t\) 的 mode 分成 **high-contribution** / **low-contribution** 两桶，看三种范式的 \(G_t\) 能量分别落在哪桶、以及 off-diagonal。

意义：几何指标回答"更新落在权重谱的哪里"，功能指标回答"落的那些 mode 是否重要"。这把 "signal 应按 utility 而非 magnitude 判断" 变成可测的量。

候选假设：RLVR 的更新避开高 \(c_i\) 的 mode（不破坏已有能力，对应 ISO 的 spectrum preservation）；SFT 直接改它们（对应 forgetting）。

---

## 5. 执行阶段

### Phase 1：观察（2–3 周）

跑完三种范式，出三张主图：

1. \(R_{\rm spectrum}(G_t)\) 随 step，三条线（+ 按 layer type 分面）
2. \(R_{\rm spectrum}(G_t)\) vs \(R_{\rm spectrum}(H_t)\)，AdamW 和 Muon 各一组 → 回答等谱倾向是 objective 还是 optimizer 的
3. mode-wise SNR 曲线，三种范式

**决策点**：若三种范式几何上无差异，主线改为 optimizer 效应；若有差异，进入 Phase 2。

Phase 1 只回答三个问题：
- \(R_{\rm spectrum}(G)\) 是否满足 SFT > OPD > RLVR？
- AdamW / Muon 是否改变这个 ordering？
- Layer type（attention vs FFN）差异是否比 objective 差异更大？

### Phase 2：因果干预（2 周）

把 \(H_t\) 投到受约束的子空间后再更新权重，四种约束 × 三种范式：

| 约束 | 定义 |
|---|---|
| Full | \(W\leftarrow W+H\) |
| Spectrum-only | \(H_\Sigma=U\operatorname{diag}(U^\top HV)V^\top\)，只允许 \(\Sigma\) 变 |
| Frame-only | \(H_{\rm frame}=H-H_\Sigma\)，或严格 isospectral manifold update |
| High-contribution-only | 只保留 SCA 高贡献 mode 上的分量 |
| Low-contribution-only | 只保留低贡献 mode 上的分量 |

比较性能损失。理想 pattern：

| | spectrum-only | frame-only |
|---|---|---|
| SFT | 有效 | 明显受损 |
| OPD | 部分够用 | 部分够用 |
| RLVR | 很差 | 接近 full |

若出现，claim 从 "RLVR 奇异值变化小" 升级为 "不同训练范式需要 fundamentally different parameter motion"。这个组合（统一设置下三范式 + 四约束）是现有工作没做过的。

### Phase 3：Optimizer 推论（可选，2–3 周）

1. 用 Phase 1 的 mode-wise SNR 预测 Muon 在哪些 layer / 哪种范式上会退化，与实际 Muon 训练结果对照。
2. 若时间够，实现 SNR-aware spectral optimizer：

$$
H=U\operatorname{diag}\big(f(\mathrm{SNR}_1),\dots,f(\mathrm{SNR}_r)\big)V^\top
$$

对比 Muon（\(\sigma_i\to1\)）、Pion（按幅值高通）。不必 outperform，展示"按 SNR 而非幅值决定放大 / 抑制"的解释力即可。

### 理论侧（与 Phase 1 并行）

从 \(G=\mathbb E[A\,\nabla_W\log\pi]\) 出发，分析 \(u_i^\top Gv_i\) 在什么条件下期望趋于 0 而 off-diagonal 不会。可能用到的性质：advantage 零均值、group-relative normalization、score-function 零均值、KL 正则、pretrained 权重的 stationarity。目标是给 "RLVR 偏 frame motion" 一个机制解释，而不只是观察。

---

## 6. 工程要点

- **SVD 开销**：4096×11008 全 SVD 每步太贵。用 randomized SVD 取 top-256，剩余作为 tail 块；或只在保存的 step 上离线算。
- **Mode 跨 step 对齐**：SVD 基每步漂移，DEU index 不稳定。用 principal angle / Procrustes 跟踪子空间，不按 index 对应。
- **SNR 估计**：RLVR gradient 方差大，至少 8 组 rollout，这是主要算力开销。
- **SCA 聚合**：per-target-token 量要在 batch 上聚合；attention 矩阵无 detector/effector 解释，只用几何指标。
- **SVD 符号歧义**：\(\pm u_i,\pm v_i\)，对 \(u_i^\top Gv_i\) 无影响（符号同时翻转），但功能标签需两个方向都看。

---

## 7. 预期贡献

1. 首个在统一设置下对 SFT / OPD / RLVR 的 **gradient-level** 谱几何对比。
2. 分离 objective 效应与 optimizer 效应（\(G_t\) vs \(H_t\)）。
3. Mode-wise SNR 作为新的信号 / 噪声判据，与幅值判据对照。
4. 功能坐标（SCA）+ 几何坐标的联合分析，把 forgetting 与 spectral motion 的关系变成可测量。
5. 因果干预：spectrum-only / frame-only / contribution-bucket 约束下的三范式对比。
6. （可选）SNR-aware spectral optimizer 与 Muon / Pion 对比。

---

## 8. 需要确定的事项

1. Base model 与算力预算（1.7B 还是 3B 起）。
2. OPD teacher 来源（同 family 大模型 vs 外部）。
3. 是否做 Phase 3，决定 venue 与时间线。
4. 定位：纯 analysis paper（ICLR / NeurIPS 分析类）vs 带 optimizer 结果。

---

## 附 A：讨论演进（从最初的问题到当前方案）

### A.1 起点

问题：policy gradient 对参数 \(W\) 的更新有什么性质，是否有工作研究过其对奇异值的影响。

回答：有，而且几条线正在汇合——RLVR 的 spectral inheritance（ISO）、OPD 的 off-principal / subspace locking、Muon 对 gradient spectrum 的重塑、pretraining 里的 gradient spectral anisotropy。当时的判断是：真正值得做的不是再单独证明某种方法 update 是 low-rank / sparse / 谱漂移小，而是建立更统一的 LLM optimization geometry。

### A.2 最初列出的 10 个方向

1. **Objective × Optimizer 几何分解**：固定同一批数据 / rollout，比较 SFT、OPD、RLVR 的 raw gradient，再分别过 SGD / AdamW / Muon，测最终 update 在权重奇异基下的 diagonal / off-diagonal 能量。核心问题：isospectral tendency 由 objective 决定，还是 optimizer 制造 / 破坏的。
2. **Gradient spectrum 里 signal vs noise 的分布**：Muon 把所有 active singular directions 拉到同一尺度，若 RLVR 的 tail modes 主要是 reward noise，会放大坏方向。可做 noise-aware optimizer：只 flatten 判定为 signal 的谱段，对 noise tail shrink / truncate。
3. **Weight spectrum × gradient spectrum 的 alignment**：gradient 的 top singular vectors 与 \(W\) 的 top / mid / tail 子空间对齐多少。真正有用的 optimizer 可能不是 flatten gradient spectrum，而是根据它相对 weight geometry 的位置做处理。
4. **动态 spectral optimizer**：不同训练阶段、layer、objective 下对 spectrum 做不同操作——早期 suppress spikes，中期 preserve anisotropy，post-training 切到 frame-dominant update。
5. **Pretraining 与 post-training 的 geometry 是否 fundamentally 不同**：pretraining 需要"创造 representation"，可能需要重写 spectrum；RLVR 更像"重定向已有能力"，固定 spectrum 也能学；OPD 介于两者之间。
6. **Layer-wise heterogeneous optimization**：Q/K/V/O 与 gate/up/down 的谱可能完全不同（OPD 已观察到 FFN-heavy），按 layer/type 自动选 AdamW / Muon / ISO-like / spectral clipping。
7. **Subspace discovery 后锁定以省 compute / memory**：若训练早期几十步就能识别有效 update subspace，后续只维护该子空间的 optimizer states。
8. **LoRA / PEFT 与真实 gradient geometry 是否匹配**：真实 update 可能 "numerically full rank, spectrally concentrated"，fixed-rank LoRA 未必是最自然的压缩；动态 rank、spectral energy threshold、或直接在发现的子空间上训练。
9. **Forgetting / plasticity 与 spectral motion 的关系**：改变 pretrained top singular modes 是否更易导致 catastrophic forgetting，off-principal / frame update 是否更能"加能力不毁能力"；若成立可直接设计 regularizer 限制 destructive spectral motion。
10. **用 geometry 做 optimizer switching / routing**：在线监测 gradient SNR、stable rank、spectral concentration、principal alignment，进入不同 regime 时自动切换 update rule。

当时选出的前三：(1) Objective–Optimizer Geometry Factorization；(2) Noise-aware Spectral Optimizer；(3) Adaptive Spectrum-vs-Frame Training，即把 ISO 从固定 spectrum 推广为

$$
W=U\Sigma V^\top,\qquad \eta_U,\eta_V,\eta_\Sigma \text{ 动态可调}
$$

SFT 可能需要较大 \(\eta_\Sigma\)，OPD 中等，RLVR 接近 0，不同 layer 亦不同。收敛的题目：**When Should LLM Training Change the Spectrum?**

### A.3 v0 方案（第一版）

- **核心分解**：\(\Delta W_t=\) spectral motion + frame motion + residual；假设 pretraining / SFT 需要更多 \(\Delta\Sigma\)，RLVR 主要需要 frame rotation，OPD 居中。
- **Phase 1 测 geometry**：\(C_t=U_t^\top H_tV_t\)，对角对应 \(\dot\sigma_i\)，非对角对应 frame motion；定义 \(R_{\rm spectrum}, R_{\rm frame}\)。第一张关键图：四种范式的 spectral update %。强调同时保存 \(G_t\) 和 \(H_t\)，拆 objective effect 与 optimizer effect。
- **Phase 2 比较四种 regime**：continued pretraining / SFT / OPD / RLVR，同 architecture、相近 data domain、同 budget、同初始 checkpoint。
- **Phase 3 看 layer structure**：\(W_Q,W_K,W_V,W_O\) 与 \(W_{\rm gate},W_{\rm up},W_{\rm down}\) 分开；预期 "freeze spectrum" 不应是 model-wide binary decision。
- **Phase 4 causal intervention**：full / spectrum-only / frame-only 三组训练，四种范式对比。
- **Phase 5 optimizer**：\(H=\alpha_tH_\Sigma+\beta_tH_{\rm frame}\)，\(\alpha_t,\beta_t\) 随 layer / stage / SNR / objective / alignment 变化；最简版 \(r_t=\|\operatorname{diag}(U^\top GV)\|_F/\|G\|_F\)，\(\alpha_t=f(r_t)\)。更有野心的版本：维护 approximate top spectral subspace，把 \(G\) 分成 spectral / tangent / residual 三部分，分别用 Adam-like / Muon-like / shrink-drop。
- **6 个关键测量**：\(\|\Delta\sigma(W_t)\|\)、\(R_{\rm spectrum}(G_t)\)、\(R_{\rm spectrum}(H_t)\)、subspace rotation\((U_t,U_0)\)、stable rank\((G_t,H_t)\)、gradient SNR by singular mode。
- **统一叙事**：随 pretrained representation 越成熟，优化角色从 "construct spectrum" 转向 "reorient spectrum"。若成立可解释：LoRA 在 post-training 有效、ISO 在 RLVR 能 work、Muon 在 pretraining 与 RLVR 表现不同、OPD 出现 subspace locking、post-training 可用更强的结构约束。
- **MVP**：同一 checkpoint 只做 SFT vs OPD vs RLVR，几百步，每 5–10 步存 \(G_t,H_t,W_t\)，10–20 个 layer，先回答三个问题。

### A.4 "现有 paper 是否已覆盖" 的判断

结论：若只做到现象层（SFT vs OPD vs RLVR 的 \(\Delta\sigma\)、stable rank、subspace rotation），novelty 偏弱。ISO 覆盖 RLVR fixed-spectrum；OPD 两篇覆盖 off-principal、subspace locking、full-rank-but-concentrated 与 subspace-locking intervention；Muon / Pion 覆盖 optimizer 如何 reshape spectrum 及 RLVR 低 SNR 下 tail amplification。

仍然值得做的四条，按深度排序：

1. **因果链 \(G_t\to H_t\to\Delta\Sigma,\Delta U,\Delta V\)**：现有工作多看 \(W_0\to W_T\)，ISO 虽做了 constrained optimizer 但 claim 仍是 endpoint 级。逐步拆开三个阶段各贡献多少；尤其 "RLVR 的 isospectral tendency 存在于 raw PG 里，还是 AdamW/Muon 后处理产生的"。
2. **为什么 RLVR gradient 偏 frame motion**：ISO 发现 \(\Sigma_{\rm base}\) 够用但没解释。从 \(G=\mathbb E[A\nabla_W\log\pi]\) 出发研究 \(u_i^\top Gv_i\) 为什么小，能否在某些条件下证明 \(\mathbb E[u_i^\top Gv_i]\approx0\) 而 \(\mathbb E\|P_{\rm tangent}G\|^2\) 仍大；联系 advantage centering、group-relative normalization、score-function 零均值、KL 正则、pretrained stationarity。
3. **Signal / noise 按 task utility 而非幅值区分**：Pion 隐含 "大奇异值 ≈ 信号"，post-training 里新能力可能恰在小但一致的方向。用 repeated rollout batches 估 mode-wise consistency / SNR，optimizer 按 \(\sigma_i(G)\to f(\mathrm{SNR}_i)\) 处理。自然解释：pretraining 稠密监督 SNR 普遍高所以 Muon 有效；OPD 稠密 teacher 信号多数 mode 有 coherent signal；RLVR 标量 reward + 采样，tail SNR 低所以 Muon whitening 易炸。
4. **Weight-space geometry ↔ function-space geometry**：网络有巨大 reparameterization symmetry，weight frame rotation 未必对应 representation 的 "rotation"。真正的问题是 weight spectral motion 与 activation / Jacobian / Fisher geometry 的对应：一个 update \(H\) 改变 policy 的程度更接近 \(\Delta f\approx J_\theta H\) 而非 \(\|H\|\)。若 isospectral tangent directions 恰好包含 RLVR 的 high-Fisher / high-return 方向，ISO 就从 empirical trick 升为 optimization principle。

当时推荐把 2 和 3 结合：先测清 RLVR 各谱模态的 SNR 与功能贡献，再设计 SNR-aware 矩阵 optimizer。

### A.5 从 A.4 到当前方案的收敛

- 主线定为 gradient 侧跨范式对比（回应 A.4 第 1 条），SNR 指标（第 3 条）和理论推导（第 2 条）并行。
- NaNA 的 SCA 引入为功能坐标，把 "utility 而非幅值" 变成可测量；定位为额外坐标，不是主线。
- A.4 第 4 条（Fisher / function-space）暂不进入主线，作为后续方向。
- A.3 中的 continued pretraining 从 MVP 中去掉，先做 SFT / OPD / RLVR 三者。
- A.3 Phase 5 的 optimizer 设计降为可选 Phase 3。

## 附 B：已被"吃掉"、不再作为主线的方向

- Checkpoint 级 SFT vs OPD vs RLVR 的 \(\Delta\sigma\)、stable rank、子空间旋转对比（ISO、OPD 两篇已覆盖）。
- 单独证明某种方法的 update 是 low-rank / sparse / 谱漂移小。
- 直接设计新 optimizer 而不先做机制分析。
