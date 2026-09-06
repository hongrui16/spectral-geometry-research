# When Should LLM Training Change the Spectrum?
## Credit Assignment Determines the Spectral–Frame Geometry of Gradients — 统一文档

结构：
- **Part I 分析**：理论（含证明）+ 实验协议
- **Part II 方法**：由 Part I 的发现触发的优化方法
- **Part III 实验与时间线**

---

# 0. 核心主张

给定 pretrained 权重 \(W_0\in\mathbb R^{m\times n}\)，post-training objective 产生梯度 \(G_t\)，optimizer 给出 update \(H_t\)，\(W_{t+1}=W_t+H_t\)，\(W_t=U_t\Sigma_tV_t^\top\)。

三个可证伪主张：

- **C1（Objective-determined）**：spectral-vs-frame 倾向在 raw gradient \(G_t\) 中已存在，optimizer 只是调制。
- **C2（Credit-assignment mechanism）**：RLVR 的 frame-dominance 来自 sequence-level 标量 credit 与 score-function 零均值的组合，可写成 covariance 并给出机制。
- **C3（SNR, not magnitude）**：gradient 各 singular mode 的幅值混合了"与 reward 的相关性"和"该 mode 统计量的方差"，后者与 signal 无关；按 SNR 选方向优于按幅值。

---

# 1. 背景

## 1.1 三种范式

| | SFT | OPD | RLVR |
|---|---|---|---|
| 数据 | 外部固定 | student 采样 | student 采样 |
| 监督 | 每 token 一个 target | 每 token 一个 teacher 分布 | 每序列一个标量 |
| 密度 | 稠密 | 稠密 | 极稀疏 |
| on-policy | 否 | 是 | 是 |

OPD 与 SFT 同为稠密但 on-policy；与 RLVR 同为 on-policy 但稠密。它分离了"on-policy 采样"和"信号密度"两个因素。

## 1.2 已有工作与空缺

- **ISO**（RLVR）：\(\sigma(W_T)\approx\sigma(W_0)\)，固定谱只训 \(U,V\) 即可。未解释原因。
- **OPD geometry 两篇**：\(\Delta W\) 避开主子空间、numerically full-rank 但 spectrally concentrated、FFN-heavy、早期 subspace locking。全是 checkpoint 级。
- **Muon / Pion / Spectra**：Muon 全谱拉平；Pion 按幅值保头压尾并报告 RLVR 下 Muon 放大 tail noise；隐含"大奇异值 = 信号"。
- **NaNA**（ICML 2026）：MLP 权重 SVD 的 rank-1 项为 Detector-Effector Unit，贡献 \(c_i=(\sigma_iv_i^\top x)(t^{*\top}u_i)\)；top-20 恢复 97–99%。提供 mode 的功能标签。

空缺：三范式从未在同一 checkpoint / 数据 / budget 下逐步对比；\(G_t\) 与 \(H_t\) 从未分开看；无人解释 PG 对角分量为何小；"幅值 = 信号"未被检验。

## 1.3 Weights 分析 vs Gradients 分析

Weights 分析看 \(W_0\to W_T\)：累积量抹掉过程，分不清 objective 与 optimizer 效应，分不清"无信号"与"信号被噪声平均"。Gradients 分析看每步 \(G_t,H_t\)：能分离 objective 性质、optimizer 效应、noise 结构、时间动态。本文三个主张全在 gradients 侧；weights 侧累积量作 sanity check。

---

# Part I — 理论

# 2. 统一梯度形式

## 2.1 所有 objective 的梯度都是 \(\sum_t\delta_tx_t^\top\)

对线性层 \(y_t=Wx_t\)，任意 loss：
$$
G=\sum_t\delta_tx_t^\top,\qquad\delta_t:=\partial\mathcal L/\partial y_t
$$
objective 只通过 \(\delta_t\) 进入。记 \(g_t^{(\tau)}:=\partial\log\pi(y_\tau|y_{<\tau},x)/\partial y_t\)（\(\tau\ge t\) 时非零）。

| Objective | \(\delta_t\) | credit 结构 |
|---|---|---|
| SFT | \(-\sum_{\tau\ge t}g_t^{(\tau)}\) | 每位置 credit \(=1\)，target 固定 |
| OPD (reverse KL) | \(\sum_{\tau\ge t}\big[w_\tau g_t^{(\tau)}+\partial_{y_t}\mathrm{KL}_\tau\big]\)，\(w_\tau=\log\pi_s(y_\tau)-\log\pi_t(y_\tau)\) | 每位置稠密、有符号 |
| RLVR (GRPO) | \(A\sum_{\tau\ge t}g_t^{(\tau)}\) | 整条序列一个标量 |

统一：
$$
G=\mathbb E\Big[\sum_t\sum_{\tau\ge t}c_\tau\,g_t^{(\tau)}x_t^\top\Big]+(\text{OPD 直接项})
$$
三范式唯一结构差别是 credit 向量 \(c\)：SFT 常数 1，OPD 稠密 teacher 信号，RLVR \(c_\tau\equiv A\)。**"训练范式"从离散变量变为"credit 的密度与方差"这个连续变量。**

## 2.2 投到权重奇异基

$$
a_{i,t}:=v_i^\top x_t,\qquad e_{i,t}:=u_i^\top\delta_t,\qquad C:=U^\top GV,\qquad C_{ij}=\sum_te_{i,t}a_{j,t}
$$
对角 \(C_{ii}\)：同一 mode 的 detector 激活与 effector 误差的相关（改 gain）。非对角 \(C_{ij}\)：把输入模式 \(v_j\) 路由到输出方向 \(u_i\) 的压力（改 routing）。

---

# 3. Spectral–Frame 分解

**Proposition 1（一阶谱动力学）.** 设 \(W=U\Sigma V^\top\) 奇异值互异，update \(H\)，\(C=U^\top HV\)。则
$$
\dot\sigma_i=C_{ii},\qquad
\dot u_i=\sum_{j\ne i}\frac{\sigma_iC_{ji}+\sigma_jC_{ij}}{\sigma_i^2-\sigma_j^2}u_j,\qquad
\dot v_i=\sum_{j\ne i}\frac{\sigma_iC_{ij}+\sigma_jC_{ji}}{\sigma_i^2-\sigma_j^2}v_j
$$

*证明.* 由 \(Wv_i=\sigma_iu_i\) 和 \(W^\top u_i=\sigma_iv_i\) 对 \(W\to W+\epsilon H\) 微分：
$$
Hv_i+W\dot v_i=\dot\sigma_iu_i+\sigma_i\dot u_i,\qquad H^\top u_i+W^\top\dot u_i=\dot\sigma_iv_i+\sigma_i\dot v_i.
$$
第一式左乘 \(u_i^\top\)：\(C_{ii}+\sigma_iv_i^\top\dot v_i=\dot\sigma_i+\sigma_iu_i^\top\dot u_i\)。单位范数给 \(u_i^\top\dot u_i=v_i^\top\dot v_i=0\)，故 \(\dot\sigma_i=C_{ii}\)。
第一式左乘 \(u_j^\top\)（\(j\ne i\)），记 \(a=u_j^\top\dot u_i,\ b=v_j^\top\dot v_i\)：\(C_{ji}+\sigma_jb=\sigma_ia\)。第二式左乘 \(v_j^\top\)：\(C_{ij}+\sigma_ja=\sigma_ib\)。联立解得
$$
a=\frac{\sigma_iC_{ji}+\sigma_jC_{ij}}{\sigma_i^2-\sigma_j^2},\qquad b=\frac{\sigma_iC_{ij}+\sigma_jC_{ji}}{\sigma_i^2-\sigma_j^2}.\qquad\blacksquare
$$

**推论 1.1.** Spectral motion 能量 \(=\|\operatorname{diag}C\|^2\)；frame motion 由非对角驱动，被 \(1/(\sigma_i^2-\sigma_j^2)\) 放大。定义
$$
R_{\rm spectrum}(G):=\frac{\|\operatorname{diag}(U^\top GV)\|_F^2}{\|G\|_F^2}
$$
它是 objective 施加的**压力**的分解；实际旋转量还取决于谱间隙，尾部近简并 mode 的旋转被放大。这解释 OPD 观察到的 "numerically full-rank"：即使 \(C\) 集中，尾部近简并使旋转扩散。

**Proposition 2（spectrum-only update 的表达力）.** 若 \(H_\Sigma=U\operatorname{diag}(C)V^\top\)，则对任意 \(x\)
$$
\Delta(Wx)=\sum_iC_{ii}(v_i^\top x)u_i.
$$
*证明.* \(H_\Sigma x=\sum_iC_{ii}u_i(v_i^\top x)\)。\(\blacksquare\)

输出变化只沿已有 effector 方向、以已有 detector 激活为系数。Spectrum-only update **不能建立新的输入→输出关联**，所在参数子空间 \(r\) 维（vs \(mn\)）。

**推论 2.1（可检验）.** 需要新关联的任务在 spectrum-only 下必然失败；只需重新加权已有关联的任务可以成功。

**Proposition 3（Optimizer 等变性）.** Muon 的 update \(H=\mathrm{polar}(G)=G(G^\top G)^{-1/2}\) 满足：对任意正交 \(U',V'\)，\(\mathrm{polar}(U'^\top GV')=U'^\top\mathrm{polar}(G)V'\)。因此
$$
R_{\rm spectrum}(H_{\rm Muon})=R_{\rm spectrum}\big(\mathrm{polar}(C)\big),\quad C=U^\top GV,
$$
只依赖 \(G\) 相对 \(W\) 的对齐结构。Adam 的逐元素归一化不满足此性质，\(R_{\rm spectrum}(H_{\rm Adam})\) 依赖坐标基。

*证明.* \((U'^\top GV')^\top(U'^\top GV')=V'^\top G^\top GV'\)，其 \(-1/2\) 次幂为 \(V'^\top(G^\top G)^{-1/2}V'\)，代入即得。Adam 的反例：取 \(G\) 为对角矩阵与其旋转版本，逐元素 \(\mathrm{sign}\)-like 归一化给出不同的 \(R_{\rm spectrum}\)。\(\blacksquare\)

---

# 4. RLVR 的 covariance 结构

**Lemma 4（score-function 零均值）.** 对任意参数 \(\theta\) 和 \(x\)，\(\mathbb E_{y\sim\pi_\theta(\cdot|x)}[\nabla_\theta\log\pi_\theta(y|x)]=0\)。
*证明.* \(\int\pi\nabla\log\pi=\int\nabla\pi=\nabla\int\pi=\nabla1=0\)。\(\blacksquare\)

**推论 4.1.** 对任意 baseline \(b(x)\)：
$$
G_{\rm RLVR}=\mathbb E_x\Big[\operatorname{Cov}_{y\sim\pi}\big(r(x,y),\nabla_W\log\pi(y|x)\big)\Big].
$$
*证明.* \(\mathbb E_y[(r-b)\nabla\log\pi]=\mathbb E_y[r\nabla\log\pi]-b\cdot0=\mathbb E_y[r\nabla\log\pi]-\mathbb E_y[r]\mathbb E_y[\nabla\log\pi]=\operatorname{Cov}_y(r,\nabla\log\pi)\)。\(\blacksquare\)

GRPO 的 \(A=(r-\bar r)/\mathrm{std}\) 是此 covariance 的有限样本估计。

**Proposition 5（谱分量与 frame 分量的 covariance 形式）.** 定义每条序列的
$$
S_i(y):=\frac{\partial\log\pi(y|x)}{\partial\sigma_i}=u_i^\top\nabla_W\log\pi\,v_i=\sum_te^0_{i,t}a_{i,t},
$$
$$
T^U_{ij}(y):=\frac{\partial\log\pi(y|x)}{\partial(\Theta_U)_{ij}}=\sigma_j(u_i^\top\nabla_W\log\pi\,v_j)-\sigma_i(u_j^\top\nabla_W\log\pi\,v_i),
$$
其中 \(e^0_{i,t}=u_i^\top\sum_{\tau\ge t}g_t^{(\tau)}\)（不含 credit），\(\Theta_U\) 是 \(U\leftarrow U\exp(\Theta_U)\) 的反对称生成元。则
$$
\boxed{C_{ii}=\mathbb E_x\operatorname{Cov}_y(r,S_i),\qquad \langle G,\partial W/\partial(\Theta_U)_{ij}\rangle=\mathbb E_x\operatorname{Cov}_y(r,T^U_{ij}).}
$$
*证明.* \(\partial W/\partial\sigma_i=u_iv_i^\top\)，链式法则 \(\partial\log\pi/\partial\sigma_i=\langle\nabla_W\log\pi,u_iv_i^\top\rangle=u_i^\top\nabla_W\log\pi\,v_i\)。对 \(\Theta_U\)：\(dU=U\Theta_U\)，\(dW=U\Theta_U\Sigma V^\top\)；取 \((\Theta_U)_{ij}=\epsilon,(\Theta_U)_{ji}=-\epsilon\)，\(dW=\epsilon(\sigma_ju_iv_j^\top-\sigma_iu_jv_i^\top)\)，内积即得。两式代入推论 4.1。\(\blacksquare\)

**解读.** RLVR 谱分量 = reward 与"序列 log-likelihood 对 gain \(\sigma_i\) 的敏感度"的协方差。谱分量小 ⟺ 正确与错误 rollout 对 gain 变化的 likelihood 响应相同。

**Proposition 6（尺度不变约束）.** 若网络输出对 \(W\) 的整体缩放不变，则对任意 objective \(\langle G,W\rangle=\sum_i\sigma_iC_{ii}=0\)。
*证明.* \(f(cW)=f(W)\Rightarrow\frac{d}{dc}\mathcal L(cW)|_{c=1}=\langle G,W\rangle=0\)；\(\langle G,W\rangle=\operatorname{tr}(V\Sigma U^\top G)=\sum_i\sigma_i(U^\top GV)_{ii}\)。\(\blacksquare\)

所有 objective 共有；说明谱分量净效应天然被压制，但不能解释范式差异。Transformer 中对单矩阵仅近似成立，实验报告 \(\langle G,W\rangle/(\|G\|\|W\|)\)。

**Proposition 7（单步 toy：frame-dominance 不是平凡的）.** 考虑单步 policy \(\pi(y|x)=\mathrm{softmax}(Wx)_y\)。则
$$
S_i=a_i\big(u_{i,y}-\mathbb E_\pi[u_{i,\cdot}]\big),\qquad u_i^\top\nabla_W\log\pi\,v_j=a_j\big(u_{i,y}-\mathbb E_\pi[u_{i,\cdot}]\big),
$$
即对角与非对角项只差标量 \(a_i\) vs \(a_j\)，与 reward 的相关系数相同：\(\rho^\Sigma_i=\rho^{\rm frame}_{ij}\)。
*证明.* \(\nabla_W\log\pi(y)=(e_y-\pi)x^\top\)，\(u_i^\top(e_y-\pi)=u_{i,y}-\mathbb E_\pi[u_{i,\cdot}]\)，\(x^\top v_j=a_j\)。\(\blacksquare\)

**含义.** 单 token 结构不产生 frame-dominance。若实验观察到 RLVR 的 \(|\rho^\Sigma|\ll|\rho^{\rm frame}|\)，原因只能来自序列级效应或 pretrained 权重的结构，这正是下面假设的内容。

**Hypothesis H-RLVR（gain-sensitivity 的 reward 无关性）.** 对 on-policy 采样的 pretrained model，\(S_i(y)=\sum_te^0_{i,t}a_{i,t}\) 主要由 reward 无关因素决定：
1. **温度效应**：放大 \(\sigma_i\) 近似均匀缩放该 mode 贡献的 logits，主要改熵；组内归一化后熵与 \(A\) 相关弱。
2. **长度效应**：\(S_i\) 随长度累加，长度归一化削弱其与 \(A\) 的相关。
3. **使用对称性**：已有 mode 被正确与错误 rollout 以相近频率使用（它是 pretrained 能力，不是本次新学的）；序列级求和使 \(\sum_te^0_{i,t}a_{i,t}\) 的符号交替项相互抵消。

而 \(T_{ij}\) 编码"在此 prompt 上把 \(v_j\) 路由到 \(u_i\)"，是区分正误轨迹的信息。

**可证伪量.**
$$
\rho^\Sigma_i:=\operatorname{Corr}_y(A,S_i),\qquad\rho^{\rm frame}_{ij}:=\operatorname{Corr}_y(A,T_{ij}).
$$
预测：RLVR 下 \(|\rho^\Sigma|\) 集中于 0，\(|\rho^{\rm frame}|\) 有尾；OPD 下 \(|\rho^\Sigma|\) 显著更大。对照：对 pretrained 能力之外的 mode（新增随机 LoRA 方向），\(|\rho^\Sigma|\) 应上升（检验第 3 点）。

若实验支持，可在简化模型（线性 policy + 高斯 reward + 长序列）下尝试证明 \(\rho^\Sigma\to0\) 的集中不等式。

---

# 5. SNR 理论：幅值混合了信号与尺度

**设定.** 一个 prompt，\(K\) 条 rollout，\(\hat G=\frac1K\sum_kA_k\nabla\log\pi(y_k)\)。对任一 mode 统计量 \(s(y)\)（\(S_i\) 或 \(T_{ij}\)），\(\hat C_s=\frac1K\sum_kA_ks_k\)。记 \(\mu_s=\operatorname{Cov}(A,s)\)，\(\rho_s=\operatorname{Corr}(A,s)\)，\(\sigma_s^2=\operatorname{Var}(s)\)，组内归一化下 \(\sigma_A\approx1\)，\(\mathbb E[A]=0\)。

**Lemma 8.** \(\mathbb E[s]=0\)。
*证明.* \(s\) 是 \(\log\pi\) 对某参数化方向的导数，由 Lemma 4 期望为零。\(\blacksquare\)

**Proposition 9（GRPO 估计量的信号与噪声）.**
$$
\mathbb E[\hat C_s]=\mu_s=\rho_s\sigma_s,\qquad\operatorname{Var}[\hat C_s]=\frac1K\operatorname{Var}(As).
$$
若 \((A,s)\) 联合高斯，则 \(\operatorname{Var}(As)=\sigma_s^2(1+\rho_s^2)\)，从而
$$
\boxed{\operatorname{SNR}_s:=\frac{\mu_s^2}{\operatorname{Var}[\hat C_s]}=\frac{K\rho_s^2}{1+\rho_s^2}.}
$$
*证明.* \(\mathbb E[As]=\operatorname{Cov}(A,s)+\mathbb E[A]\mathbb E[s]=\mu_s\)。i.i.d. 平均给方差。联合高斯、均零：\(\mathbb E[A^2s^2]=\sigma_A^2\sigma_s^2+2\operatorname{Cov}(A,s)^2=\sigma_s^2(1+2\rho_s^2)\)（Isserlis），\(\operatorname{Var}(As)=\sigma_s^2(1+2\rho_s^2)-\rho_s^2\sigma_s^2=\sigma_s^2(1+\rho_s^2)\)。\(\blacksquare\)

**Proposition 10（幅值 ≠ 信号）.** 观测幅值
$$
|\hat C_s|\approx|\rho_s|\sigma_s+O\!\Big(\sigma_s\sqrt{\tfrac{1+\rho_s^2}{K}}\Big),
$$
同时依赖 \(\rho_s\)（signal）与 \(\sigma_s\)（与 signal 无关的尺度）；SNR 只依赖 \(\rho_s\)。

*证明.* 由 Prop.9 的均值与标准差。\(\blacksquare\)

**推论 10.1（Muon / Pion 的退化机制）.** Muon 与 Pion 的选择变量都是幅值 \(\approx|\rho|\sigma_s\)，在 \(\sigma_s\) 异质时系统性偏向高方差 mode。Pion 报告的"RLVR 下 Muon 放大 tail noise"的精确机制：不是 tail 本身是噪声，而是 tail 中 \(\sigma_s\) 大而 \(\rho_s\) 小的 mode 被幅值判据误选。

**推论 10.2（密度效应）.** 稠密监督（SFT / OPD）每位置有独立 credit。若各位置信号对齐、噪声独立，mode 统计量的信号 \(\propto T\)，噪声 \(\propto\sqrt T\)，有效 SNR 比 RLVR 高 \(O(T)\)。这是"Muon 在 pretraining 有效、RLVR 退化"的定量解释；可通过在 RLVR 中加密 credit（process reward）验证 SNR 随之上升。

**与 Adam 的关系.** Adam 的 \(m/\sqrt v\) 是逐坐标 SNR 归一化（\(m^2/v\)）；Muon 在谱基下做幅值归一化不看 SNR。本文方法 M1 = **谱基下的 SNR 归一化**。

---

# 6. Part I 实验协议

## 6.1 Setup（固定）

- Base：Qwen3-1.7B 或 Llama-3.2-3B，验证后上 7–8B。
- Prompt 集：数学（GSM8K / MATH 子集），三范式共用。SFT: prompt+标准解；OPD: student rollout + 同 family 大模型 reverse KL；RLVR: GRPO，correctness reward，\(K=8\)。
- 同 token budget，AdamW 为主，Muon 对 SFT / RLVR 各跑一遍。几百步。
- 每 10 步存 \(G_t,H_t,W_t\)，约 16 个矩阵（早/中/晚层 × Q/K/V/O/gate/up/down 抽样）。
- 每保存 step 额外 \(B=8\) 组 minibatch / rollout 组算 \(G^{(b)}\)。

## 6.2 指标

**几何**（对 \(G_t\) 与 \(H_t\) 各算）：\(R_{\rm spectrum}\)；\(G\) 能量在 \(W\) 的 top/mid/tail 子空间占比；\(G\) 的 top 奇异向量与 \(W\) 子空间的 principal angles；\(G\) 的 stable rank；累积量 \(\|\sigma(W_t)-\sigma(W_0)\|\)、子空间旋转（sanity）。

**统计**：\(\widehat{\rm SNR}_{ij}=\bar C_{ij}^2/\widehat{\operatorname{Var}}_b(C^{(b)}_{ij})\)；RLVR 下直接估 \(\rho^\Sigma_i,\rho^{\rm frame}_{ij}\)（每条 rollout 单独反传，只对少数矩阵）。

**功能**（MLP）：NaNA 的 SCA 贡献 \(c_i\)，held-out 集聚合 \(|c_i|\) 均值；RLVR 用高 advantage 位置。把 mode 分 high-/low-contribution 两桶，看 \(G\) 能量落在哪桶。

## 6.3 假设与预测

| 假设 | 定量预测 | 检验 |
|---|---|---|
| H1 谱倾向由 credit 密度决定 | \(R_{\rm spectrum}(\bar G)\): SFT > OPD > RLVR | Phase 1 主图 |
| H2 optimizer 只调制 | AdamW/Muon 不改 H1 的 ordering；Muon 下 \(R_{\rm spectrum}\) 只依赖 \(\mathrm{polar}(C)\)（Prop.3） | \(G\) vs \(H\) |
| H3 RLVR 谱分量 reward 无关 | \(\rho^\Sigma\) 集中于 0，\(\rho^{\rm frame}\) 有尾；OPD 的 \(\rho^\Sigma\) 更大 | §4 直接估计 |
| H4 幅值混合 \(\rho\) 与 \(\sigma_s\) | RLVR 中 \(|\bar C_{ij}|\) 与 \(\widehat{\rm SNR}_{ij}\) 秩相关低；SFT 高 | §6.2 |
| H5 spectrum-only 无法建新关联 | RLVR: spectrum-only 失败、frame-only ≈ full；SFT 相反 | Phase 2 |
| H6 SNR 选择优于幅值 | 同稀疏度 SNR-top-\(q\) > Mag-top-\(q\)，差距 RLVR 最大 | Phase 2 |
| H7 高贡献 mode 谱变化致遗忘 | 限制 high-\(c\) 的 \(\dot\sigma\) 减少遗忘不损任务 | Phase 2 + held-out |

## 6.4 干预协议（Phase 2）

训练循环中把 \(H_t\) 替换为投影：

| 约束 | 投影 |
|---|---|
| Spectrum-only | \(P_\Sigma(H)=U\operatorname{diag}(U^\top HV)V^\top\) |
| Frame-only | \(H-P_\Sigma(H)\)，或按 Prop.1 做严格 isospectral 更新 |
| High-\(c\) only / Low-\(c\) only | 按 SCA 桶取行列块 |
| SNR-top-\(q\) / Mag-top-\(q\) | 只保留 \(\widehat{\rm SNR}\) 或 \(|\bar C|\) 前 \(q\%\) 项 |

---

# Part II — 优化方法

每个方法绑定 Part I 的触发条件；条件不成立则方法不进 paper。

## M1. Spectral-SNR Descent（SSD）

**触发**：H4 + H6。
**规则**：梯度奇异基下 SNR 加权。
$$
\bar G=P\Lambda Q^\top,\qquad H=P\operatorname{diag}\big(f(\widehat{\rm SNR}_\ell)\big)Q^\top
$$
$$
m_\ell\leftarrow\beta_1m_\ell+(1-\beta_1)p_\ell^\top G_tq_\ell,\quad v_\ell\leftarrow\beta_2v_\ell+(1-\beta_2)(p_\ell^\top G_tq_\ell)^2,\quad\widehat{\rm SNR}_\ell=m_\ell^2/v_\ell
$$
\(f(s)=s/(1+s)\)（Wiener 系数，高斯下 MSE 最优）或硬阈值。\(f\equiv1\) 退化为 Muon。
基漂移：用上一步 \((P,Q)\) warm start 做 Newton–Schulz，Hungarian 匹配对齐 mode 后更新 EMA；只对 top-\(k\) 维护，尾部统一收缩。
**对比**：AdamW、Muon、Pion。
**预期**：RLVR 上不出现 Muon 退化；增益与 Part I 测得的 \(\rho,\sigma_s\) 异质性成正比（按 layer 散点）。SFT 上与 Muon 持平。
**开销**：与 Muon 同量级 + \(2k\) 标量 EMA / 矩阵。

## M2. Adaptive Spectrum–Frame Learning Rates

**触发**：H1 + H5。
**规则**：
$$
C=U^\top HV,\qquad H'=U\big(\alpha_t\operatorname{diag}C+\beta_t(C-\operatorname{diag}C)\big)V^\top,\qquad\alpha_t=g\Big(\tfrac1r\sum_i\widehat{\rm SNR}_{ii}\Big),\ \beta_t=1
$$
只有 gradient 持续给出可靠 gain 信号才允许改谱。ISO 是 \(\alpha\equiv0\)，普通训练 \(\alpha\equiv1\)。按 (layer, type) 独立维护。
**对比**：ISO、AdamW/Muon。三范式同一套规则。
**预期**：不劣于各自最优固定选择（RLVR≈ISO，SFT≈full），RLVR 上遗忘低于 full。
**开销**：每 \(N\) 步刷新权重 SVD，中间用 Prop.1 一阶传播。

## M3. Contribution-Aware Spectral Regularizer

**触发**：H7。
**规则**：SCA 贡献 \(c_i\)，\(\lambda_i\propto|c_i|\)，
$$
\mathcal R(W)=\sum_i\lambda_i\big(\sigma_i(W)-\sigma_i(W_0)\big)^2
$$
或 update 层面 \(C_{ii}\leftarrow C_{ii}/(1+\lambda_i)\)。只惩罚谱变化不惩罚 frame——允许新关联，禁止改写已有关联强度。
**对比**：KL-to-base、weight decay toward \(W_0\)、EWC。
**预期**：同 forgetting 下任务更高，SFT 上最明显。
**限制**：SCA 只对 MLP 有定义。

## M4. Layer-Wise Optimizer Routing

**触发**：Part I 发现 layer type 异质性大于 objective 差异。
**规则**：每矩阵监测对角 SNR 均值与 mode SNR 离散度，四象限选 AdamW/Muon、SSD、ISO-like、SSD+\(\alpha\to0\)。是 controller 不是新 optimizer。

## M5. Subspace-Locked Optimizer States

**触发**：gradient 高 SNR 子空间早期锁定。
**规则**：前 \(T_0\) 步全量；之后固定 rank-\(k\) 子空间，只在 \(P_k^\top G_tQ_k\) 上维护 Adam 状态，\(O(k^2)\) 而非 \(O(mn)\)；每 \(N\) 步用 SNR 检查覆盖。与 GaLore 区别：按 SNR 而非幅值选。

## M6. SNR-Guided Credit Densification

**触发**：推论 10.2。
**规则**：\(c_\tau=A+\gamma_tw^{\rm OPD}_\tau\)，\(\gamma_t\) 由 mode SNR 均值控制——SNR 低时多用 teacher，高时退回纯 RLVR。

## M7. SNR-Aware Rank Allocation for LoRA

**触发**：update spectrally concentrated 且高 SNR mode 数按层差异大。
**规则**：\(r_\ell=\#\{\widehat{\rm SNR}_\ell>\tau\}\)，总预算固定按层重分配；LoRA 初始化到高 SNR 子空间。

## 推荐配置

一篇 paper：Part I 完整 + M1 + M2，M3 附加。M4–M7 视 Part I 结果，留 future work 或第二篇。

**每个方法过三关**：(1) 消融——去掉 SNR / 谱分解退回 baseline；(2) 预测——Part I 测得的量能预测方法在哪些设置增益大；(3) 跨范式——RLVR 有效且 SFT 不劣。第 (2) 关是与"又一个 optimizer"的区别。

---

# 6b. Part II 理论

## 6b.1 M1 的最优性

**设定.** 固定谱基 \((P,Q)\)（假设在一步内稳定），真实梯度在该基下的 mode 系数 \(\mu_\ell\)，观测 \(\hat c_\ell=\mu_\ell+\varepsilon_\ell\)，\(\varepsilon_\ell\sim\mathcal N(0,\tau_\ell^2)\) 独立；先验 \(\mu_\ell\sim\mathcal N(0,s_\ell^2)\)。定义 \(\mathrm{SNR}_\ell:=s_\ell^2/\tau_\ell^2\)。update \(H=\sum_\ell h_\ell p_\ell q_\ell^\top\)，\(h=h(\hat c)\) 可测。一阶 loss 下降为 \(\langle G^*,H\rangle=\sum_\ell\mu_\ell h_\ell\)。

**Lemma 11（Wiener 收缩）.** \(\mathbb E[\mu_\ell|\hat c_\ell]=\dfrac{\mathrm{SNR}_\ell}{1+\mathrm{SNR}_\ell}\hat c_\ell=f(\mathrm{SNR}_\ell)\hat c_\ell\)。
*证明.* 高斯先验 + 高斯噪声的后验均值：\(\mathbb E[\mu|\hat c]=\frac{s^2}{s^2+\tau^2}\hat c\)。\(\blacksquare\)

**Proposition 12（SSD-Wiener 是范数预算下的 Bayes 最优 update）.** 在约束 \(\mathbb E\|h\|^2=1\) 下，
$$
\max_h\ \mathbb E\Big[\sum_\ell\mu_\ell h_\ell\Big]
$$
的解为 \(h^*_\ell\propto f(\mathrm{SNR}_\ell)\hat c_\ell\)，最优值为 \(\sqrt{\mathbb E\|f(\mathrm{SNR})\odot\hat c\|^2}=\sqrt{\sum_\ell s_\ell^2f(\mathrm{SNR}_\ell)}\)。

*证明.* 塔性质：\(\mathbb E[\sum_\ell\mu_\ell h_\ell]=\mathbb E[\sum_\ell\mathbb E[\mu_\ell|\hat c]h_\ell]=\mathbb E[\langle m,h\rangle]\)，\(m_\ell:=f(\mathrm{SNR}_\ell)\hat c_\ell\)。Cauchy–Schwarz：\(\mathbb E\langle m,h\rangle\le\sqrt{\mathbb E\|m\|^2}\sqrt{\mathbb E\|h\|^2}=\sqrt{\mathbb E\|m\|^2}\)，等号当且仅当 \(h\propto m\)。\(\mathbb E[m_\ell^2]=f^2\mathbb E[\hat c_\ell^2]=f^2(s_\ell^2+\tau_\ell^2)=s_\ell^2f(\mathrm{SNR}_\ell)\)。\(\blacksquare\)

**推论 12.1（Muon 的损失）.** Muon 取 \(h_\ell=\mathrm{sign}(\hat c_\ell)/\sqrt r\)（每 mode 单位增益）。设 \(r_0\) 个 mode 为纯噪声（\(s_\ell=0\)），\(r_1=r-r_0\) 个 mode 同 SNR \(=\kappa\)、同 \(s\)。则
$$
\frac{\mathbb E\langle G^*,H_{\rm Muon}\rangle}{\mathbb E\langle G^*,H^*\rangle}
=\sqrt{\frac{r_1}{r}}\cdot\frac{\mathbb E[\mu\,\mathrm{sign}(\hat c)]}{s\sqrt{f(\kappa)}}
=\sqrt{\frac{r_1}{r}}\cdot\sqrt{\frac{2}{\pi}}\cdot\sqrt{\frac{\kappa}{1+\kappa}}\Big/\sqrt{\frac{\kappa}{1+\kappa}}
=\sqrt{\frac{2r_1}{\pi r}}.
$$
*证明.* 噪声 mode 上 \(\mathbb E[\mu\,\mathrm{sign}(\hat c)]=0\)，Muon 把 \(r_0/r\) 的范数预算浪费在那里。信号 mode 上 \((\mu,\hat c)\) 联合高斯、\(\mathrm{Corr}=\sqrt{\kappa/(1+\kappa)}\)，\(\mathbb E[\mu\,\mathrm{sign}(\hat c)]=s\sqrt{2/\pi}\sqrt{\kappa/(1+\kappa)}\)。代入 Prop.12 的最优值。\(\blacksquare\)

**解读.** Muon 相对最优的效率是 \(\sqrt{2r_1/\pi r}\)：即使无噪声 mode（\(r_1=r\)）也只有 \(\sqrt{2/\pi}\approx0.80\)（sign 相对线性收缩的固有损失）；噪声 mode 占比越高损失越大。这与 Part I 的 H4（RLVR 中大量 mode \(\rho\approx0\)）直接对接：\(r_0/r\) 就是 Phase 1 可测的量，E3 预测关的横轴即由此而来。

**SSD 的两个变体.**
- **SSD-Wiener**：\(h_\ell=f(\mathrm{SNR}_\ell)\hat c_\ell\)，Prop.12 的最优解。
- **SSD-Muon**：\(h_\ell=f(\mathrm{SNR}_\ell)\mathrm{sign}(\hat c_\ell)\)，保留 Muon 的谱平坦化但按 SNR 门控。由推论 12.1 的计算，其效率为 \(\sqrt{2/\pi}\cdot\sqrt{\sum_\ell s_\ell^2f/\sum_\ell s_\ell^2}\) 量级，在噪声 mode 上不浪费预算。
两者都在 E2 消融中对比。

## 6b.2 M2 的谱漂移控制

**Proposition 13.** 令 \(H'=U(\alpha\operatorname{diag}C+\beta\,\mathrm{off}(C))V^\top\)，\(C=U^\top HV\)。则
1. \(\dot\sigma_i=\alpha C_{ii}\)，故 \(\|\sigma(W_{t+1})-\sigma(W_t)\|\le\alpha_t\|\operatorname{diag}C_t\|+O(\|H\|^2)\)，累积谱漂移 \(\le\sum_t\alpha_t\|\operatorname{diag}C_t\|\)。
2. 若 \(H=-\eta G\)，一阶 loss 变化为 \(-\eta(\alpha\|\operatorname{diag}C\|^2+\beta\|\mathrm{off}(C)\|^2)\le0\)，任意 \(\alpha,\beta\ge0\) 都是下降方向。
3. \(\alpha=0\) 时更新一阶等谱，恢复 ISO；\(\alpha=\beta=1\) 恢复原 update。

*证明.* (1) Prop.1 应用于 \(H'\)：\(U^\top H'V\) 的对角为 \(\alpha C_{ii}\)。(2) \(\langle G,H'\rangle=\langle C,\alpha\operatorname{diag}C+\beta\,\mathrm{off}(C)\rangle\)（Frobenius 内积在正交变换下不变），对角与非对角正交。(3) 直接。\(\blacksquare\)

**推论 13.1（SNR 门控的意义）.** 取 \(\alpha_t=g(\overline{\mathrm{SNR}}^{\rm diag}_t)\)，\(g(0)=0\)、单调。由 Lemma 11，对角 mode 的 Bayes 后验均值 \(\propto f(\mathrm{SNR}_{ii})C_{ii}\)；当对角 SNR 均为零时后验均值为零，此时 \(\alpha=0\) 是 Prop.12 意义下的最优选择，M2 与 SSD 在对角块上一致。M2 是 SSD 在"对角 vs 非对角"这个二分粒度上的粗化，代价是每步权重 SVD，好处是不需要 mode 级对齐。

## 6b.3 M3 的遗忘界

**Proposition 14.** 对 held-out 输入 \(x\)、目标 \(t^*\)，NaNA 的 DEU 贡献 \(c_i(x)=(\sigma_iv_i^\top x)(t^{*\top}u_i)\)，目标 logit \(=\sum_ic_i(x)\)。在 spectrum-only 变化 \(\Delta\sigma\) 下，
$$
\Delta\text{logit}(x)=\sum_i\frac{\Delta\sigma_i}{\sigma_i}c_i(x),\qquad
|\Delta\text{logit}(x)|\le\Big(\sum_i\lambda_i\Delta\sigma_i^2\Big)^{1/2}\Big(\sum_i\frac{c_i(x)^2}{\lambda_i\sigma_i^2}\Big)^{1/2}.
$$
取 \(\lambda_i=\bar c_i/\sigma_i^2\)（\(\bar c_i\) 为 held-out 集上 \(|c_i|\) 的均值），第二因子在 held-out 分布上的期望被 \(\sum_i\bar c_i\) 控制，与训练无关。因此 M3 的正则项 \(\mathcal R=\sum_i\lambda_i\Delta\sigma_i^2\) 直接界住 held-out logit 的谱漂移部分。

*证明.* 由 Prop.2，\(\Delta(W x)=\sum_i\Delta\sigma_i(v_i^\top x)u_i\)，与 \(t^*\) 内积得 \(\sum_i\Delta\sigma_i(v_i^\top x)(t^{*\top}u_i)=\sum_i(\Delta\sigma_i/\sigma_i)c_i\)。不等式为 Cauchy–Schwarz。\(\blacksquare\)

**解读.** M3 只约束 \(\Delta\sigma\)，frame 变化对 held-out logit 的影响不受约束——这是设计选择：由 Prop.2，新关联只能通过 frame 建立，M3 保留了这条通道。KL-to-base 和 L2 同时约束两者，因此在"加能力不毁能力"上 M3 的 Pareto 前沿应更靠外（E6 检验）。反过来，若 Phase 1 发现 held-out 能力损失主要来自 frame 而非谱（H7 不成立），M3 的界虽然成立但无用，方法不进 paper。

---



## 7.1 通用设置

- 模型、prompt 集、budget 与 §6.1 相同，保证 Part I 的测量可直接用于预测。
- 评测：
  - **任务**：GSM8K / MATH500 pass@1（RLVR、OPD、SFT 各自的目标任务）。
  - **遗忘**：MMLU、HumanEval、IFEval 相对 base 的变化，作为 held-out 能力。
  - **训练动态**：reward / loss 曲线、每步 \(R_{\rm spectrum}(H_t)\)、\(\|\sigma(W_t)-\sigma(W_0)\|\)。
- 每个配置 3 seeds，报告均值 ± std。
- 超参：每个 optimizer 在 SFT 上 sweep lr 一次，其它范式沿用（避免"调参优势"）。

## 7.2 实验清单

**E1：SSD vs baselines（M1 主实验）**
- 配置：\(\{\text{AdamW},\text{Muon},\text{Pion},\text{SSD}\}\times\{\text{SFT},\text{OPD},\text{RLVR}\}\)，12 组。
- 输出：任务性能表；RLVR 训练曲线（重点看 Muon 是否退化、SSD 是否不退化）。
- 通过标准：RLVR 上 SSD ≥ AdamW 且 > Muon；SFT / OPD 上 SSD ≈ Muon。

**E2：SSD 消融**
- \(f\equiv1\)（退化为 Muon）、硬阈值 vs Wiener、只对 top-\(k\) 维护 SNR 的 \(k\in\{64,256,\text{full}\}\)、EMA 系数、是否做 mode 对齐。
- 通过标准：去掉 SNR 加权后性能回落到 Muon 水平；mode 对齐去掉后明显退化。

**E3：预测关（Part I → M1）**
- 对每个 layer，用 Phase 1 测得的 mode-wise \((\rho_s,\sigma_s)\) 异质性（例如 \(|\bar C|\) 与 \(\widehat{\rm SNR}\) 的秩相关的 \(1-\)值）作横轴，SSD 相对 Muon 在该层单独启用时的增益作纵轴，画散点。
- 通过标准：正相关显著。这是 paper 区别于"又一个 optimizer"的关键图。

**E4：Adaptive spectrum–frame（M2 主实验）**
- 配置：\(\{\text{full},\ \text{ISO }(\alpha=0),\ \text{M2 自适应 }\alpha_t\}\times\{\text{SFT},\text{OPD},\text{RLVR}\}\)，AdamW 底座。
- 输出：任务性能 + 遗忘指标双轴；\(\alpha_t\) 随 step 和 layer 的轨迹图（预期 RLVR 上 \(\alpha_t\to0\)，SFT 上保持高）。
- 通过标准：M2 在三范式上都不劣于各自的最优固定选择；RLVR 上遗忘低于 full。

**E5：M2 的 layer 分解**
- 只在 attention 或只在 FFN 上启用自适应 \(\alpha\)，其余用 full。
- 输出：哪类 layer 贡献了 M2 的主要增益，与 Part I 的 layer 异质性对照。

**E6：Contribution-aware regularizer（M3，附加）**
- 配置：\(\{\text{无正则},\ \text{KL-to-base},\ \text{L2-to-}W_0,\ \text{M3}\}\)，SFT 与 RLVR。
- 输出：任务性能 vs 遗忘的 Pareto 曲线（扫 \(\lambda\) 强度）。
- 通过标准：M3 的 Pareto 前沿在 KL / L2 之外。

**E7：SSD 在 Muon 已知强项上的 sanity**
- 小规模 continued pretraining（几千步），SSD vs Muon。
- 通过标准：不劣于 Muon。防止审稿人质疑 SSD 只是在 RLVR 上 "修" 了 Muon 而牺牲了 pretraining。

## 7.3 全文图表清单

| 编号 | 内容 | 来源 |
|---|---|---|
| Fig.1 | 概念图：credit 密度连续谱 + spectral/frame 分解 | §2 |
| Fig.2 | \(R_{\rm spectrum}(\bar G_t)\) 三范式随 step，按 layer type 分面 | Phase 1, H1 |
| Fig.3 | \(R_{\rm spectrum}(G)\) vs \(R_{\rm spectrum}(H)\)，AdamW / Muon | Phase 1, H2 |
| Fig.4 | \(\rho^\Sigma\) 与 \(\rho^{\rm frame}\) 的分布，RLVR vs OPD | Phase 1, H3 |
| Fig.5 | \(|\bar C_{ij}|\) vs \(\widehat{\rm SNR}_{ij}\) 散点，三范式 | Phase 1, H4 |
| Fig.6 | 干预实验：spectrum-only / frame-only 下三范式性能 | Phase 2, H5 |
| Fig.7 | SNR-top-\(q\) vs Mag-top-\(q\) 性能随 \(q\) | Phase 2, H6 |
| Tab.1 | E1 主表：4 optimizer × 3 范式 | E1 |
| Fig.8 | RLVR 训练曲线：Muon 退化 vs SSD | E1 |
| Fig.9 | 预测关散点 | E3 |
| Fig.10 | \(\alpha_t\) 轨迹 + 性能 / 遗忘双轴 | E4 |
| Tab.2 | 消融 | E2, E5 |
| Fig.11 | M3 Pareto（附录或正文） | E6 |

## 7.4 算力估计（3B 模型）

- Phase 1：3 范式 × ~500 步 + 8 组 SNR 反传 ≈ 主训练的 3–4 倍开销，单节点 8×A100 约 3–4 天。
- Phase 2：6 种约束 × 3 范式 × 3 seeds ≈ 54 runs，每 run 短（几百步），约 1 周。
- E1–E7：约 60 runs，含 seeds，约 1.5–2 周。
- 7–8B 复现：只跑 E1 与 Fig.2/4，约 1 周。

## 7.5 依赖关系

```
Phase 1 (H1–H4) ──→ 决策点 ──→ Phase 2 (H5–H7)
     │                              │
     └──→ SNR 估计、α 调度 ──→ E1–E5 (M1, M2) ──→ E3 预测关（回用 Phase 1 数据）
                                    │
                                    └──→ E6 (M3, 依赖 H7)
```

Phase 1 的数据在 E3 里被第二次使用：这是分析与方法之间的硬链接。

---

# Part III — 执行

## 时间线

| 阶段 | 内容 | 周 |
|---|---|---|
| Phase 1 | 三范式跑完，出 \(R_{\rm spectrum}\)、\(G\) vs \(H\)、SNR 三张主图；估 \(\rho^\Sigma,\rho^{\rm frame}\) | 2–3 |
| 决策点 | H1 成立→Phase 2；不成立→主线改为 optimizer 效应（H2）+ §5 | — |
| Phase 2 | 干预实验，检验 H5–H7 | 2 |
| Phase 3 | M1、M2 实现与对比 | 2–3 |
| 理论 | 与 Phase 1 并行：H-RLVR 在简化模型下的证明尝试 | — |

## 工程

- SVD：randomized top-256 + tail 块；或只在保存 step 离线算。
- Mode 跨 step 对齐：principal angle / Hungarian，不按 index。
- SNR 估计：RLVR 至少 8 组 rollout，主要算力开销。
- SVD 符号歧义对 \(u_i^\top Gv_i\) 无影响；功能标签两方向都看。

## 待定

1. Base model 与算力（1.7B 还是 3B 起）。
2. OPD teacher 来源。
3. 是否做 Phase 3 / M1–M2，决定 venue 与时间线。

---

# 附：理论的诚实边界

- **已证（Part I）**：Prop.1、2、3、6、7，Lemma 4、8，推论 4.1，Prop.9、10。
- **已证（Part II）**：Lemma 11（Wiener 收缩）、Prop.12（SSD-Wiener 的 Bayes 最优性）、推论 12.1（Muon 效率 \(\sqrt{2r_1/\pi r}\)）、Prop.13（M2 谱漂移界与下降性）、Prop.14（M3 遗忘界）。Part II 的证明依赖"谱基在一步内稳定 + 高斯先验"两个假设，前者由 E2 的 mode 对齐消融检验。
- **假设**：H-RLVR。§4 给出其精确 covariance 形式与可证伪量；Prop.7 说明它不是平凡的。若 Phase 1 支持，尝试在简化模型下证明。
- **近似**：Prop.9 的闭式依赖联合高斯；不依赖近似的是 \(\mu_s=\rho_s\sigma_s\) 与 \(\operatorname{SNR}\propto K\rho_s^2\)。Prop.6 在 Transformer 中对单矩阵近似成立。
- **独立性**：即使 H1/H3 被证伪，§5 SNR 理论 + M1 仍构成一篇完整 paper。
