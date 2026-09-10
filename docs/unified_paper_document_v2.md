# When Should Post-Training Change the Spectrum?
## Reliable Gradient Geometry and the Conditional Value of Spectral Motion

**中文题目：后训练何时值得改变权重谱？——可靠梯度几何与谱更新的条件增益**

版本：v2，2026-09-09  
文档类型：统一研究文档，包含论文叙事、完整局部理论、测量定义、方法设计与实验裁决协议。  
状态：已有探索性观察 + 本文推导的局部命题 + 尚待执行的确认性实验。本文不是已经完成所有验证的结果论文。

---

# 0. 研究问题、贡献与证据边界

## 0.1 核心问题

给定一个 pretrained checkpoint，post-training 可以改变权重的奇异值，也可以改变其左右奇异向量。本文研究：

> **当 frame 更新已经可用时，额外开放 spectral 更新是否提供了可靠、不可被当前 frame 更新替代、且值得其功能代价的局部收益？**

问题中的“可靠”指对独立采样可复现；“功能代价”由指定参考分布上的局部 KL 度量及实际能力评测共同描述；“不可替代”只相对于明确列出的候选方向与局部模型成立。

我们不预设 RLVR 必然保谱、SFT 必然改谱，也不预设 spectrum 方向高曲率、frame 方向低曲率。SFT、OPD、RLVR 是检验适用范围的训练设置，而不是预先规定答案的三个类别。

## 0.2 三个贡献目标

| 编号 | 内容 | 当前性质 | 成功标准 |
|---|---|---|---|
| C1 | 在权重奇异基下，将观测更新能量分为可复现平均梯度与采样噪声，并显式校正子空间维数 | 测量框架；恒等式可证明 | 独立样本与合成数据能恢复已知信号；真实模型上有稳定、带区间的测量 |
| C2 | 定义“允许 frame 后，额外开放 spectrum 的条件增益”，保留参考 Fisher 的谱—frame 耦合，并校正估计噪声造成的乐观偏差 | 本文的核心研究提案；局部代数结果可证明 | 在未用于设计的 checkpoint／任务上，预测有限步干预收益优于对角能量、幅值和单独 SNR |
| C3 | 根据独立 probe 对候选更新增益的评估，动态决定是否开放谱更新 | 方法假设，尚未验证 | 在包含 probe 成本的匹配预算下，改善任务收益—参考功能变化的折中 |

**创新定位。** Fisher 恒等式、Schur complement、交叉样本二次型、置信下界均是标准数学工具，不单独声称其为新定理。拟研究的贡献是把它们用于同一个 spectral/frame 决策问题，形成可复现测量、条件增益预测和独立干预裁决的闭环。是否构成充分的论文创新，取决于 C2/C3 的确认性结果及进一步文献比较，不以换名替代验证。

## 0.3 本文明确不作的推断

- 对角能量小，不等于 policy gradient 具有特殊的保谱机制。
- action-independent baseline 不改变期望 policy gradient，不能单独解释谱方向的选择性消失。
- 未观察到两类相关性分布的差异，不等于证明二者相等。
- 某个机制假设未获支持，不会使另一机制成为唯一解释。
- Fisher 曲率不等于任意数据分布上的 pretraining Hessian。
- 局部二次 surrogate 的最优性，不等于整个 LLM 训练过程的最优性。
- 固定谱下能学习，不等于 unrestricted RLVR 的每一步都严格保谱。
- 本文新方法尚未实现和验证，旧 SSD 的运行结果不能转记为本方法结果。

---

# 1. 摘要与论文主线

## 1.1 摘要草案

大语言模型后训练中的谱继承现象引出了一个优化问题：何时需要改变 pretrained 权重的奇异值？仅观察谱漂移或梯度的非对角能量，难以回答这一问题，因为高维几何、采样噪声和功能敏感度会同时影响这些统计量。我们提出一个局部分析框架，在权重的奇异基下分别度量可复现梯度、梯度估计噪声与参考策略 Fisher 几何。在指定候选空间和正定二次代价下，我们将开放谱方向的价值定义为相对于 frame-only 局部最优解的增量，并通过 Schur complement 显式保留谱—frame 的功能耦合。该增量可以用独立组样本的交叉二次型无偏估计，从而避免把噪声平方误认作可用收益。基于此，我们设计独立 probe 驱动的谱更新门控，比较已经固定的 full 与 frame 候选更新。现有探索性结果显示范式间的归一化谱能量差异温和，尚不足以确认统一的 credit-density 机制。本文给出跨目标函数、架构、学习阶段及预算的确认性协议，用于检验条件谱增益能否预测实际干预收益，并指导何时开放谱更新。

这是研究提案的摘要；完成确认性实验后，应以实际效果量和适用边界替换最后两句，不提前使用“显著提升”“首次证明”等措辞。

## 1.2 逻辑链

1. 定义当前权重下的 spectral/frame 方向及其维数。
2. 在固定 checkpoint 上分离平均梯度与采样噪声。
3. 用指定参考分布上的 Fisher 描述局部功能敏感度。
4. 测量开放谱方向相对于 frame 候选空间的额外价值。
5. 用独立样本评价实际候选更新，再检验有限步和长期训练收益。

全文的主要被解释量从“谁的对角比例更小”改为“何时开放谱方向更有效”。

---

# 2. 相关工作与可检验的差异

## 2.1 谱继承与固定谱优化

ISO 报告 RLVR 中的谱继承、checkpoint surgery 和固定谱训练结果，并将其作为优化的归纳偏置，而非无约束 policy gradient 严格等谱的定律。本文接受其作为研究动机，进一步问谱约束何时有利、何时应该释放。[ISO](https://arxiv.org/html/2607.19331v1)

## 2.2 OPD 的参数更新几何

On the Geometry of On-Policy Distillation 分析 OPD 的 off-principal 结构与 subspace locking，并提供子空间约束和目标混合对照；Dense Supervision, Sparse Updates 分析稀疏性、谱集中和功能性子网络。这些工作已经覆盖部分轨迹与干预问题，不能笼统称为“只有 endpoint 分析”。本文的区别拟在于固定 checkpoint 的噪声校正、功能度量和条件增益预测。[OPD geometry](https://arxiv.org/abs/2606.07082)；[Dense Supervision, Sparse Updates](https://arxiv.org/abs/2606.13657)

近期关于极稀疏 OPD 监督的结果，也说明监督 token 数与学习效果之间不存在无需条件的单调规律。因此本文将 credit 密度降为一个需要受控检验的因素。[Extremely Sparse Supervision Incentivizes Reasoning Ability](https://arxiv.org/abs/2609.04565)

## 2.3 谱滤波、方差适配与自然梯度

Muon 对动量进行近似正交化；高通 Pion 保留主要谱方向并抑制尾部，还讨论 RLVR 的低 SNR。本文不能把“RLVR 噪声较大”或“尾部需要收缩”作为独立新贡献。[Muon](https://github.com/KellerJordan/Muon)；[Pion: high-pass optimization](https://arxiv.org/html/2605.19282v1)

文献中另有同名 Pion，通过左右正交变换保持权重谱，需在引用与实验配置中明确区分。[Pion: spectrum-preserving optimization](https://arxiv.org/html/2605.12492v1)

Adam 的相对方差解释和自然梯度均已有长期研究。本文的 score/Fisher 联系属于已有理论的应用；不能宣称 Adam 就是在当前权重奇异基下估计 SNR，也不能把经验标签梯度的 outer product 直接当成真实 Fisher。[Dissecting Adam](https://arxiv.org/abs/1705.07774)；[A Natural Policy Gradient](https://papers.neurips.cc/paper/2073-a-natural-policy-gradient.pdf)；[Limitations of the Empirical Fisher](https://arxiv.org/abs/1905.12558)

GRPO 的组相依性及 U-statistic 分析已有专门工作。本文使用独立完整 rollout group 进行统计，不将组内项错误地视为独立，也不把组基线偏差修正当作新的贡献。[GRPO U-statistic analysis](https://arxiv.org/abs/2603.01162)

## 2.4 需要超越的最接近解释

我们的比较对象不仅是优化器名称，还包括四个预测：

- 仅由 \(R_\Sigma\) 或真实谱漂移判断是否开谱；
- 仅由 gradient/momentum 奇异值大小判断可用方向；
- 仅由方向 SNR 判断是否更新；
- 仅由 Fisher 对角项判断更新代价。

只有保留谱—frame 耦合的条件增益，在独立数据上稳定提供额外预测价值，C2 才成立。若简单 SNR 或对角 Fisher 已经足够，则应报告这一结果并简化方法。

---

# 3. 统一符号与训练目标

## 3.1 三个不同对象

本文统一采用最大化 \(J(\theta)\) 的符号：

\[
g=\nabla_\theta J,\qquad \widehat g=\text{随机梯度估计},\qquad
\Delta\theta=\text{优化器实际施加的参数变化}.
\]

单个权重矩阵用 \(G,\widehat G,H\) 表示相应对象。若代码最小化 loss，则分析时使用其负梯度。动量、预条件、学习率、裁剪和 decoupled weight decay 均不隐含在 \(G\) 的定义里。

| 符号 | 定义 |
|---|---|
| \(W=U\Sigma V^\top\) | 当前 checkpoint 的权重薄 SVD |
| \(r=\min(m,n)\) | 主理论假设下的满秩矩阵秩 |
| \(\Pi_\Sigma,\Pi_F\) | 参数空间中的正交 spectral/frame 投影 |
| \(g_{\rm alg}\) | 指定实际随机更新估计量的均值，不一定是真实 reward gradient |
| \(\Gamma\) | 单个独立 gradient unit 的协方差 |
| \(F_{\rm cur},F_{\rm ref}\) | 指定采样分布下的当前策略 Fisher 与参考功能 Fisher |
| \(Q=[Q_\Sigma,Q_F]\) | 向量化参数空间中的有限正交候选字典 |
| \(c=Q^\top g\) | 候选字典下的目标梯度 |
| \(A\succ0\) | 候选系数空间中的二次代价矩阵 |
| \(S_{\Sigma\mid F}\) | \(A\) 的 Schur complement；不是奇异值矩阵 |
| \(\mathcal U_{\Sigma\mid F}\) | 在该字典和代价下，开放谱方向的局部条件增益 |

用下标 cur/ref、alg/R 明确区分分布与目标，避免在不同公式之间替换它们。

## 3.2 线性层 outer-product 恒等式

对层内激活 \(h_t=Wx_t\)，任一可微标量样本目标均满足：

\[
G=\sum_t\delta_t x_t^\top,\qquad
\delta_t=\frac{\partial J_{\rm sample}}{\partial h_t}.
\]

这只是反向传播恒等式。训练设置不仅改变 \(\delta_t\)，也通过轨迹分布改变 \(x_t\)、隐状态和 Jacobian。因此不能说“三范式唯一结构差别是一个 credit 向量”。

## 3.3 SFT

固定外部数据分布：

\[
J_{\rm SFT}=\mathbb E_{(x,y)\sim D_{\rm SFT}}
\sum_t\log\pi_\theta(y_t\mid x,y_{<t}).
\]

长度加权、token 平均方式、mask 和 label smoothing 必须按实现登记。SFT 的每个 token 有监督，不意味着 token 梯度噪声独立。

## 3.4 OPD：区分固定采样 surrogate 与完整目标

默认研究可实现的 frozen-prefix reverse-KL surrogate。先从冻结行为策略采样 prefix 集 \(\mathcal S\)，对这些 prefix 做 stop-gradient：

\[
J_{\rm OPD}^{\rm frozen}(\theta)
=-\mathbb E_{s\sim\mathcal S}
D_{\rm KL}\big(\pi_\theta(\cdot\mid s)\Vert\pi_T(\cdot\mid s)\big).
\]

对词表分布直接求导，teacher 固定。若使用 sampled-token estimator、forward KL 或不同 token 权重，则另列配置，不复用未验证的公式。

如果选择完整 trajectory reverse KL：

\[
L(\theta)=\mathbb E_{y\sim\pi_\theta}
[\ell_\theta(y)],\qquad
\ell_\theta=\log\pi_\theta(y\mid x)-\log\pi_T(y\mid x),
\]

则在标准正则条件及固定 teacher 下：

\[
\nabla L=\mathbb E[\ell_\theta(y)\nabla\log\pi_\theta(y\mid x)],
\]

因为直接导数的期望为零。该完整目标与 frozen-prefix surrogate 不可混称为同一个梯度；本文的实验必须选择并记录其中一种。

## 3.5 RLVR：原始 policy gradient 与 GRPO 分开

对参数无关的 reward 和固定 prompt 分布：

\[
J_R=\mathbb E_{x,y\sim\pi_\theta}[r(x,y)],\qquad
g_R=\mathbb E[(r-b(x))z],\quad
z=\nabla_\theta\log\pi_\theta(y\mid x).
\]

真实 GRPO 的 group advantage、token/sequence averaging、old-policy ratio、clipping、KL 项和多轮 reuse 会改变实际估计量。记完整 group 的 ascent gradient 为 \(X_b\)，定义：

\[
g_{\rm alg}:=\mathbb E[X_b].
\]

统计框架可直接分析 \(g_{\rm alg}\)，但解释真实 reward 改进时，需要独立估计 \(g_R\) 或直接评估 reward。不能仅靠 \(g_{\rm alg}\) 的局部改进保证 \(J_R\) 上升。GRPO 原始定义见 [DeepSeekMath](https://arxiv.org/html/2402.03300v2)。

若需要抽象 credit，必须写出具体 estimator \(X=\sum_t c_tz_t\) 以及 \((c_t,z_t)\) 的联合分布；“稠密/稀疏”只描述这个 estimator 的一部分。

---

# 4. 正确的 spectral/frame 几何

## 4.1 主理论的适用域

先假设 \(W\in\mathbb R^{m\times n}\) 满秩，其 \(r\) 个正奇异值互异。薄 SVD 的 \(U,V\) 分别为 \(m\times r,n\times r\)。候选方向使用 Frobenius 内积；涉及多个矩阵时使用向量化后的直和空间。

**命题 1：谱投影与完整余空间。**

\[
\Pi_\Sigma(H)=\sum_{i=1}^r\langle H,u_iv_i^\top\rangle u_iv_i^\top,
\qquad \Pi_F(H)=H-\Pi_\Sigma(H).
\]

则这两个投影正交：

\[
\|H\|_F^2=\|\Pi_\Sigma H\|_F^2+\|\Pi_FH\|_F^2.
\]

谱子空间维数为 \(r\)，\(\Pi_F\) 对应的空间是该点 fixed-spectrum orbit 的切空间，维数为 \(mn-r\)。

**证明。** \(u_iv_i^\top\) 为 Frobenius 正交归一组，故第一式是正交投影。左右正交作用的切向量为 \(\Omega_LW-W\Omega_R\)，其中生成元反对称，其谱对角为零。对互异正奇异值，每对内部非对角项的生成元方程可逆；矩形矩阵的外部项由相应左右生成元实现，因而切空间等于上述正交补。∎

## 4.2 矩形矩阵与薄 SVD

令 \(C=U^\top HV\)。一般有：

\[
H=UCV^\top+(I-UU^\top)HVV^\top
+UU^\top H(I-VV^\top)
+(I-UU^\top)H(I-VV^\top).
\]

满秩矩形矩阵的最后一项为零，但其中一个外部项仍可非零。因此，薄 SVD 中 \(U\,\operatorname{offdiag}(C)V^\top\) 通常不是完整 frame 更新。

实现时优先使用：

\[
H_F=H-\Pi_\Sigma H,
\]

而不是仅保留 \(C\) 的非对角项。所有能量分母使用 \(\|H\|_F^2\)；若只分析 core block，必须明确称作 conditional core energy。

## 4.3 一阶导数与有限步

**命题 2：简单正奇异值的一阶变化。**

\[
\left.\frac{d}{d\epsilon}\sigma_i(W+\epsilon H)\right|_{\epsilon=0}
=u_i^\top Hv_i.
\]

**证明。** 对 \(Wv_i=\sigma_i u_i\) 微分并左乘 \(u_i^\top\)，利用单位向量导数与自身正交即可。∎

完整的薄 SVD 向量导数为：

\[
\dot u_i=\sum_{j\ne i}
\frac{\sigma_iC_{ji}+\sigma_jC_{ij}}{\sigma_i^2-\sigma_j^2}u_j
+\frac{(I-UU^\top)Hv_i}{\sigma_i},
\]

\[
\dot v_i=\sum_{j\ne i}
\frac{\sigma_iC_{ij}+\sigma_jC_{ji}}{\sigma_i^2-\sigma_j^2}v_j
+\frac{(I-VV^\top)H^\top u_i}{\sigma_i}.
\]

在固定谱间隙和正奇异值下，足够小扰动满足：

\[
\sigma_i(W+\epsilon H)
=\sigma_i(W)+\epsilon C_{ii}
+O_W(\epsilon^2\|H\|_F^2).
\]

余项常数依赖局部谱条件；不能用这个公式跨过简并点。逐步谱漂移的累积上界必须同时累加一阶项和受控的二阶余项。

frame-only 加性更新只是一阶等谱。严格保谱比较器使用正交因子更新及 retraction，或显式固定奇异值重构。二者是不同算法。

## 4.4 简并、近简并与截断

对重复正奇异值块 \(B\)，一阶谱 splitting 由
\(\operatorname{sym}(U_B^\top HV_B)\) 的特征值决定，不由任意选定基下的对角独立决定。该块的 symmetric part 是谱法向部分，skew part 对应切向部分。重复块会改变法向维数。

实际协议：

1. 用预先固定的相对 gap 阈值标记近简并块，并报告阈值敏感性。
2. 精确简并使用块定义；近简并同时报告块级敏感性与真实有限步谱漂移，不把近似块划分当成精确定理。
3. 数值秩亏或零奇异值不直接使用满秩理论；零—零块可能产生新的奇异值，需要单独分析。
4. top-\(k\) SVD 只覆盖选定谱方向，未覆盖部分单列为 residual，不称为 frame。

若只研究 dominant spectrum，则比较的是“开放 top-\(k\) 谱方向”与一个指定背景空间；背景可以包含 tail-spectrum 和 frame。此时应标记 \(\mathcal U_{\Sigma_{\rm top}\mid{\rm background}}\)，不得称为全谱—frame 分解。

## 4.5 维数校正

**命题 3：各向同性基线。** 对非零随机 \(H\)，若其向量化方向在单位球面上均匀分布，则：

\[
\mathbb E R_\Sigma(H)=\frac{r}{mn},\qquad
R_\Sigma(H)=\frac{\|\Pi_\Sigma H\|_F^2}{\|H\|_F^2}.
\]

**证明。** 球面对称性给每个正交坐标相同的期望能量占比 \(1/(mn)\)，谱投影保留 \(r\) 个坐标。∎

定义：

\[
\mathcal E_\Sigma(H)=\frac{R_\Sigma(H)}{r/(mn)}.
\]

还应使用保持 \(H\) 奇异值、随机化左右方向的 Haar null，以排除梯度低秩本身的影响；其平均谱能量比例同样为 \(r/(mn)\)。头结构保持的 block null 回答不同问题，需另列。

“frame energy 大”与“相对随机方向偏好 frame”必须分别报告。

---

# 5. 从观测能量中分离可靠梯度与噪声

## 5.1 独立统计单位

冻结 checkpoint、采样温度、目标实现和数据分布，取独立单位 \(X_1,\ldots,X_B\)。RLVR 的单位通常是“一个 prompt 及其完整 rollout group”，而非共享归一化的单条 rollout。若固定 prompt 做组内条件分析，则每个 \(X_b\) 是该 prompt 的一个独立完整 group。

\[
\mathbb E X_b=g,\qquad \operatorname{Cov}(X_b)=\Gamma,\qquad
\bar X=B^{-1}\sum_bX_b.
\]

若 \(X_b\) 包含多个 prompt，则必须固定 batching 规则。把不同 checkpoint 或不同 prompt 的条件均值变化当成重复采样噪声，会改变 estimand。

## 5.2 观测能量的精确分解

**命题 4。** 对预先固定、或由独立设计样本确定的正交投影 \(\Pi\)，若二阶矩有限：

\[
\boxed{
\mathbb E\|\Pi\bar X\|^2
=\|\Pi g\|^2+\frac1B\operatorname{tr}(\Pi\Gamma\Pi^\top).
}
\]

**证明。** 写 \(\bar X=g+\varepsilon\)，其中 \(\mathbb E\varepsilon=0\)、\(\operatorname{Cov}(\varepsilon)=\Gamma/B\)，展开平方并取期望。∎

分别取 \(\Pi_\Sigma,\Pi_F\)，即可得到平均梯度与采样噪声在两类空间中的分布。该恒等式不要求高斯。

## 5.3 交叉样本信号能量

定义：

\[
\widehat E_\Pi^{\rm signal}
=\frac{1}{B(B-1)}\sum_{b\ne c}
\langle\Pi X_b,\Pi X_c\rangle.
\]

**命题 5。** 在命题 4 的条件下，
\(\mathbb E\widehat E_\Pi^{\rm signal}=\|\Pi g\|^2\)。

**证明。** 对 \(b\ne c\)，独立性给
\(\mathbb E\langle\Pi X_b,\Pi X_c\rangle=\langle\Pi g,\Pi g\rangle\)。∎

无需显式计算所有样本对；令 \(\widehat\Gamma\) 为无偏样本协方差：

\[
\widehat E_\Pi^{\rm signal}
=\|\Pi\bar X\|^2-\frac1B\operatorname{tr}(\Pi\widehat\Gamma\Pi^\top).
\]

有限样本估计可以为负，表示信号未被当前精度分辨，而不是“真实能量为负”。推断和区间计算不先裁零。若总信号能量接近零，不构造不稳定的 signal-energy ratio；报告两部分能量及区间。

## 5.4 两种 SNR 必须区分

对固定单位范数方向 \(d\)，令 \(c_b=d^\top X_b\)、\(\mu_d=d^\top g\)、\(\tau_d^2=\operatorname{Var}(c_b)\)：

\[
\mathrm{SNR}_{\rm unit}(d)=\frac{\mu_d^2}{\tau_d^2},\qquad
\mathrm{SNR}_{\rm mean}(d)=\frac{B\mu_d^2}{\tau_d^2}.
\]

旧指标 \(\bar c^2/\widehat{\operatorname{Var}}(c_b)\) 是带有限样本偏差的 unit-SNR plug-in，不是均值梯度估计的 SNR。若方向由同一批样本的 SVD 选出，还会有选择偏差。

推断使用独立设计样本确定方向；在测量样本上估计均值、方差和区间。任何去噪平方估计也不直接当成无偏的“比值估计”。

## 5.5 跨 prompt 与跨时间的含义

对不同 prompt，条件梯度的异质性是任务总体抽样方差的一部分，但不等于固定 prompt 的 rollout noise。层级分解为：

\[
\operatorname{Cov}(X)
=\mathbb E_x[\operatorname{Cov}(X\mid x)]
+\operatorname{Cov}_x(\mathbb E[X\mid x]).
\]

两项均应测量。跨 step 的方向变化则属于非平稳性；不能把长时间 EMA 的全部波动解释为固定 checkpoint 的采样噪声。

---

# 6. Reward correlation、Fisher 与 GRPO

## 6.1 Score/Fisher 恒等式

假设支持集在局部固定、可交换微分与期望且所需矩有限。对固定 prompt：

\[
z=\nabla_\theta\log\pi_\theta(y\mid x),\quad
\mathbb E[z\mid x]=0,\quad
F_x=\mathbb E[zz^\top\mid x].
\]

对固定方向 \(d\)，令 \(s_d=d^\top z\)。

**命题 6。**

\[
\boxed{
\operatorname{Var}(s_d\mid x)=d^\top F_xd,\qquad
d^\top g_R(x)=
\rho_{r,s_d\mid x}\,\sigma_{r\mid x}\sqrt{d^\top F_xd}.
}
\]

**证明。** 第一式来自 score 零均值与二阶矩定义；第二式由 \(g_R(x)=\mathbb E[(r-b(x))z\mid x]\) 和 covariance 定义。相关性仅在两变量方差非零时使用。∎

因此 score 方差是功能敏感度，不是“与信号无关的任意尺度”。而真实梯度估计方差：

\[
\operatorname{Var}((r-b(x))s_d\mid x)
\]

还涉及 reward 与 score 的联合高阶矩及 baseline。它不等于 \(d^\top F_xd\)。

跨 prompt 平均得到的是上述条件 covariance 的均值，不能随意用混池后的一个相关系数替代。

## 6.2 KL 曲率、pretraining Hessian 与参考分布

固定 prompt 分布 \(D_{\rm ref}\)，在当前参数 \(\theta\) 附近定义：

\[
K_{\rm ref}(\delta)
=\mathbb E_{x\sim D_{\rm ref}}
D_{\rm KL}\big(\pi_\theta(\cdot\mid x)
\Vert\pi_{\theta+\delta}(\cdot\mid x)\big).
\]

在光滑条件下：

\[
K_{\rm ref}(\delta)
=\tfrac12\delta^\top F_{\rm ref}\delta+o(\|\delta\|^2).
\]

这里参考 policy 是当前 \(\pi_\theta\)，参考 prompt 分布可以是 held-out 能力集。若换成固定 pretrained policy \(\pi_{\theta_0}\) 与当前模型的 KL，在 \(\theta\ne\theta_0\) 处一般有非零线性项，不能继续使用同一个展开。

估计 \(F_{\rm ref}\) 要在 \(D_{\rm ref}\) 上从相应当前 policy 采样，或直接计算条件输出 KL 的局部导数。真实标签上的梯度 outer product 需另称 empirical Fisher；通常不等于本文的 \(F_{\rm ref}\)。[Empirical Fisher limitations](https://arxiv.org/abs/1905.12558)

held-out supervised loss 的 Hessian \(H_{\rm pre}\) 可不定，且其梯度未必为零。若分析实际遗忘，要同时保留：

\[
\Delta L_{\rm pre}
=\nabla L_{\rm pre}^\top\delta+
\tfrac12\delta^\top H_{\rm pre}\delta+o(\|\delta\|^2).
\]

小的逐步参考 KL 不保证长期不遗忘，更不保证全部能力保留；长期指标必须直接评估。

## 6.3 单步 softmax 不提供通用 frame 对称性定理

固定 \(x\)，\(\pi(y\mid x)=\operatorname{softmax}(Wx)_y\)。令
\(a_j=v_j^\top x\)、\(b_i(y)=u_i^\top(e_y-\pi)\)，则：

\[
C_{ij}(y)=a_jb_i(y).
\]

若方差非零且 \(a_i,a_j\ne0\)，对固定 \(i\)：

\[
|\operatorname{Corr}(r,C_{ii})|
=|\operatorname{Corr}(r,C_{ij})|.
\]

带符号的等式取决于 \(a_i,a_j\) 的符号。它不能推出旋转生成元的相关性相同，因为：

\[
T^U_{ij}=\sigma_jC_{ij}-\sigma_iC_{ji}.
\]

实验分别报告 entry-direction correlation 和 generator correlation。生成元矩阵
\(\sigma_ju_iv_j^\top-\sigma_iu_jv_i^\top\) 的范数为
\(\sqrt{\sigma_i^2+\sigma_j^2}\)；比较方差和曲率前需归一化。左右生成元合起来可能冗余；构造字典时要正交化并去除零方向，不能把它们的平方直接相加称为 frame 总能量。

## 6.4 Baseline 与有限 group 的区别

总体 action-independent baseline 满足：

\[
\mathbb E[(r-b(x))z\mid x]=\mathbb E[rz\mid x].
\]

但组内均值包含当前样本自身。对固定 \(x\)，独立 rollout 对 \((r_k,s_k)\)、\(\mathbb E s_k=0\)，有：

\[
\boxed{
\mathbb E\left[\frac1K\sum_k(r_k-\bar r)s_k\right]
=\frac{K-1}{K}\operatorname{Cov}(r,s).
}
\]

**证明。** \(\mathbb E[r_ks_k]=\operatorname{Cov}(r,s)\)；在 \(\mathbb E[\bar r s_k]\) 中只有第 \(k\) 项保留，等于 \(\operatorname{Cov}(r,s)/K\)。∎

leave-one-out baseline 可消除此特定因子，但不消除组内各项的相依性。除以随机组标准差还会引入额外耦合及 prompt-dependent weighting。全对或全错的 binary-reward group 需要明确的 epsilon/zero-advantage 规则；不能静默删除这些组后仍声称原始目标不变。

## 6.5 SNR 闭式的准确边界

对独立总体样本，\(\mathbb E A=\mathbb E s=0\)，一般有：

\[
\mathbb E[\widehat c]=\operatorname{Cov}(A,s),\qquad
\operatorname{Var}(\widehat c)=\frac1K\operatorname{Var}(As).
\]

若额外假设 \((A,s)\) 联合高斯且 \(\operatorname{Var}(A)=1\)，由 Isserlis 公式：

\[
\operatorname{Var}(As)=\sigma_s^2(1+\rho^2),\qquad
\mathrm{SNR}_{\rm mean}=\frac{K\rho^2}{1+\rho^2}.
\]

联合高斯模型只用于解释尺度消去，不作为 binary-reward GRPO 的精确模型。实际完整 group \(X_b\) 的方差由跨组样本直接估计。

关于序列长度，任何 \(X=\sum_t\xi_t\) 都有：

\[
\operatorname{Var}(X)=\sum_t\operatorname{Var}(\xi_t)
+2\sum_{t<u}\operatorname{Cov}(\xi_t,\xi_u).
\]

只有额外控制跨 token covariance、平均信号和长度归一化，才能推出随长度的 SNR scaling；不存在本文已证明的“SFT/OPD 必然比 RLVR 高 \(O(T)\)”结论。

---

# 7. 核心理论：谱更新的条件增益

## 7.1 为什么仅看对角幅值或 SNR 不够

即使谱方向有可复现梯度，该收益也可能已被允许的 frame 更新在功能空间中实现；反过来，一个原始谱梯度很小的方向，也可能通过抵消 frame 更新的参考功能扰动而提供增益。因此需要同时考虑梯度、代价和谱—frame 耦合。

在有限候选字典中进行分析：

\[
\delta=Q_\Sigma a+Q_Fb,\qquad Q^\top Q=I.
\]

其中 \(Q_\Sigma\) 的列属于谱子空间，\(Q_F\) 的列属于 frame 切空间。若只取少量列，下面的“full”仅指该候选字典的全部方向，不是全模型参数空间。

取：

\[
A=\lambda Q^\top F_{\rm ref}Q+\gamma I,\qquad
\lambda\ge0,\quad \gamma>0.
\]

\(\gamma\) 控制参数步长与病态求逆，\(\lambda\) 控制参考 KL 代理代价。也可使用另一个明确指定的正定二次代价；所有结论均以实际 \(A\) 为条件。

定义局部 surrogate：

\[
\mathcal J(a,b)
=c_\Sigma^\top a+c_F^\top b
-\frac12
\begin{pmatrix}a\\b\end{pmatrix}^{\!\top}
\begin{pmatrix}
A_{\Sigma\Sigma}&A_{\Sigma F}\\
A_{F\Sigma}&A_{FF}
\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}.
\]

这是“目标的一阶收益减去指定二次代价”，不是完整 reward Taylor 展开；它没有包含 reward Hessian。要研究真实有限步收益，需要独立干预验证。

## 7.2 条件谱价值定理

**命题 7：开放谱方向的增量。** 若 \(A\succ0\)，定义：

\[
S_{\Sigma\mid F}
=A_{\Sigma\Sigma}-A_{\Sigma F}A_{FF}^{-1}A_{F\Sigma},
\]

\[
r_{\Sigma\mid F}
=c_\Sigma-A_{\Sigma F}A_{FF}^{-1}c_F.
\]

则 \(S_{\Sigma\mid F}\succ0\)，frame-only 最优系数为
\(b_0=A_{FF}^{-1}c_F\)，且：

\[
\boxed{
\mathcal U_{\Sigma\mid F}
:=\max_{a,b}\mathcal J(a,b)-\max_b\mathcal J(0,b)
=\frac12r_{\Sigma\mid F}^\top
S_{\Sigma\mid F}^{-1}r_{\Sigma\mid F}.
}
\]

联合最优解为：

\[
a_*=S_{\Sigma\mid F}^{-1}r_{\Sigma\mid F},
\qquad
b_*=A_{FF}^{-1}(c_F-A_{F\Sigma}a_*).
\]

**证明。** 固定 \(a\)，对 \(b\) 求极值得
\(b(a)=A_{FF}^{-1}(c_F-A_{F\Sigma}a)\)。代回：

\[
\max_b\mathcal J(a,b)
=\tfrac12c_F^\top A_{FF}^{-1}c_F
+r_{\Sigma\mid F}^\top a
-\tfrac12a^\top S_{\Sigma\mid F}a.
\]

正定矩阵的 Schur complement 正定；再对 \(a\) 完成平方即可。∎

这一定理是标准块二次优化的应用。本文赋予其具体的 spectral/frame 测量与实验意义，不主张 Schur complement 公式本身是新的。

## 7.3 可解释性与反例

- \(c_\Sigma\)：原始目标对谱方向的压力。
- \(A_{\Sigma F}A_{FF}^{-1}c_F\)：在当前代价模型下，最优 frame 更新与谱方向之间的耦合项。
- \(r_{\Sigma\mid F}\)：允许 frame 后剩余的谱收益梯度。
- \(S_{\Sigma\mid F}\)：允许 frame 联动后的谱有效代价。

只有当 \(A_{\Sigma F}=0\) 时，条件增益才退化为
\(\tfrac12c_\Sigma^\top A_{\Sigma\Sigma}^{-1}c_\Sigma\)。若再取 \(A=I\)，才与谱梯度平方直接对应。

**反例 A：原始谱梯度为零，条件谱价值仍可非零。**

\[
A=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix},
\quad |\rho|<1,\qquad c=\begin{pmatrix}0\\1\end{pmatrix}.
\]

则：

\[
r_{\Sigma\mid F}=-\rho,\qquad
\mathcal U_{\Sigma\mid F}=\frac{\rho^2}{2(1-\rho^2)}.
\]

谱方向通过降低与 frame 联动的二次代价提供收益。

**反例 B：原始谱梯度非零，条件谱价值为零。** 在相同 \(A\) 下取 \(c=(\rho,1)^\top\)，则 \(r_{\Sigma\mid F}=0\)，额外开放谱方向没有 surrogate 增益。

因此，谱梯度幅值既不是条件谱价值为正的必要条件，也不是充分条件。

## 7.4 这个量没有保证什么

1. \(\mathcal U_{\Sigma\mid F}\ge0\) 来自候选空间嵌套，不能据此证明 full 训练实际更好。
2. \(\mathcal U\) 依赖参考分布、\(\lambda,\gamma\) 和候选字典，必须随这些选择报告。
3. 增加未匹配的谱自由度可能提高拟合能力，故需同维随机扩展作为对照。
4. 该量只评价当前局部空间，不代表全局能力“存储于 spectrum”。
5. 小字典的结果不能直接推广到全部 frame 方向；候选覆盖率需单独验证。
6. 改变 \(A\) 为不定 Hessian 后，正定最优性和上式不再自动成立。
7. 换成精确 isospectral retraction 后，实际 reward 的二阶项会包含路径曲率；不能把加性 surrogate 的二阶结论原样移植。

## 7.5 噪声会系统性抬高 plug-in 条件增益

固定 \(Q,A\)，使用独立测量样本的系数 \(\widehat c\)，满足
\(\mathbb E\widehat c=c\)、\(\operatorname{Cov}(\widehat c)=V\)。定义：

\[
L=[I,\,-A_{\Sigma F}A_{FF}^{-1}],\quad
\widehat r=L\widehat c,\quad V_r=LVL^\top.
\]

**命题 8：二次 plug-in 偏差。**

\[
\mathbb E\left[
\tfrac12\widehat r^\top S_{\Sigma\mid F}^{-1}\widehat r
\right]
=\mathcal U_{\Sigma\mid F}
+\tfrac12\operatorname{tr}(S_{\Sigma\mid F}^{-1}V_r).
\]

**证明。** 对 \(\widehat r=r+\varepsilon\) 展开二次型；交叉项均值为零，噪声二次项均值为 trace。∎

这明确指出：方向多、测量噪声大、有效代价小，都会抬高表观“开谱收益”。直接用正的 plug-in \(\mathcal U\) 开门，可能只是拟合噪声。

## 7.6 交叉样本条件增益

取独立完整 units 的投影 \(c_b=Q^\top X_b\)、\(r_b=Lc_b\)：

\[
\boxed{
\widehat{\mathcal U}_{\Sigma\mid F}^{\rm cross}
=\frac{1}{2B(B-1)}
\sum_{b\ne c}r_b^\top S_{\Sigma\mid F}^{-1}r_c.
}
\]

**命题 9。** 当 \(Q,A\) 在测量样本之外固定，units 独立同分布且二阶矩有限时：
\(\mathbb E\widehat{\mathcal U}^{\rm cross}=\mathcal U_{\Sigma\mid F}\)。

**证明。** 独立性给每个交叉项的期望为
\(r^\top S_{\Sigma\mid F}^{-1}r\)，再按系数求和。∎

等价的实现是：

\[
\widehat{\mathcal U}^{\rm cross}
=\frac12\left[
\bar r^\top S_{\Sigma\mid F}^{-1}\bar r
-\frac1B\operatorname{tr}
(S_{\Sigma\mid F}^{-1}\widehat{\operatorname{Cov}}(r_b))
\right].
\]

这只是条件于固定字典与代价的无偏性：若 \(A\) 是估计值，估计目标是该固定估计代价下的 \(\mathcal U\)，不是自动对真实 Fisher 代价无偏。应独立检查 metric estimation error。

零信号下该二次 U-statistic 可能退化，不能默认普通正态近似或朴素 bootstrap 给出有效区间。实际方法门控采用下一节的固定候选线性比较；\(\widehat{\mathcal U}^{\rm cross}\) 主要作为机制诊断，区间用经零信号/弱信号模拟校准的程序，并报告校准结果。

## 7.7 坐标与度量

对候选空间内不改变 spectral/frame 两个子空间的可逆换基，若 \(c,A\) 一起正确变换，两个优化问题的最优值及其差不变。保持 \(Q\) 正交可以让 \(\gamma I\) 明确对应参数 Euclidean damping。

不能在非正交换基后继续机械地使用同一个 \(\gamma I\)；那会改变代价。改变网络参数化、参考分布或候选空间同样不是单纯换基。

---

# 8. 方法：Probe-Gated Spectral Updates

## 8.1 方法目的

暂名 **Probe-Gated Spectral Updates（PGSU）**。它是“何时开放谱方向”的更新控制器，不重新发明完整基础优化器。默认不假设 RLVR 总应关谱，也不要求三范式产生固定 ordering。

主要研究版本先使用固定小字典和显式二次求解，验证决策原则。训练时的低开销近似是后续工程目标，不能引用旧 SSD 的 +15% 开销作为 PGSU 的测量。

## 8.2 三份数据，各有固定用途

| 数据池 | 用途 | 不能用于 |
|---|---|---|
| Design | 构造 \(Q\)、估计代价 \(A\)、估计梯度并生成候选更新 | 报告对这些候选的无选择偏差确认性增益 |
| Probe | 对已经固定的候选做增益比较；估计交叉条件增益 | 反复搜索许多候选后只报告最好的未经校正结果 |
| Evaluation | 评估真正的有限步和长期性能 | 构造方向、选阈值、选学习率 |

Design 与 Probe 至少使用独立完整 groups；若要声称对新 prompt 泛化，必须进一步使用独立 prompt。同 prompt 的新 rollout 只检验条件采样复现性。

每次 probe 都在同一个冻结 checkpoint 下采样；复用旧 checkpoint 的 probe 需要另行建模分布漂移。

## 8.3 字典构造与候选更新

以 2–16 个方向为初始可行性配置，实际数量由成本预实验确定，而非固定为最终最优超参。

谱字典可取预先指定的 \(u_iv_i^\top\)，或 Design 数据中 \(\Pi_\Sigma G\) 的归一化聚合方向。frame 字典可取 \(\Pi_FG\)、Design 中多个独立梯度的 frame 投影经正交化后的方向，以及匹配的随机 frame 方向。

同一矩阵至少比较：

- full-candidate：允许 \(Q_\Sigma\) 和 \(Q_F\)；
- frame-candidate：固定谱系数为零；
- random-extension：在匹配维数及候选选择预算下添加随机方向。

用 Design 的 \(\widehat c_D,A\) 计算命题 7 的 full 与 frame 系数，得到候选 \(\delta_{\rm full},\delta_F\)。也可以使用基础优化器的 full/frame 投影构造候选，但此时不再声称它们是命题 7 的局部最优解。

步长、damping、参考 KL 目标以及任何 rescaling 都在查看 Probe 前确定。若对候选使用不同缩放，则比较缩放后的真实候选，不能继续把命题 7 的未缩放闭式值当作二者增益差。

## 8.4 独立 probe 比较实际候选

设候选 \(\delta_1,\delta_0\) 与代价算子 \(M\) 在 Probe 之前固定，其中在候选空间有 \(Q^\top MQ=A\)。定义：

\[
D=\delta_1-\delta_0,\qquad
\Delta q=\frac12(\delta_1^\top M\delta_1-\delta_0^\top M\delta_0).
\]

真实的固定 surrogate 增益差为：

\[
\Delta\mathcal J=g^\top D-\Delta q.
\]

使用 Probe 中的独立 units：

\[
Y_b=X_b^\top D,\qquad
\widehat{\Delta\mathcal J}=\bar Y-\Delta q.
\]

这是一个一维均值估计，比在 Probe 中搜索高维最优方向更易校准。其方差为 \(\operatorname{Var}(Y_b)/B\)；两候选共用同一 Probe 实现配对比较。

令 \(\ell_\alpha\) 是 \(g^\top D\) 的有效单侧下置信界，门控为：

\[
\boxed{
\text{open spectrum}
\iff \ell_\alpha-\Delta q>\epsilon_{\rm practical}.
}
\]

\(\epsilon_{\rm practical}\ge0\) 是 Design/validation 阶段预先确定的实际收益阈值。否则选择 frame 候选。这个保守默认是待检验的设计选择，不是理论规定；应报告其漏开谱的代价。

**命题 10：固定 surrogate 的条件决策保证。** 如果在给定 Design 的条件下，\(\ell_\alpha\le g^\top D\) 的概率至少 \(1-\alpha\)，则“接受 full 但其真实固定-surrogate 增益差不超过 \(\epsilon_{\rm practical}\)”的概率不超过 \(\alpha\)。

**证明。** 在下界覆盖事件上，接受条件给
\(\Delta\mathcal J\ge\ell_\alpha-\Delta q>\epsilon_{\rm practical}\)；错误接受只能发生在覆盖失败事件上。∎

该保证不包含 \(M\) 对真实功能代价的误差、目标一阶近似误差或 GRPO estimator bias。它也不保证长期训练收益。多个 layer、多个候选和反复 probing 需要多重比较或顺序决策校正；单次覆盖率不能被写成整次训练的保证。

## 8.5 置信界如何获得

- 若可以证明 \(Y_b\in[a,b]\) 且边界在 Probe 前固定，可使用 Hoeffding 单侧界：
  \[
  \ell_\alpha=\bar Y-(b-a)\sqrt{\frac{\log(1/\alpha)}{2B}}.
  \]
- 在独立正态 \(Y_b\) 模型下，可以用 Student-\(t\) 单侧界；真实 rollout gradient 不默认满足此模型。
- 大样本近似或 group bootstrap 只能标作渐近/经验校准；必须在零信号、弱信号与重尾合成数据中检查误开率。
- 若重尾和样本量使覆盖不可验证，报告点估计与不确定性，不宣称有限样本保证。可以使用有明确矩条件的稳健均值估计，但须同时给出所用条件。

门控消融必须包含“同样额外采样但不做门控”的对照，区分方法收益与额外 rollout 的收益。

## 8.6 精确保谱与加性 frame 更新分别评测

PGSU 的基础研究版本使用加性候选，frame 候选满足 \(\Pi_\Sigma\delta_F=0\) 的一阶条件。

精确保谱变体需要把 frame 候选变换为实际 retraction 后的参数位移，再作为固定候选交给 Probe；若其实际位移含二阶法向分量，应按实际 \(D\) 比较。命题 10 的固定候选线性比较仍可使用，但命题 7 的加性最优性不能直接套用。

实验中 ISO、additive frame-only 和 PGSU 的 exact-retraction 变体分别列名。

## 8.7 工程成本与在线近似

成本按以下项分别报告：

- 权重 SVD/候选字典构建；
- 方向导数与参考 Fisher 小矩阵；
- Design、Probe 额外 rollout 和 backward；
- 小型线性系统求解；
- 模型候选应用与验证；
- 常驻状态内存及总 wall-clock。

每矩阵不同方向的 \(F_{\rm ref}\) 交叉项不能省略后还声称验证了耦合机制；可以先在小字典精确计算，再比较 block-diagonal 近似。

跨 layer 的全模型 Fisher 也有交叉块。初期因果实验只改一个矩阵，避免把逐层最优拼起来称为全局最优；多层训练阶段将独立门控视作近似并加入联合验证。

如果每步 Probe 不可承受，可以周期性重估并冻结门控若干步，但需测量失效速度；该近似不保留每步独立决策保证。

## 8.8 旧 SSD 的处理

旧 SSD 留为基线或独立消融，不作为 v2 主方法：

1. 其权重基诊断与动量基更新需要分开测量。
2. 理想平稳矩下 \(m^2/v\) 近似 \(\mu^2/(\mu^2+\tau^2)\)，不是 \(\mu^2/\tau^2\)；实际不同 EMA 系数下还需处理偏差与非平稳性。
3. Wiener posterior mean 是 \(f(\mathrm{SNR})\widehat c\)，与仅按 \(f\) 门控的 Muon-like 更新不同。
4. 从当前噪声数据构造 SVD 基并同时评价可靠性，会产生选择偏差。
5. Newton–Schulz 可近似 polar factor，并不自动返回用于逐 mode 对齐的左右 SVD 基。
6. 候选平均梯度平方与 Bayesian prior variance 不是同一个 SNR 定义。

旧实现中 reward 正常上升只能说明该实现能运行，不能证明去噪估计正确或本文条件决策成立。

---

# 9. 已有探索性数据：保留数字，重写推断

## 9.1 来源与追溯

以下数字转录自 docs/unified_paper_document_v1.md 和 paper/results_draft.md（v1 登记处），日期均为 2026-09-08。本次重写没有重新运行实验或独立重算这些数字。它们用于设计确认性实验，不作为 v2 新命题的实证验证。

原数值登记文档记载主设置：Qwen3.5-0.8B，GSM8K，同 prompt 流，主训练 seed 0，500 步；对 27 个矩阵与 50 个保存点进行中位数聚合。重复矩阵和保存点并非独立 training seeds。

**2026-09-09 更正。** 初稿曾质疑原文档“跨家族全部完成”的表述强于登记表。经核对 paper/results_draft.md 与 results/v1/result_A/，Llama-3.2-3B 与 Llama-3.2-1B 的 SFT/OPD/RLVR 六格均已登记且目录齐全（各 500 步日志、约 1051 行指标），该质疑撤回。仍需通过 run manifest 记录每格的 seed、饱和窗口与捕获配置后再决定采用范围。

## 9.2 保留的观察

| 观察 | 原登记数字 | 当前可支持的表述 |
|---|---|---|
| Qwen raw-gradient 谱富集 | SFT 0.886，OPD 0.869，RLVR 0.839 | 这组探索性结果存在温和 ordering；不推出普遍规律或 credit 因果性 |
| Qwen 幅值—SNR 秩相关 | 0.880 / 0.874 / 0.848 | ordering 存在，但三者绝对相关均高。原因是统计量本身耦合：v1 代码 analysis/compute_metrics.py 中 \(\widehat{\rm SNR}=\bar C^2/\widehat{\rm Var}\) 与 \(|\bar C|\) 取自同一组 8 个微批次，\(|\bar C|\) 同时出现在两侧，秩相关不可能低；且 RLVR 每微批次仅 2 条序列、同组 advantage 经组归一化，微批次并非独立单位。作者B 的 E1 各 run（任意 optimizer × 范式）该值均落在 0.845–0.890。**已重算（2026-09-10，analysis_v2/compute_metrics.py，results/v2/result_A）**：把 8 个微批次分成独立两半后，\(|\bar C_A|\) 对 \(\widehat{\rm SNR}_B\) 的秩相关在六个 run 全为 0.000–0.002，\(|\bar C_A|\) 对 \(|\bar C_B|\) 为 0.05–0.06。原 0.880/0.874/0.848 的 ordering 无可解释性，撤销 |
| **交叉样本信号能量 / 噪声能量（新，v2 指标）** | SFT 0.0036 / 0.0022，OPD 0.0073 / 0.0038，RLVR 0.0004 / 0.0004（AdamW / Muon）；信号为正的矩阵-步占比 17–27%；300 步后均为 0 | B=8 下真实平均梯度的能量不到噪声的 1%，v1 的 G 侧谱富集、家族差异与范式 ordering 全部处于噪声主导区。H-A 在 0.8B 成立。稠密范式的信号占比高于 RLVR，方向与 credit 密度影响 SNR 一致，但量级不足以支持任何机制主张 |
| Llama-3B SFT/RLVR 全程秩相关 | 0.863 / 0.785 | 两设置有差异；需控制训练饱和与估计偏差 |
| Llama-3B 谱富集 | 全程 1.073 / 1.072；step≤100 为 1.084 / 1.060；OPD 1.082 | 对阶段选择敏感；活跃窗口为探索性事后分析，需独立确认 |
| Llama-1B（无饱和，reward 全程 53–57%）谱富集 / 秩相关 | SFT 1.067 / 0.846；OPD 1.073 / 0.818；RLVR 1.057 / 0.806 | OPD 富集高于 SFT，SFT>OPD 子排序不成立；三模型上仅“稠密 > RLVR”方向一致，且跨度 ≤2% |
| 加强采样的 \(|\rho^\Sigma|,|\rho^{\rm frame}|\) 中位数 | 0.1314 / 0.1308；匹配 null 0.1305 | 当前统计未显示预测中的显著分离；不等于总体相关性完全相等 |
| 相同实验 \(|\rho|>0.3\) 的占比 | 13.0% / 12.9%；null 11.4% | 是观测超阈率，不是“真实相关 mode 比例” |
| Qwen 架构分组谱富集 | linear attention 约 0.65，MLP 约 0.98 | 提示结构异质性；架构因果归因仍需匹配 shape/head 结构等对照 |
| Muon 的 H/G 谱富集比 | SFT 约 1.13，OPD 1.10，RLVR 约 1.0 | optimizer 的单步/轨迹调制可能依赖设置；不证明噪声机制 |

H5/H6 与方法主表数字已由作者B 于 2026-09-09 交付，见 §9.4。仍缺失的是 Qwen3.5-4B 的 RLVR/OPD 行与 9B 全部数字，不得补出。不同文档的不一致应进入审计记录，不能凭叙事需要择取。

## 9.3 暂停使用的原推断

- “H2 强成立，因此 objective 决定几何”：跨 optimizer 训练轨迹还混有参数、rollout 和学习率差异。
- “H3 干净证伪，因此 SNR 是唯一机制”：不成立。
- “Prop.7 的对称性被真实序列模型验证”：该 proposition 与 generator correlation 的对象不一致。
- “8×统计力”：应登记实际独立样本量与效应检测能力，不由样本数倍增直接命名。
- “H6 通过即可证明原机制”：任何后续干预还需排除额外采样、子空间维数和步长差异。

## 9.4 作者B 交付的干预与方法结果（2026-09-09）

来源：results/v1/result_B/（36 个条目，每 run 含 args、log、eval、metrics）。设置：Qwen3.5-0.8B，GSM8K，P=8、K=8；Phase 2/E2/E3/E4 为 300 步，E1 为 500 步；评测为 GSM8K test 前 500 题 greedy pass@1，base = 0.546（A 的 200 题口径为 0.575）。单个数字的二项 95% 区间约 ±0.044，**5 点以内的差异不作结论**。这些结果是在 v1 协议下、未做步长/KL 匹配、大多单 seed 的探索性干预，不是 §11 的确认性实验。

| 实验 | 结果（GSM8K pass@1） | 当前可支持的表述 |
|---|---|---|
| H5，RLVR：full / frame_only / spectrum_only | 0.636 / 0.636 / 0.566 | 去掉谱分量无可测损失；只保留谱分量时训练 reward 停在 0.36（full 为 0.60） |
| H5，SFT：full / frame_only / spectrum_only | 0.412 / 0.406 / 0.354 | 与 RLVR 同型。v1 预测的“SFT 相反”未出现 |
| H6，RLVR：snr_topq / mag_topq（q=0.1） | 0.544 / 0.558 | 无差异；两者都低于 full，训练 reward 均停在 0.36–0.39 |
| H6，SFT：snr_topq / mag_topq | 0.384 / 0.382 | 无差异 |
| E1，RLVR：AdamW / Muon / SSD / SSD-Muon | 0.670, 0.694 / 0.604, 0.624 / 0.568, 0.564, 0.560 / 0.584 | SSD 最差且跨 3 seed 一致；训练 reward SSD 末 50 步 0.38，AdamW 0.59–0.64 |
| E1，SFT：AdamW / Muon / SSD | 0.438 / 0.356 / 0.420, 0.404 | SSD ≈ AdamW，优于 Muon |
| E1，OPD：AdamW / Muon / SSD | 0.544 / 0.470 / 0.558, 0.556 | 同上 |
| E2，SSD 消融（rlvr 300 步）：k64 / sign / noalign / notail | 0.552 / 0.558 / 0.574 / 0.552 | 四个消融互相不可分，没有任何部件的去除造成可测变化 |
| E3，分层 SSD（其余层 Muon，300 步）：0–5 / 6–11 / 12–17 / 18–23 | 0.584 / 0.580 / 0.618 / 0.600 | 均在全 Muon（0.604/0.624，500 步）噪声范围内；没有某段层上 SSD 带来增益 |
| E4，自适应 α（300 步）：SFT / OPD / RLVR | 0.416 / 0.556 / 0.626；MMLU 0.365 / 0.262 / 0.478 | GSM8K 与各自对照持平（SFT none 0.412，RLVR none 0.636）；α 在三范式都从约 0.2–0.4 于 40 步内收敛到 0.04–0.07，无范式差异；SFT 的 MMLU 遗忘（0.483→0.366）未被 α 减少 |
| 4B：base / SFT AdamW | 0.862 / 0.818 | SFT 再次下降；4B 的谱富集 0.98（0.8B 为 0.83–0.88） |

**必须一并写入的 caveat。**

1. **步长混淆。** 干预直接对 AdamW 的实际更新 \(H\) 做投影，未做 §8.4 / E4a 的等范数或等 KL 匹配。none 的 \(R_\Sigma(H)\approx 2.8\times10^{-4}\)，即 spectrum_only 只保留了约 0.03% 的更新能量，frame_only 保留 99.97%；top-q 同样只保留 \(r\times r\) 块的 10% 项。因此 H5 目前只能支持“去掉谱分量无损失”，不能支持“谱方向不能学习”；H6 的 SNR 与幅值比较也在同样的欠训练区间内进行。
2. **SSD 的失败与 §8.8 的第 2 条一致**：\(m^2/v\in[0,1]\)，再经 \(f=s/(1+s)\le 0.5\) 与按 \(\max|c|\) 归一化，等效步长远小于同学习率的 Muon，训练曲线显示其几乎未学习。E1 未做学习率扫描，SSD 的阴性结果不能与 PGSU 的任何主张互相转记。
3. **SFT 在两个尺度上都降低 GSM8K 与 MMLU**（0.8B：0.546→0.412，0.483→0.366；4B：0.862→0.818）。可能是 GSM8K 短答案风格压制了模型原生推理，也可能是作者B 的评测脚本未包含 adf1786 的 stop-ids 修复。在作者B 报出代码 commit 前，所有 SFT 行只做同口径内部比较。
4. **OPD 的 α run MMLU 为 0.262**，接近四选一随机，而 GSM8K 正常。缺 OPD none 的 MMLU 对照，必须补跑后才能解释。
5. **捕获的 \(H\) 与干预不一致**：spectrum_only run 中，捕获的 \(H\) 在权重奇异基下的 \(R_\Sigma\) 仅为随机基线的 1.3–9 倍（SFT 随步数从 9 倍衰减到 3.7 倍），而按代码应接近 1，即基线的约 \(10^3\) 倍。见 §12.4 的核查项。此项不影响 eval 结论，但在澄清前，作者B 交付的所有 \(H\) 侧谱指标不得使用。
6. 除 E1 RLVR 行外均为单 seed；所有 run 共享同一 prompt 流。

**这批数据对 v2 的意义。** 在三范式上，frame-only 与 full 不可分，自适应 α 一致收敛到接近关谱，这是 §0.1 核心问题目前唯一的直接证据，方向是“开放谱的边际价值在该设置下测不出来”。它把 ISO 类的观察从 RLVR 推广到 SFT 与 OPD，同时否定了 v1 的范式依赖预测。这一结论需要 E4a 的等范数版本确认后才可写成主结果。

---

# 10. 确认性假设与裁决标准

## 10.1 假设表

| 假设 | 预测 | 拒绝或降级条件 |
|---|---|---|
| H-A：观测几何混有重要采样噪声 | 随独立 group 平均数增加，观测能量按命题 4 分解；部分 raw ordering 在信号校正后改变 | noise correction 不影响关键结论，或估计精度不足以分辨 |
| H-B：条件谱增益有额外预测价值 | \(\widehat{\mathcal U}^{\rm cross}\) 对未见干预收益的预测，优于谱能量、幅值、SNR 与对角曲率 | 与简单指标无稳定增量预测力 |
| H-C：谱—frame 耦合具有实用意义 | 保留 \(A_{\Sigma F}\) 的预测优于 block-diagonal \(A\)，且差异随耦合强度变化 | 去掉交叉块没有影响，或效果仅来自更高估计成本 |
| H-D：独立 probe 门控提高决策质量 | 对实际固定候选收益，门控减少误开谱且代价可接受 | 无法校准，或额外样本给 baseline 带来的收益同样大 |
| H-E：方法在总预算下有效 | PGSU 在至少一个预登记设置改善任务—功能代价折中，并明确其它设置的适用边界 | 只在不计 overhead 或 baseline 调参不足时获益 |

H-B/H-C 是论文核心，H-E 决定是否加入优化方法主张。不要求三范式出现固定的 SFT>OPD>RLVR 排序。

**当前状态（2026-09-10）。** H-A：在 0.8B 六个 Phase 1 run 上成立（§9.2 新行；paper/results_draft.md v2 登记处 A1）——B=8 下平均梯度信号能量不到噪声的 1%，v1 的 H4 统计量为耦合产物。H-B 至 H-D 尚无针对性数据。H-E 目前没有任何正面证据：同一方法家族的两个实现（§9.4 的 SSD 与自适应 α）在 0.8B 上分别为阴性与无效，PGSU 本身未实现。因此在阶段 C 完成之前，优化方法主张不进入论文主线；H-E 若届时仍无正面结果，按 §14 的阴性路径只保留离线诊断。

## 10.2 预先锁定的主要终点

主要预测终点：固定候选 full 与 frame 在 Evaluation 上的一步实际目标收益差，连同实际参考 KL 代价。对 RLVR 另外报告真实 reward 差，而不只报告训练 surrogate。

主要方法终点：总 GPU 时间预算内的任务性能与参考能力保持；等 token 曲线作为补充。由于额外 Probe 可能改变总训练步数，两种预算都要展示。

实际收益阈值、参考代价权重、主要 checkpoint、主要模型、方向字典大小和比较指标，在确认性 Evaluation 数据解封前锁定。

如需“不劣于”主张，预先给出 non-inferiority margin 及统计设计；“没有显著差异”不自动表示不劣。

---

# 11. 实验设计

## E0. 数学与测量校验

合成实验必须覆盖：

1. 方阵与高/宽矩形矩阵的谱投影能量守恒。
2. frame 外部子空间分量及其一阶奇异向量导数。
3. 一阶 frame 加性更新的二阶谱漂移与精确保谱的差别。
4. 各向同性及固定梯度谱的 Haar null。
5. 独立噪声下交叉信号能量、plug-in 偏差与交叉条件增益的期望。
6. Schur complement 的 full/frame 最优值差及 §7.3 两个反例。
7. 组内中心化的 \((K-1)/K\) 因子；group std、zero-reward groups 对估计目标的影响。
8. 单步 softmax 下 entry correlation 与 generator correlation 的差别。
9. 零信号、弱信号、重尾噪声及不同独立样本量下的区间覆盖和误开率。

这些校验验证实现与有限模型，不替代证明或 LLM 实验。

## E1. 固定 checkpoint 的三目标比较

每个模型选预登记 checkpoint；在每个 checkpoint 上分别评估 SFT、OPD、RLVR 的梯度：

- 共享 prompt 集，但承认监督答案与 rollout 来源不同；
- 同一 frozen rollout 可以同时构造 OPD/RLVR 的目标贡献；
- SFT 原始答案不强制替换为 student rollout，除非明确另设受控 surrogate；
- objective-only 对照与真实训练范式比较分开；
- 每个梯度记录长度、reward success rate、有效组比例、裁剪比例及归一化规则。

报告 \(R_\Sigma,\mathcal E_\Sigma\)、信号/噪声能量、实际 \(H\) 谱漂移、字典覆盖率及功能代价。用 \(B\in\{8,16,32,64\}\) 作为样本量敏感性起点；这些只是候选采样档位，是否足够由置信区间和成本决定。

**Optimizer 对照。** 对同一个冻结 \(W,G\)，比较明确规定状态下的优化器变换；零状态分析和固定历史分析分别报告。Muon 的输入应是实际动量，weight decay 单列。数学上，给定同一 \(W\) 和采样目标，raw gradient 不依赖之后选用哪个 optimizer；这是控制条件，不是待发现的经验定律。

## E2. 参考功能几何

选择相同 Frobenius 范数的谱/frame 方向，测量：

- reward 或指定目标的方向导数；
- 当前任务分布下的 \(d^\top F_{\rm cur}d\)；
- held-out prompt 分布下的 \(d^\top F_{\rm ref}d\)；
- 梯度采样噪声 \(\operatorname{Var}(d^\top X_b)\)；
- held-out supervised loss 的线性项和曲率；
- 有限步真实 KL、任务收益和能力变化。

Fisher 与 loss Hessian 独立估计，不以其中一个替代另一个。采用 \(\pm\eta d\) 的对称 loss probe 时，一阶/二阶有限差分及步长敏感性都要记录。

对至少三个足够小的步长检查 KL 的二次 scaling；若局部近似失败，该设置不用于支持局部机制。

## E3. 核心预测：条件谱增益是否优于简单指标

对每个 Design-selected 字典，得到 full/frame 候选；独立 Probe 产生预测；独立 Evaluation 测真实有限步收益差。

比较以下预测器：

1. raw spectral energy 或 enrichment；
2. 去噪 spectral signal energy；
3. 对角方向 SNR 的预登记聚合；
4. diagonal Fisher 加权谱能量；
5. block-diagonal surrogate 的条件增益；
6. 保留谱—frame 交叉块的 \(\widehat{\mathcal U}^{\rm cross}\)；
7. 对实际固定候选的配对 Probe 增益估计。

第 7 项是直接局部梯度预测的强基线，用于判断结构性诊断是否还有价值，而不是故意只和弱基线比较。

按模型/任务或完整 training seed 划分训练与验证折，避免同一轨迹的相邻 checkpoint 分散到不同折造成信息泄漏。主指标包括 held-out 增益符号预测、预测误差、门控 regret；相关性作为辅助。

同时报告反向问题 \(\mathcal U_{F\mid\Sigma}\)，避免只测 spectrum 的不对称叙事。比较时匹配候选维数和构建成本。

## E4. 受控因果干预

### E4a. Frame 与 spectrum

从同一 checkpoint 分叉：

- 原始基础优化器；
- additive spectrum-only；
- additive frame-only；
- exact-isospectral baseline；
- PGSU；
- 同维随机候选控制。

分别报告等 Frobenius 步长和等实际参考 KL 两种匹配。等 KL 缩放由独立 Design/validation 数据确定，不使用 Evaluation 找最佳缩放。若某方向局部 KL 近零，保留最大步长上限并标记无法精确匹配。

### E4b. Credit 的可识别对照

不直接把三范式差异归因于密度。对同一 frozen rollout、同一 teacher、同一 checkpoint，取确定的 token 梯度贡献 \(v_t\)，用独立 mask \(M_t\sim\mathrm{Bernoulli}(p)\)：

\[
\widehat G_p=\sum_t\frac{M_t}{p}v_t,\qquad
\mathbb E_M\widehat G_p=\sum_t v_t.
\]

条件于 rollout：

\[
\operatorname{Cov}_M(\widehat G_p)
=\frac{1-p}{p}\sum_tv_tv_t^\top.
\]

这是一个保持目标平均梯度、只引入可控采样方差的实验，检验方法是否对噪声变化作出正确反应。它不把 OPD 变成 RLVR，也不证明 outcome reward 与 dense teacher 的差异只是噪声。

有偏 token 筛选、process reward、teacher/RL 混合另设实验，因为它们通常同时改变目标和噪声。

### E4c. 信号、曲率和噪声的独立合成操纵

在可解析的小模型中分别改变 \(c_\Sigma\)、\(A_{\Sigma F}\)、\(\Gamma\)，保持其它量不变，验证预测是否随正确因素变化。LLM 中无法保证这种完全独立操纵，不作同等因果解释。

## E5. 方法主实验及公平性

先选一个模型和一个未饱和任务验证，再扩展一个架构与一个目标函数；不同时铺开七种新优化器。

最低基线：AdamW、Muon、明确版本的高通 Pion、exact-isospectral baseline、固定开谱/关谱选择、直接 Probe 比较基线。若引用谱保持 Pion，则作为单独算法列出。

每个 optimizer × objective 使用同等超参搜索预算，而非仅在 SFT 上调一次学习率后直接迁移。迁移能力可作为额外独立实验。

主要消融：

- 去掉 Probe 独立性；
- 去掉噪声校正；
- 去掉 \(A_{\Sigma F}\)；
- \(A=I\) 或只保留 diagonal Fisher；
- 去掉实际收益阈值；
- 用 magnitude/SNR 门控替代；
- 把额外 Probe 样本给 baseline 正常训练；
- 字典大小、参考分布和刷新频率；
- 加性 frame 与 exact retraction。

至少 3 个独立 training seeds 用于初步方法比较，但不把“3 seeds”当作充分统计力的保证。确认性样本数基于实际效果阈值与先导方差计算；不足时明确降低结论强度。

## E6. 长期功能保持

任务指标根据训练任务选择 GSM8K/MATH 或相应 coding evaluation；参考能力集必须预登记并与训练和调参隔离。可使用 MMLU、HumanEval、IFEval 等作为不同能力轴，但不把它们机械压成一个无解释的总分。

同时报告：

- 当前策略相邻步的参考 KL；
- 相对 pretrained/base policy 的累计功能偏移；
- held-out loss/accuracy；
- 实际谱漂移及近简并块变化；
- 任务性能随总 wall-clock 和训练 token 的曲线。

若 frame 更新也造成明显遗忘，应如实报告；固定谱没有理论上的全能力保护保证。

---

# 12. 统计、数值稳定性与可复现性

## 12.1 分析单位和置信区间

- 原始模型训练的独立单位是 seed/run；同一 run 的矩阵和 checkpoint 为嵌套重复测量。
- 梯度噪声的单位是冻结 checkpoint 下的独立完整 groups。
- 模型泛化的单位是预登记的模型/任务划分，不能把很多 modes 当成很多独立模型。
- prompt-conditioned 与 prompt-pooled 结果分别报告。
- 用层级汇总或匹配其抽样结构的区间；不把所有相关 mode 拼接后套用独立样本检验。
- 对相关性“相等”的主张，预先给出等效性边界；无法排除有意义差异时结论是未分辨。
- 二次无偏估计的负值、宽区间和无效信号窗口均保留，不为了清晰曲线裁掉。

## 12.2 训练饱和

所有主结果报告完整预登记训练窗口。另按 reward 成功率、非退化组比例定义活跃阶段规则；规则在确认性数据前锁定。

不因观察到 ordering 只在前 100 步出现，就把后续数据从主分析中去除。可以报告探索性的前段结果，但需要新 run 复核。

对全对/全错组分别报告频率和 estimator 处理。奖励饱和引起的低梯度不是“谱方向没有任务信息”的普遍证据。

## 12.3 方向选择与稳定性

固定 \(W\) 下同时翻转 \((u_i,v_i)\) 不改变谱 rank-1 方向；不能将这一点推广到任意跨 checkpoint mode 匹配。

近简并时单向量索引不稳定，使用子空间 principal angles/Procrustes 及块级测量。动量基和权重基使用不同命名与日志字段。

字典和代价在 Probe 外构造；超参或方向数量若经 Probe 选择，应嵌套额外划分或采用适当校正，不能继续使用固定候选的单次保证。

## 12.4 数值计算

使用线性系统求解实现 \(A_{FF}^{-1}c_F\) 和 Schur solve，不显式构造逆矩阵。记录：

- \(A\) 的最小特征值、条件数和 damping；
- \(Q^\top Q-I\) 的范数；
- spectral/frame 投影残差；
- SVD 截断与数值秩阈值；
- Fisher 小矩阵的对称性和 PSD 检查；
- 求解残差及预测步长；
- 真实有限步 KL 对二次预测的误差；
- **投影干预的自洽核查**：对 spectrum_only / frame_only 的每个捕获步，在捕获 \(W_{\rm before}\) 的奇异基下重算 \(R_\Sigma(H)\)，spectrum_only 必须接近 1、frame_only 必须接近 0。作者B 的捕获（§9.4 第 5 条）在 spectrum_only 下仅为基线的 1.3–9 倍。用真实 k_proj 权重做的检查排除了两个解释：bf16 存储 \(W\) 后重算基（读数 ≥0.996）与 engine 基相对捕获步滞后 9 步（AdamW 类低秩相干更新 lr \(10^{-5}\)×9 步读数 0.957）。剩余待查项是作者B 侧 engine.post_step 与 instr.post_optimizer 的调用顺序，以及 tracked 参数对象是否与 engine 投影的对象一致。**2026-09-10 作者A 的 v2 smoke（results/v2/result_A/smoke_v2_sft_*）：spectrum_matched 捕获 H 的 \(R_\Sigma\) 中位数 0.967、frame_matched 1e-5、random_ext 为基线的 1.07 倍，均与理论值一致；A 侧代码路径无此问题，待查项仅剩 B 侧。**

**精度要求（2026-09-10 实测，results/v2/result_A/kl_probe_0.8b_bf16）**：训练循环使用的 bf16 autocast 前向不能用于有限步 KL 或 Fisher 的估计。相对扰动 \(\le 10^{-2}\) 低于 bf16 权重分辨率（ulp \(2^{-8}\)），纯缩放方向的 KL 恰为 0，其它方向的 KL 停在约 \(2.4\times10^{-4}\) nats/token 的舍入底噪且与步长无关。E2、E4a 的等 KL 匹配与 §12.4 的所有 KL 项一律用 fp32 前向并关闭 TF32。

经验 Fisher 小矩阵可由独立 score outer product 得到 PSD 估计，但“采样自模型”与“使用真实标签”必须区分。固定估计矩阵上的代数保证不意味着它准确估计真实参考 Fisher。

## 12.5 最小 run manifest

每个图表可追溯到以下字段：

模型与 tokenizer 精确版本、checkpoint hash、矩阵名称/shape/head 分块、dataset split、prompt IDs、teacher 版本、目标公式、符号约定、token/sequence 归一化、采样参数、group size、零方差组处理、KL/clip 设置、optimizer 及完整状态来源、学习率、seed、训练窗口、独立 probe 分组、字典来源、参考分布、Fisher 估计方式、damping、方向维数、步长匹配规则、硬件和 wall-clock。

确认性图表必须附上独立单位数量和不确定性定义。当前旧数字若无法追溯，只能保留为探索性笔记。

---

# 13. 图表与正文结构

## 13.1 最小主图集合

| 图表 | 回答的问题 | 对应证据 |
|---|---|---|
| Fig.1 | 观测谱能量为何不等于可用谱收益？ | 维数、信号、噪声、功能代价的定义图 |
| Fig.2 | 范式/架构差异中有多少来自信号、多少来自噪声？ | E1 的独立 group 分解与区间 |
| Fig.3 | 谱/frame 是否有不同的参考功能代价？ | E2 的等范数方向测量与有限步检查 |
| Fig.4 | 条件谱增益是否比简单指标更能预测收益？ | E3 的 held-out 预测比较；主机制图 |
| Fig.5 | 谱—frame 耦合是否必需？ | block-diagonal 对照及 §7.3 反例的真实对应 |
| Fig.6 | 门控是否作出可靠决定？ | 错开/漏开谱、实际候选收益、额外样本控制 |
| Table 1 | 方法是否在总预算下有用？ | E5 的性能、参考能力、时间与内存 |

若 Fig.4/5 不支持假设，不用更多优化器主表掩盖机制缺失。

## 13.2 建议的投稿正文顺序

1. Introduction：从谱继承转向“何时值得开谱”。
2. Measurement setup：正确几何、独立 group 和去噪能量。
3. Functional geometry：score/Fisher 联系及参考分布。
4. Conditional spectral utility：块代价、Schur residual、噪声偏差。
5. Independent interventions：预测、耦合消融和有限步证据。
6. Method：只有方法条件成立时加入 PGSU 主实验。
7. Limitations：局部模型、字典覆盖、估计成本及任务边界。

基础恒等式和完整证明可移到附录，正文保留定义、核心公式与反例。文献结果与本项目结果始终分开。

---

# 14. 执行顺序与停止条件

**截止 2026-09-26 前的优先级（2026-09-09 定）。** 阶段 A–D 与 E0–E6 的完整流程超出剩余工期。以下为必做项，其余标为投稿后工作：

1. E4a 的等范数版本：从同一 checkpoint 出发，spectrum-only / frame-only / random-extension 按相同 Frobenius 步长各跑一次（RLVR 与 SFT），这是把 §9.4 的 H5 从“去掉谱无损失”提升为可写主结果的唯一缺口。
2. OPD none 的 MMLU 对照，以及作者B 的代码 commit 与 \(H\) 捕获顺序确认（§9.4 第 3、5 条）。
3. ✅（2026-09-10）E0 中与已有数据直接相关的部分：§5.4 的交叉样本 SNR 已在 0.8B 六格上重算并替换 §9.2 的耦合指标；矩形投影与维数基线见 §4.5。
4. 4B 的 RLVR 行仅在已排队的情况下接收，不再新增规模。

阶段 B、C（条件增益检验与 PGSU）不在本轮投稿范围内。

## 阶段 A：修正测量并复核旧证据

先修矩形投影、统计单位、SNR 命名、generator 对照和 null。生成可追溯 manifest。优先检查旧 ordering 在修正后是否保留，而非直接启动更大规模训练。

完成条件：E0 关键数值校验通过；至少一个真实 checkpoint 的重复采样结果可复现，且不确定性可报告。

## 阶段 B：一个模型上的条件增益检验

选择少量矩阵和小字典，完成 E2/E3。将数据严格分为 Design、Probe、Evaluation，测量真实一小步干预。

完成条件：确认能否可靠区分零增益、正增益和未分辨情形；确定交叉块与噪声校正是否带来额外价值。

若条件增益不优于简单预测器，则降级 C2，不继续以它为依据扩大方法工程。

## 阶段 C：门控可行性与成本

实现 PGSU 研究版，记录全部 probe 成本，与同样额外采样的 baseline 比较。只在预登记验证集上选择刷新频率和阈值。

完成条件：门控有可复现的决策价值，且总体成本有竞争力。否则只保留离线诊断方法，不声称是实用训练优化器。

## 阶段 D：跨架构、目标与训练阶段确认

在新模型或新任务上锁定设计进行测试。只有在前面阶段成立后扩展训练规模。算力与日历工期由实际 profile 决定，不复用旧版本未经测量的天数估计。

## 阴性结果的论文路径

- 若范式 ordering 消失，但信号/噪声校正显著改变既有解释，可形成严格的测量与复现实证论文。
- 若条件增益能预测干预、但门控成本过高，可形成分析论文并报告离线诊断价值。
- 若简单 diagonal Fisher 或 SNR 足够，应报告更简单的规律，不坚持复杂控制器。
- 若校正后所有信号均无法分辨，只能报告测量限制，不能把它写成“谱方向无信息”或“二者完全对称”。

---

# 附录 A. 从旧版到 v2 的实质变化

| 旧版内容 | v2 处理 |
|---|---|
| Credit assignment determines spectral/frame geometry | 改为待检验因素；核心问题是条件谱增益 |
| RLVR 的 frame-dominance 唯一来自 SNR | 删除唯一机制归因，保留 competing explanations |
| raw diagonal 能量作为主要机制证据 | 加入维数基线、独立样本平均梯度与噪声分解 |
| score 方差是无关尺度 | 明确它等于指定方向的 Fisher 敏感度 |
| \(H_{\rm pre}^{-1}g_R\) 代表普通 RL 更新 | 限定为明确局部二次问题，区分 Fisher/Hessian/实际 optimizer |
| 单步 entry symmetry 支持 generator symmetry | 分别定义、分别测量，不互相替代 |
| GRPO 组内项 i.i.d. | 使用完整独立 groups，并明确实际 estimator 均值 |
| SNR-only SSD 为主方法 | 降为基线；主提案为条件增益诊断和独立 Probe 决策 |
| \(\alpha=0\) 等于 ISO | 区分一阶 frame additive 与精确保谱 |
| spectrum-only 不能学新关联 | 限定为单层固定基的可表示交叉项，不作网络能力不可能性断言 |
| M1–M7 同时推进 | 收敛为一个主诊断、一个门控方法和有裁决力的实验 |
| 已证理论与在途实验混写 | 每个数学结论列假设；每个经验主张列当前状态和拒绝条件 |

# 附录 B. 已证明与待验证的清单

**在本文明确条件下可直接证明：**

- spectral/frame 正交分解及简单正奇异值一阶导数；
- 各向同性维数基线；
- 独立 group 平均的信号—噪声能量分解；
- 交叉样本信号能量无偏性；
- score variance/Fisher 和 reward covariance 恒等式；
- 组内均值中心化的有限样本因子；
- 固定正定二次模型下的条件谱增益；
- 固定字典/代价下二次 plug-in 偏差与交叉增益无偏性；
- 有效独立置信界下的固定 surrogate 错误接受控制。

**尚待实证确认：**

- 哪些范式或架构的真实平均梯度更偏离随机谱基线；
- 哪些谱方向在当前任务或参考能力集上代价更高；
- 条件谱增益对有限步/长期收益的增量预测价值；
- 谱—frame Fisher 耦合的实际重要性；
- PGSU 的校准、成本、跨设置收益与能力保持；
- dominant-spectrum、全谱和小字典之间的覆盖关系。

**本文不声称的新颖性：**

SVD 微分、Fisher 定义、baseline 恒等式、Wiener 收缩、Schur complement、交叉二次型及一般置信界均不作为独立数学首创。创新需要由本文特定决策问题的测量设计、可验证预测及干预结果建立。

# 附录 C. 主要参考文献

以下为本版本核查过的主要来源；涉及具体算法公式以链接版本及最终实验锁定版本为准。

1. **ISO: An RLVR-Native Optimization Stack.** [arXiv:2607.19331](https://arxiv.org/abs/2607.19331)
2. **On the Geometry of On-Policy Distillation.** [arXiv:2606.07082](https://arxiv.org/abs/2606.07082)
3. **Dense Supervision, Sparse Updates: On the Sparsity and Geometry of On-Policy Distillation.** [arXiv:2606.13657](https://arxiv.org/abs/2606.13657)
4. **Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR.** 本文的高通 Pion 基线。[arXiv:2605.19282](https://arxiv.org/abs/2605.19282)
5. **Pion: A Spectrum-Preserving Optimizer via Orthogonal Equivalence Transformation.** 与第 4 项同名但不同的算法。[arXiv:2605.12492](https://arxiv.org/abs/2605.12492)
6. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.** [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
7. **Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic.** [arXiv:2603.01162](https://arxiv.org/abs/2603.01162)
8. **A Natural Policy Gradient.** [NeurIPS paper](https://papers.neurips.cc/paper/2073-a-natural-policy-gradient.pdf)
9. **Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients.** [arXiv:1705.07774](https://arxiv.org/abs/1705.07774)
10. **Limitations of the Empirical Fisher Approximation for Natural Gradient Descent.** [arXiv:1905.12558](https://arxiv.org/abs/1905.12558)
11. **Extremely Sparse Supervision Incentivizes Reasoning Ability.** [arXiv:2609.04565](https://arxiv.org/abs/2609.04565)
12. **Muon 官方实现。** [KellerJordan/Muon](https://github.com/KellerJordan/Muon)

本版本的文献核查用于明确最接近工作和避免明显重复，不构成“不存在相同想法”的穷尽性证明。提交前应围绕 conditional subspace utility、noise-aware trust regions、adaptive parameter subspaces 及 spectral optimization 做进一步定向比较，并按结果收紧创新表述。

# 附录 D. 本次重写的数值校验范围

2026-09-09 对本文关键公式做了独立的小规模数值 sanity checks：随机正定块系统的条件增益与直接求解一致；两个反例满足所述结论；高/宽矩形及方阵的投影和一阶 SVD 导数通过有限差分检查；有限 group 中心化因子通过枚举验证；softmax 模型中的 score/Fisher 恒等式及 entry/generator 区别得到验证；各向同性谱基线、交叉条件增益期望和 plug-in 噪声偏差通过 Monte Carlo 检查。

这些检查验证公式在选定有限算例中的行为，并辅助发现实现错误。它们不是原项目 LLM 实验复算，也不代表完整 E0、重尾置信校准或任何 PGSU 性能实验已经完成。本文“尚待验证”的状态不因这些校验而改变。
