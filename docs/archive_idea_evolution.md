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
