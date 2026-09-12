# When Should Post-Training Change the Spectrum?
## Is the Weight SVD Basis a Privileged Coordinate System for LLM Post-Training?

**中文题目:后训练何时值得改变权重谱?——权重奇异基是否是后训练的特权坐标系**

版本:v3,2026-09-11(第三稿,已生效。第二稿以"双重分离"为主线,经复核 dense 基线为崩溃态后改写;见 §2.5)
文档类型:统一研究文档。只写 v3 相对 v2 的实质变化、证据总账、核心问题、理论骨架与实验裁决协议;
v2 §3–§7 中可证明的测量框架(符号、outer-product 恒等式、维数基线、信号/噪声分解、交叉样本估计)原样继承,不重复。
状态:**生效(用户 2026-09-11 确认)。** v2 已冻结;代码 `*_v3/`,结果 `results/v3/`,任务清单 `TASKS_A_v3.md` / `TASKS_B_v3.md`。§3.2 于 09-12 按 E-v3-0c 结果改写。
数字的唯一来源是 `paper/results_draft.md`(v2 登记处,2026-09-11 已登记 B 的全部 v2 交付)。

---

# 前一版本(v2)的状况(一段话)

- **v2 主线**:frame 更新可用时,额外开放 spectrum 方向是否有不可替代的增益(C1 测量、C2 条件谱增益、C3 门控)。
- **成立**:C1(观测梯度几何由噪声主导,v1 的谱 ordering 与 H4 是耦合产物);A5(谱/frame 单位步长 KL 代价无全局符号)。
- **推翻**:v2 主线。P3 显示奇异基字典 ≈ 同维随机字典,"开放 spectrum"没有独立价值,C2 降附录、C3 删除。
  v1 B 侧全部训练结局因 bf16 master 作废。
- **意外发现**:r 维更新保能力、mn 维更新崩 MMLU。但 mn 维(dense)run 在 lr×1 下处于崩溃态(§2.5),以它为锚的比较不可靠。
- **主线是否还成立:不成立。** v3 以"奇异基不特殊"为可靠主线,以"r 维约束是否改善折中"为待裁决的第二层。逐项证据见 §2.0。

---

# 0. 核心问题、主张与证据边界

## 0.1 核心问题

> **权重的奇异基是否是 LLM post-training 的特权坐标系?**
> 即:把更新限制在谱方向、去掉谱方向、精确保谱、或换成同维随机方向,在匹配步长后,任务收益与能力保持是否有可测差别?
> 若没有,决定折中的变量是什么(自由度、推进量)?

谱继承、ISO 类保谱训练与 SVD 型 PEFT(PiSSA、MiLoRA、SVFT)都隐含"奇异基有特殊地位"或"保谱即保能力"。v3 直接检验。

## 0.2 主张,按可靠性分层

**第一层(现有数据已可靠,v3 只加 seed 与档位):**

| 编号 | 主张 | 证据 | 为何可靠 |
|---|---|---|---|
| C1 | 奇异基下可复现平均梯度的谱能量处于随机字典基线;观测谱几何由采样噪声主导 | A1 六格;P2 B 侧 24 run(G 侧) | 30 个 run,fp32,交叉样本估计无偏 |
| C2 | **基不特殊(r 维)**:奇异基字典与同维随机秩一字典在任务收益与能力保持上不可分 | P3:RLVR 0.634/0.474 vs 0.593/0.475;SFT 0.387/0.478 vs 0.393/0.466 | 两范式 × 2 seed;r 维 run 在健康区(reward 单调、seed 差 ≤0.03) |
| C3 | 单位步长 KL 代价无全局谱/frame 符号 | A5 | fp32,局部二次区确认 |

**第二层(需先建立健康的 dense 基线,再裁决):**

| 编号 | 主张 | 现状 | 需要什么 |
|---|---|---|---|
| C4 | **基不特殊(mn 维)**:frame_matched、exact_iso 与 full 在健康区不可分,即"去掉谱变化"与"精确保谱"都不改变结局 | lr×1 下三者都崩溃,崩溃态下不可分(MMLU 0.23–0.26)但 GSM8K seed 差 0.08–0.11 | 在 lr\*(§5 批次一定义)上重跑三者,各 ≥2 seed |
| C5 | **r 维约束改善折中**:在(推进量,MMLU)前沿上,r 维字典优于 dense 的 lr 前沿 | 只在 lr×1 一点成立,而该点 dense 为崩溃态 | lr 扫描 + s_rel 扫描画前沿(§5 批次二) |

**第三层(协议与数值贡献,无需新实验):**

| 编号 | 主张 |
|---|---|
| C6 | 等 Frobenius 步长 ≠ 等推进(§3.1);等范数单点比较的混淆及其修正(前沿协议) |
| C7 | bf16 master 精度事故:更新被舍入 → 训练"有效且不遗忘"的假象;fp32 后同 lr 的 dense 崩溃 |

**论文的底线与上限。** 底线 = C1 + C2 + C3 + C6 + C7,再加 C4 在 lr\* 上成立:一篇"奇异基不是后训练关心的轴"的阴性结果论文,所有部件现在都有数据或只差 seed。
上限 = 底线 + C5 为正:多一个"逐矩阵 r 维秩一约束在等推进下改善折中"的正结果。**v3 的实验按先锁底线、再试上限排序。**

## 0.3 明确不作的推断

- 奇异基 ≈ 随机字典只在本文设置(0.8B、GSM8K、300 步、逐矩阵 r 维秩一字典)内主张。
- lr×1 下 dense 崩溃、r 维不崩溃,不写成"r 维约束防遗忘",直到 C5 在健康区裁决。
- MMLU 降到随机在格式检查(E-v3-0b)前不写成"知识丢失"。
- SFT 在本文数据(GSM8K 短解)上降低 GSM8K 是数据格式效应;SFT 行只度量"损伤"。
- 不提出新优化器、门控或 PEFT 方法;不把 v1 的任何训练结局当证据;不把 §3.3 的猜测写进主张。

---

# 1. 与相关工作的可检验差异

- **谱继承 / ISO 类保谱训练**:主张后训练应保谱或天然保谱。本文 exact_iso(每步把奇异值精确重置为预训练谱)在健康区若与 full 不可分,则保谱既不帮助也不损害;在崩溃区二者同崩。
- **SVD 型 PEFT(PiSSA、MiLoRA、SVFT、只训奇异值)**:主张奇异基是好的适配坐标。本文 random_ext(随机正交秩一字典,同维)与 spectrum_matched 不可分 → 收益来自约束本身而非基。
- **"LoRA learns less, forgets less" 与 intrinsic dimension**:与 C5 同向。差异:本文逐步等范数并扫描步长画前沿,字典是逐矩阵 r 维秩一方向而非低秩乘积,并有精确保谱对照。C5 若为阴性,即"逐步等范数下 r 维约束不优于调 lr 的 dense",同样是对该文献的有用边界。

---

# 2. 已有证据(v1 fp32 部分 + v2)

## 2.0 v2 逐项判定

| | 结论 |
|---|---|
| **成立** | C1 测量框架:B=8 下可复现梯度能量 <1% 噪声,v1 的 G 侧谱几何与 H4 是耦合产物(A1;B 侧 24 run 复现)。A5 fp32 KL 探针:谱/frame 方向单位步长功能代价逐矩阵各向异性(0.26–5.2),无全局符号。P3 等范数干预(fp32,两范式 × 2 seed):奇异基字典 ≈ 同维随机字典。 |
| **推翻** | v2 主线本身:P3 显示"开放 spectrum"能做到的,同维随机字典同样能做到,C2 要预测的增益在该设置下为零 → 条件谱增益降附录,门控删除。v1 B 侧全部训练结局(Phase 2 H5、E1–E3 SSD、E4 α 的训练结局、4B):B 机器 transformers 5.9.0 忽略 dtype,master 权重为 bf16,更新大半被舍入;v1 的 RLVR full(0.636/0.477)与 fp32 的 spectrum_matched(0.634/0.474)几乎相同,说明 v1 运行在隐式小步长区间,数字不可与 fp32 混排。 |
| **P3 给出但需重新解读的事实** | mn 维更新(full、frame_matched、exact_iso)在 RLVR 与 SFT 上都把 MMLU 打到随机(0.23–0.26);r 维更新(spectrum_matched、random_ext)保住 MMLU(≈0.47)且 GSM8K 更高。**复核后:mn 维三种在 lr×1 是崩溃态(§2.5),这组对比不能作为主张的锚点。** |
| **v2 协议的缺陷** | 等 Frobenius 步长 ≠ 等推进:投影后放大 s≈50 倍的更新与原更新余弦为 1/s≈0.02(§3.1);且 dense 只测了一个 lr,而该 lr 是崩溃态。 |
| **主线是否还成立** | **v2 的主线不成立。** v2 §14 阴性路径第 1、3 条即为 v3 的论文路径。 |

## 2.1 测量(C1)

A1:0.8B 六格,B=8,交叉信号能量/噪声能量 0.0004–0.0073;R_enrich 0.84–0.89(观测 G 的谱能量略低于随机基线);
split-half Spearman ≈ 0.06 而同样本 Spearman ≈ 0.85 → v1 H4 为耦合产物。P2 在 B 的 24 个 run 上复现 G 侧。

## 2.2 功能代价(A5)

fp32 前向、TF32 关闭,η∈[1e-3,1e-2] 四类单位范数方向斜率 2.01 → 局部二次成立;谱/frame KL 代价比逐矩阵 0.26–5.2,中位 0.95;
q_proj 谱步便宜 3 倍,o_proj 谱步贵 2 倍;k_proj 纯缩放方向 KL≈0。

## 2.3 等范数干预(P3 + P4;lr×1)

| 干预 | 改变什么 | 自由度 | RLVR GSM8K / MMLU | SFT GSM8K / MMLU | 区间 |
|---|---|---|---|---|---|
| base | — | — | 0.546 / 0.483 | 0.546 / 0.483 | — |
| full | 谱 + frame | mn | 0.453 / 0.234 | 0.301 / 0.239 | 崩溃 |
| frame_matched(scale 1.00) | frame(去掉谱分量,能量 −0.04%) | mn−r | 0.528 / 0.238 | 0.283 / 0.236 | 崩溃 |
| exact_iso(1 seed) | frame,谱精确固定为预训练谱 | mn−r | 0.600 / 0.255 | 0.290 / 0.240 | 崩溃 |
| spectrum_matched(scale 50) | 只改奇异值(当前步精确基,U/V 不变) | r | 0.634 / 0.474 | 0.387 / 0.478 | 健康 |
| random_ext(scale 52) | r 个固定随机秩一方向 | r | 0.593 / 0.475 | 0.393 / 0.466 | 健康 |

可靠的读法只有两条:(i)r 维两行不可分(C2);(ii)mn 维三行在崩溃态下也不可分(C4 在崩溃区的弱形式)。
"r 维 vs mn 维"的差异是"健康 vs 崩溃"的差异,不是干预本身的差异,除非 C5 在 lr\* 上复现。

SFT 训练 loss:五种模式从第 51 步起逐窗相同(0.53 → 0.49–0.51)。这说明 loss 在 100 步内到达该数据的底(格式学完),
之后 loss 不再区分模式;它不能用作"推进量"指标,SFT 的推进量改用 GSM8K-SFT 格式准确率或不定义(§4 裁决规则)。

## 2.4 精度事故(v1)

B 机器 master 权重 bf16,每步更新元素(~1e-5)与量化步长(~2e-5)同量级。v1 的 full 结局与 fp32 的 r 维干预几乎相同。
写进方法论警示(C7),不作证据。

## 2.5 dense 在 lr×1 是崩溃态(2026-09-11 复核,来自 P3 日志)

| 指标(RLVR,末 50 步) | full / frame_matched / exact_iso | spectrum_matched / random_ext |
|---|---|---|
| 训练 reward 走势 | 101–150 步达峰 0.59–0.63 后回落到 0.41–0.52 | 单调上升到 0.53–0.58 |
| 有效 group 数(8 个中 reward 不全同) | 3.5–4.4 | 4.8–5.5 |
| 策略梯度 loss 幅度 | 0.04 → 0.01(衰减) | 稳定 0.04 |
| 两 seed 最终 GSM8K 之差 | 0.08–0.11 | 0.01–0.03 |
| MMLU | 0.23–0.26(随机) | 0.47 |

on-policy 训练中 reward 达峰后持续回落、有效 group 减少、seed 间结局发散,是策略退化的标准信号。SFT 侧 loss 不发散,
但 MMLU 同样降到随机、GSM8K 掉到 0.30,与 RLVR 同型。**结论:lr×1(RLVR 2e-6,SFT 1e-5)对 fp32 dense 训练过大;
v3 的一切 dense 比较必须先找到健康的 lr\*。** 这也解释了 v1(bf16 隐式截断)为何"看起来正常"。

---

# 3. 理论骨架(新增部分;其余继承 v2)

## 3.1 等范数投影的一阶算术(C6)

设 P 为到 r 维秩一字典的正交投影,Hp = s·P(H),‖Hp‖_F = ‖H‖_F ⇒ s = ‖H‖/‖PH‖(实测 ≈50,能量比 ≈ r/(mn))。
cos(Hp,H) = 1/s;沿 H 的推进 ⟨Hp,H⟩/‖H‖² = 1/s。令 s_rel = s/s_match:步长范数 = s_rel‖H‖,沿 H 推进 = s_rel/s_match。
**结论:等 Frobenius 步长不是等一阶推进;单点等范数比较混淆了推进量。** 修正:对每个更新族扫描步长(dense 扫 lr,r 维扫 s_rel),在前沿上比较。

## 3.2 单步功能步长:等范数投影更新的 KL 比 full 小两个量级(E-v3-0c,2026-09-12 已测)

原稿假设"A5 表明单步单位范数 KL 在谱/frame 方向无全局差异,故随机游走下 r 维与 mn 维的累计 KL 相同"。
E-v3-0c 在 SFT smoke 第 8 步直接测实际更新:full 的单步 KL 6.5e-3 nats/token;spectrum_matched(s_rel=3,范数 3.8 倍)1.3e-4;
random_ext(s_rel=1)5.1e-5。**投影更新的功能步长小 50–130 倍,单位范数平方的代价小 300–700 倍。**
A5 的等代价只对随机方向成立;实际 Adam 步把能量集中在高曲率的梯度方向,投影后保留的是低曲率分量。

因此 §3.1 的一阶算术低估了差距的性质:r 维干预不只是沿 H 的推进小 50 倍,它每步对函数的扰动也小两个量级。
**这把 M1(有效步长)推到首位**,并给出定量预测:KL ∝ lr²,dense 在 lr×1/10 附近的单步 KL 才与 r 维干预相当,
所以批次一的 lr 扫描 {1/3, 1/10, 1/30} 正好跨过这个点;H-4 若成立,lr\* 预计在 1/10 档。
累计 KL(E-v3-0a)仍要测:它回答的是"300 步后的总功能位移",与单步 KL 一起构成 Fig.3 的账目。
第 4 步复核与 RLVR 捕获上的复测列入 A3。

## 3.3 开放问题(不作主张)

r 维干预对任意固定方向的相干推进都削弱 s_match 倍,却在 300 步内学会任务(RLVR reward 0.35→0.58)。这说明任务学习不是沿固定方向的一阶推进。
v3 不对此建模;只记录为开放问题,供 C5 的结果解释时引用。

---

# 4. 假设表与预先锁定的裁决规则

| 假设 | 预测 | 拒绝或降级条件 |
|---|---|---|
| H-0 存在健康 lr\* | lr ∈ {1/3, 1/10, 1/30}×lr×1 中存在最大的一档,使 dense 300 步 reward 不回落(末 100 步均值 ≥ 峰值 −0.03)、MMLU ≥ base −0.03、两 seed GSM8K 差 ≤0.04 | 三档都不满足 → 训练配方本身要改(加 KL 惩罚或加大 batch),批次二暂停;这是 v3 的第一道闸 |
| H-1 基不特殊,r 维(C2) | s_rel=1、3 seed 下 spectral 与 random 的 GSM8K、MMLU 差 < margin(0.03 / 0.02);s_rel=3 上同样 | 任一档位差 > margin 且跨 seed 同向 → C2 改条件性,用 A5 逐矩阵代价做关联 |
| H-2 基不特殊,mn 维(C4) | lr\* 上 full、frame_matched、exact_iso 两两差 < margin(2 seed) | exact_iso 或 frame_matched 与 full 差 > margin → "保谱/去谱在健康区有可测效应",C4 改写,论文主线随之改 |
| H-3 r 维约束改善折中(C5) | 以 GSM8K 为横轴、MMLU 为纵轴,r 维前沿(s_rel ∈ {1,3,10})严格在 dense 前沿(lr ∈ {lr\*, lr\*×3, lr×1})右上方,且两字典一致 | 前沿重合或交叉 → C5 阴性,写为"逐步等范数下 r 维约束不优于调 lr",论文取底线 |
| H-4 MMLU 崩溃含格式漂移 | lr×1 dense ckpt 在答案位置 {A,B,C,D} 概率质量 < base 一半 | 质量不变 → 崩溃为真实能力损失;否则主文改报 generation-based MMLU |
| H-5 相干漂移(§3.2) | 崩溃区 dense 的累计位移/累计 KL 相对 √T·单步值的比 ≥ 3× r 维;健康区各族相同 | 不成立 → §3.2 只作阴性报告 |

裁决规则(数据解封前锁定):
- 所有比较用 seed 合并均值,报告二项 SE(GSM8K 500 题 0.022,MMLU 1000 题 0.016);"不可分"= 差 < margin 且 |t| < 2。
- 前沿以 300 步终点为主;每 50 步中间 ckpt 画轨迹,不作独立样本。
- 推进量指标:RLVR 用 GSM8K(训练 reward 只作健康性判据);SFT 不定义推进量,只比较(GSM8K, MMLU)。
- MMLU 主指标 letter-logit;H-4 成立则主文改 generation-based(200 题),letter-logit 进附录。
- non-inferiority 主张按 margin 报告。

---

# 5. 实验设计(作者B;0.8B,fp32,300 步,其余参数同 P3;每 50 步存 ckpt 并评测 GSM8K 500 / MMLU 1000)

原则:**小批次、先闸门、看到结果再开下一批。** 每批一天内能出结果。

## 批次一(闸门,第 1 天):健康 dense 基线 + 零训练诊断

- **E-v3-0a 累计位移与累计 KL**(不训练):18 个 e4a run 的 ckpt_000300 对 base,逐矩阵 ‖ΔW‖_F 与 KL(base‖ckpt)(A5 的 32 条参考序列,fp32,TF32 关)。A 给脚本。
- **E-v3-0b MMLU 格式检查**(不训练):base、e4a_rlvr_full_s0、e4a_rlvr_spectrum_matched_s0、e4a_sft_full_s0、e4a_sft_spectrum_matched_s0,200 题答案位置 {A,B,C,D} 概率质量与 20 条生成。
- **E-v3-1 dense lr 扫描**:full,lr × {1/3, 1/10, 1/30} × {RLVR, SFT} × 2 seed = **12 run**。SFT 便宜,RLVR 6 个约 3 h/卡。
- 闸门判定(H-0):定 lr\*。若无 lr 通过,改配方(优先加 KL 惩罚 β=0.01 或 prompts_per_step 8→32),重扫一次。

## 批次二(主结果,第 2–3 天,在 lr\* 上)

- **E-v3-2 mn 维三种在 lr\***:frame_matched、exact_iso × {RLVR, SFT} × 2 seed = **8 run**(full 已在批次一)。→ H-2 / C4。
- **E-v3-3 r 维两种在 lr\***:spectrum_matched、random_ext,s_rel=1 × {RLVR, SFT} × 2 seed = **8 run**。→ 与批次一 dense 同 lr 的等范数对照;H-1 在第二个 lr 上复现。
- **E-v3-4 r 维加 seed(lr×1)**:spectrum_matched、random_ext,seed 2 × {RLVR, SFT} = **4 run**。→ H-1 达 3 seed。

## 批次三(前沿,第 3–4 天,视批次二结果)

- **E-v3-5 s_rel 扫描**:spectrum_matched、random_ext,s_rel ∈ {3, 10},lr\*,RLVR,1 seed = **4 run**;SFT 视 RLVR 结果加。→ H-3 / C5。
- **E-v3-6 dense 中间档**:full,lr\*×3,RLVR,2 seed = **2 run**(补前沿点)。
- **E-v3-7 第三范式(可选)**:OPD full(lr\* 同比例)、spectrum_matched、random_ext,1 seed = **3 run**。
- **E-v3-8 第二模型(可选)**:Llama-3.2-1B(v1 管线现成),RLVR lr\* 同比例,full、spectrum_matched、random_ext、exact_iso,1 seed = **4 run**。

## 预算

批次一 12 run + 诊断;批次二 20 run;批次三 6–13 run。RLVR 约 2.5–3.5 h/卡,SFT 0.5–1.2 h/卡。8 卡并行:批次一 1 天、批次二 1.5 天、批次三 1 天。
每批结束前 A 出裁决,再开下一批。

---

# 6. 作者A 的工作

- **代码 `specgeom_v3/ scripts_v3/ analysis_v3/ slurm_v3/`(从 v2 复制后叠加):**
  - `intervene_engine.py`:`--intervention-scale s_rel`;smoke 打印 cos(Hp, H_full) 与 ‖Hp‖/‖H‖。
  - `instrument.py`:捕获中 W 存 fp32。
  - `train.py`:`--eval-every 50`(存 ckpt,调用两个评测脚本,写 `eval_step{n}.json`);`--kl-beta`(批次一闸门失败时的备选)。
  - `analysis_v3/cum_kl.py`(E-v3-0a)、`analysis_v3/mmlu_format.py`(E-v3-0b)、`analysis_v3/step_kl.py`(在 A 的 smoke 捕获上算单步 KL(W_0+H) 与 KL(W_0+Hp),A 侧一次 GPU 前向)。
  - `analysis_v3/health.py`:从 log.jsonl 自动算 H-0 的三个健康判据,批次一交付当天出 lr\*。
- **分析与图:**
  - Fig.1 前沿:(GSM8K, MMLU)× {RLVR, SFT}:dense lr 线、spectral 与 random 各一条 s_rel 线、frame_matched 与 exact_iso 点。
  - Fig.2 基不特殊:六种更新在 lr\* 与 lr×1 两个档位的(GSM8K, MMLU)带 SE。
  - Fig.3 崩溃区诊断:reward、有效 group、seed 发散(§2.5)与累计位移/KL 比(E-v3-0a)。
  - Fig.4 KL 代价各向异性(A5,已有);Fig.5 信号/噪声分解(v2 Fig.C/D,已有)。
- **写作:** 按 §8;v2 §3–§7 压缩为"测量框架"一节 + 附录。

---

# 7. 数值与工程要求(继承 v2 §12.4)

1. master fp32,加载后断言(已在 680fa78);交付 README 逐 run 核实 ckpt safetensors 为 F32。
2. 捕获 G、H、W 全 fp32(v3 起);H 自检目标 ≥0.99。
3. 所有 KL 测量 fp32 前向、TF32 关闭。
4. 交付 README 必须写:transformers 版本、代码 commit、秒/步、评测口径、同口径 base;每 run 的 log.jsonl 增加 `resp_len_mean`、`trunc_frac`(RLVR)。
5. 所有 run 共享同一 prompt 流;seed 只改初始化与采样;random_ext 的字典 seed 与训练 seed 分开记录。
6. 评测:GSM8K 500 题 greedy pass@1(含 stop-ids 修复);MMLU 1000 题 letter-logit;H-4 决定是否加 generation-based MMLU。

---

# 8. 论文结构与主图

题目候选:
1. *The SVD Basis Is Not Special: Spectral, Frame, Isospectral and Random Rank-One Updates Are Interchangeable in LLM Post-Training*
2. *When Should Post-Training Change the Spectrum? It Doesn't Matter*

正文顺序(底线版本;C5 为正时加第 5 节):
1. 问题与前提:谱继承 / ISO / SVD-PEFT 隐含"基有特权"或"保谱即保能力";我们直接检验。
2. 测量框架与 C1:可复现梯度在谱方向不富集;观测谱几何由噪声主导。
3. 功能代价(C3):单位步长 KL 无全局谱/frame 符号。
4. 干预(C2、C4):r 维两字典不可分;mn 维三种在健康区不可分、在崩溃区同崩。
5. (条件)r 维约束的前沿(C5)。
6. 协议与数值(C6、C7):等范数 ≠ 等推进;bf16 master 事故与 fp32 下的 lr 重标定。
7. 讨论与局限:单模型单任务;逐矩阵 r 维秩一字典的特殊性;§3.3 开放问题。

---

# 9. 时间线与停止条件

| 天 | A | B |
|---|---|---|
| D0(09-12) | v3 代码 + smoke + TASKS_A/B_v3;`cum_kl.py`、`mmlu_format.py`、`health.py` 交 B | 批次一开跑;E-v3-0a/0b 交付 |
| D1(09-13) | 批次一裁决(lr\*、H-4、H-5 初判);§0–§3 初稿 | 批次一交付;批次二开跑 |
| D2–D3 | 批次二裁决(H-1、H-2);决定批次三范围 | 批次二交付;批次三开跑 |
| D4–D5 | 批次三裁决(H-3);全部登记、出图 | 批次三交付、README |
| D6 起 | 正文 | 复核 |

停止条件:
- H-0 失败(无健康 lr):停批次二,改配方重扫;这一步不跳过。
- H-2 被拒(exact_iso 或 frame_matched 在 lr\* 上与 full 可分):主线改为"保谱/去谱在健康区的可测效应",批次三改为该效应的 seed 与档位复现。
- H-3 阴性:取底线论文;不再加 OPD/Llama 之外的实验。
- H-1 被拒:C2 改条件性,补 A5 关联分析;底线仍成立(C1、C3、C4、C6、C7)。

---

# 附录 A. v1 / v2 结果的处置

| 数据 | 处置 |
|---|---|
| v1 Phase 1 六格 G 侧(A,fp32)+ Llama 六格 | C1 主证据(经 A1 分半重算);Llama 作附录复现 |
| v1 Phase 1 H 侧(A,fp32) | 可用,无新主张,不进主文 |
| v1 B 侧全部训练结局 | 附录"精度事故"一节一张表,标明 bf16 master、不可与 fp32 混排 |
| v1 α 收敛数值 | 对角 SNR 极低的旁证,C1 一节脚注 |
| v2 A1、A5、smoke | 主文 |
| v2 P3、P4 | 主文:C2 的 lr×1 点;C4 的崩溃区弱形式;Fig.3 的崩溃诊断 |
| v2 P1、P2 | P1(OPD full MMLU 0.247)进可选第三范式的动机;P2 G 侧作为 C1 在 B 侧复现进主文一句 |

# 附录 B. v2 → v3 的实质变化

- 核心问题从"何时开放谱方向"改为"奇异基是否是特权坐标系"。
- 主张分三层:可靠(C1、C2、C3)、待裁决(C4、C5)、协议(C6、C7);论文有底线与上限两个版本。
- 发现 dense 在 lr×1 为崩溃态(§2.5);所有 dense 比较改在健康 lr\* 上进行,先过闸门再做主结果。
- 干预协议从单点等 Frobenius 改为前沿比较(dense 扫 lr、r 维扫 s_rel)。
- 新增理论预测 §3.2 与其检验;§3.3 只作开放问题。
- 实验按"小批次、先闸门、看结果再开下一批"组织,每批一天。
- 数值要求新增:捕获 W 存 fp32;日志加响应长度与截断率;README 核实 ckpt dtype。
