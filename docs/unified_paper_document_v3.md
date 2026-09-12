# When Should Post-Training Change the Spectrum?
## Changing the Spectrum Is Neither Necessary Nor Sufficient: A Double Dissociation Between Spectral Motion and Forgetting

**中文题目:后训练何时值得改变权重谱?——谱变化与遗忘的双重分离**

版本:v3 草案,2026-09-11(重写稿;首稿同日上午,已被本稿替换)
文档类型:统一研究文档。只写 v3 相对 v2 的实质变化、证据总账、核心问题、理论骨架与实验裁决协议;
v2 §3–§7 中可证明的测量框架(符号、outer-product 恒等式、维数基线、信号/噪声分解、交叉样本估计)原样继承,不重复。
状态:草案,待用户确认后生效。生效后 v2 冻结(只改引用),代码进 `*_v3/`,结果进 `results/v3/`,任务清单为 `TASKS_A_v3.md` / `TASKS_B_v3.md`。
数字的唯一来源是 `paper/results_draft.md`(v2 登记处,2026-09-11 已登记 B 的全部 v2 交付)。

---

# 前一版本(v2)的状况

**v2 的主线是"frame 更新可用时,额外开放 spectrum 方向是否有可靠、不可替代、值得功能代价的增益"(v2 §0.1),
提出 C1 测量框架、C2 条件谱增益、C3 probe 门控。经 2026-09-10/11 的 A1/A2/A5 与 B 的 P0–P5 交付检验:**

| | 结论 |
|---|---|
| **成立** | C1 测量框架:B=8 下可复现梯度能量 <1% 噪声,v1 的 G 侧谱几何与 H4 是耦合产物(A1;B 侧 24 run 复现)。A5 fp32 KL 探针:谱/frame 方向单位步长功能代价逐矩阵各向异性(0.26–5.2),无全局符号。P3 等范数干预(fp32,两范式 × 2 seed):奇异基字典 ≈ 同维随机字典。 |
| **推翻** | v2 主线本身:P3 显示"开放 spectrum"能做到的,同维随机字典同样能做到,C2 要预测的增益在该设置下为零 → C2 降附录,C3 删除。v1 B 侧全部训练结局(Phase 2 H5、E1–E3 SSD、E4 α 的训练结局、4B):B 机器 transformers 5.9.0 忽略 dtype,master 权重为 bf16,更新大半被舍入;v1 的 RLVR full(0.636/0.477)与 fp32 的 spectrum_matched(0.634/0.474)几乎相同,说明 v1 运行在隐式小步长区间,数字不可与 fp32 混排。 |
| **v2 没预设、P3 给出的新事实** | mn 维更新(full、frame_matched、exact_iso)在 RLVR 与 SFT 上都把 MMLU 打到随机(0.23–0.26);r 维更新(spectrum_matched、random_ext)保住 MMLU(≈0.47)且 GSM8K 更高。精确保谱(exact_iso)不防遗忘;只改谱(spectrum_matched)不引起遗忘。 |
| **v2 协议的缺陷** | 等 Frobenius 步长 ≠ 等推进:投影后放大 s≈50 倍的更新与原更新余弦为 1/s≈0.02(§3.1)。但 SFT 的训练 loss 从第 51 步起五种模式完全相同(§2.3),所以"r 维干预只是步长小"不能解释全部;需要 v3 的前沿实验分开步长与自由度。 |
| **主线是否还成立** | **v2 的主线不成立,改写为 v3 主线。** v2 §14 预先写好的阴性路径第 1、3 条("ordering 消失但测量校正改变既有解释""简单规律优于复杂控制器")即为 v3 的论文路径。 |

---

# 0. 核心问题、主张与证据边界

## 0.1 核心问题

> **权重的奇异基是否是 LLM post-training 的特权坐标系?谱的变化与通用能力的丢失是否有因果联系?
> 若二者都否,决定"任务收益—能力保持"折中的是什么?**

谱继承、ISO 类保谱训练与 SVD 型 PEFT(PiSSA、MiLoRA、SVFT)都隐含"奇异基有特殊地位"或"保谱即保能力"。v3 直接检验这两个前提。

## 0.2 主张(按证据强度排序)

| 编号 | 主张 | 现有证据 | v3 需补 |
|---|---|---|---|
| C1 | 在权重奇异基下,可复现平均梯度的谱方向能量处于随机字典基线,观测到的谱几何由采样噪声主导(v2 C1 原样保留) | A1、P2 G 侧;30 个 run | 无(fp32 W 捕获后的 H 侧复核为可选) |
| C2 | **双重分离**:只改谱(spectrum_matched)学会任务且不遗忘;精确保谱(exact_iso)与 full 一样遗忘。谱变化既非遗忘的必要条件,也非充分条件 | P3:RLVR 0.634/0.474 vs exact_iso 0.600/0.255;SFT 0.387/0.478 vs 0.290/0.240 | exact_iso 加 seed;两者在等推进点上的复核(E-v3-2、E-v3-4) |
| C3 | **基不特殊**:同维随机秩一字典与奇异基字典在任务收益与能力保持上不可分;单位步长 KL 代价无全局谱/frame 符号 | P3(2 seed);A5 | 第 3 个 seed;至少两个步长档位上复现(E-v3-1、E-v3-3) |
| C4 | **决定折中的是自由度与推进量,不是谱/frame**:在(训练目标推进,MMLU)平面上,r 维字典的前沿与 dense 的学习率前沿的关系由 E-v3 裁决;结果两个方向都可报告 | SFT 训练 loss 等推进下遗忘不等(§2.3);RLVR reward 曲线 | E-v3-1、E-v3-3 前沿;E-v3-0 位移/KL 账目;E-v3-5 字典大小 |
| C5 | 协议与数值贡献:等范数 ≠ 等推进的算术;bf16 master 的精度事故及其对"训练有效"判断的影响 | §3.1;v1/v2 对照 | 无 |

## 0.3 明确不作的推断

- 奇异基 ≈ 随机字典只在本文设置(0.8B、GSM8K、300 步、r 维秩一字典)内主张,不推广到所有低秩方法或规模。
- "r 维更新不遗忘"在 C4 裁决前不写成"低自由度防遗忘"的普遍规律。
- MMLU 降到随机在格式检查(E-v3-0b)前不写成"知识丢失"。
- SFT 在本文数据(GSM8K 短解)上降低 GSM8K 是数据格式效应,SFT 行只度量"损伤",不度量"任务收益"。
- 不提出新优化器、门控或 PEFT 方法;不把 v1 的任何训练结局当证据。

---

# 1. 与相关工作的可检验差异

- **谱继承 / ISO 类保谱训练**:主张后训练应保谱或天然保谱。本文 exact_iso(每步把奇异值精确重置为预训练谱)在两范式上与 full 同样遗忘 → 保谱不是能力保持的机制。
- **SVD 型 PEFT(PiSSA、MiLoRA、SVFT、只训奇异值)**:主张奇异基是好的适配坐标。本文 random_ext(随机正交秩一字典,同维)与 spectrum_matched 不可分 → 收益来自约束本身而非基。
- **"LoRA learns less, forgets less" 与 intrinsic dimension**:与 C4 同向。差异:本文按每步 Frobenius 范数匹配并扫描步长画前沿,而非各自调 lr;字典是逐矩阵 r 维秩一方向,不是低秩乘积;并给出精确保谱对照。
- **"RL forgets less than SFT"类主张**:本文等范数下 RLVR full 与 SFT full 的 MMLU 同为随机(0.234 / 0.239),只在讨论中提及,不作主张(lr 与每步范数不同)。

---

# 2. 已有证据(v1 fp32 部分 + v2)

## 2.1 测量(C1)

A1:0.8B 六格,B=8,交叉信号能量/噪声能量 0.0004–0.0073;R_enrich 0.84–0.89(观测 G 的谱能量略低于随机基线);
split-half Spearman ≈ 0.06 而同样本 Spearman ≈ 0.85 → v1 H4 为耦合产物。P2 在 B 的 24 个 run 上复现 G 侧。

## 2.2 功能代价(A5)

fp32 前向、TF32 关闭,η∈[1e-3,1e-2] 四类单位范数方向斜率 2.01 → 局部二次成立;谱/frame KL 代价比逐矩阵 0.26–5.2,中位 0.95;
q_proj 谱步便宜 3 倍,o_proj 谱步贵 2 倍;k_proj 纯缩放方向 KL≈0。

## 2.3 等范数干预(P3 + P4)

| 干预 | 改变什么 | 自由度 | RLVR GSM8K / MMLU | SFT GSM8K / MMLU |
|---|---|---|---|---|
| base | — | — | 0.546 / 0.483 | 0.546 / 0.483 |
| full | 谱 + frame | mn | 0.453 / 0.234 | 0.301 / 0.239 |
| frame_matched(scale 1.00) | frame(去掉谱分量,能量 −0.04%) | mn−r | 0.528 / 0.238 | 0.283 / 0.236 |
| exact_iso(1 seed) | frame,谱精确固定为预训练谱 | mn−r | 0.600 / 0.255 | 0.290 / 0.240 |
| spectrum_matched(scale 50) | **只改奇异值**(当前步精确基,U/V 不变) | r | 0.634 / 0.474 | 0.387 / 0.478 |
| random_ext(scale 52) | r 个固定随机秩一方向 | r | 0.593 / 0.475 | 0.393 / 0.466 |

训练曲线:
- RLVR reward(50 步窗):mn 维三种在 101–150 步达峰 0.59–0.63 后回落到 0.41–0.52;r 维两种单调上升到 0.53–0.58(末窗最高)。
- **SFT 训练 loss:五种模式从第 51 步起逐窗相同**(0.53 → 0.49–0.51;r 维只在前 50 步高 0.1)。
  即在 SFT 上,r 维干预对训练目标的推进与 dense 相同,而遗忘相差 0.24。这是现成的"等推进、不等遗忘"证据,
  也是 C4 的出发点;它同时说明 §3.1 的一阶算术不能预测实际推进。

## 2.4 精度事故(v1)

B 机器 master 权重 bf16,每步更新元素(~1e-5)与量化步长(~2e-5)同量级。v1 的 full 结局与 fp32 的 r 维干预几乎相同。
写进方法论警示(附录),不作证据。

---

# 3. 理论骨架(新增部分;其余继承 v2)

## 3.1 等范数投影的一阶算术

设 P 为到 r 维秩一字典的正交投影,Hp = s·P(H),‖Hp‖_F = ‖H‖_F ⇒ s = ‖H‖/‖PH‖(实测 ≈50,能量比 ≈ r/(mn))。
cos(Hp,H) = 1/s;沿 H 的推进 ⟨Hp,H⟩/‖H‖² = 1/s。更一般地令 s_rel = s/s_match:步长范数 = s_rel‖H‖,沿 H 推进 = s_rel/s_match。
**结论:等 Frobenius 步长不是等一阶推进;任何单点等范数比较都混淆了推进量。** v2 P3 只测了 s_rel=1 与 lr×1 两点。

## 3.2 局部二次模型下累计 KL 与字典无关

在参考分布 Fisher F 的局部二次模型下,T 步独立零均值步 δ_t 的累计 KL 期望为 ½Σ_t E[δ_tᵀFδ_t] = T × 单步 KL(交叉项为零)。
A5 表明单步单位范数 KL 在谱/frame 方向无全局差异 → **若步是随机游走,r 维与 mn 维更新在 300 步后的累计 KL 应相同**。
P3 中二者的 MMLU 相差 0.24,故差异必须来自相干漂移(步之间正相关)或局部模型之外。
**可检验预测**:累计位移 ‖W_300−W_0‖_F 与累计 KL(W_0→W_300)相对 √T·单步值的比,dense 显著大于 r 维(E-v3-0a);
若不成立,则遗忘差异来自非局部效应,C4 的机制段改为纯经验报告。

## 3.3 自由度与相干漂移

对任意固定方向 d,随机 r 维字典的投影只保留 ‖P d‖² ≈ ‖d‖²/s_match² 的能量,放大 s 后沿 d 的推进为 1/s_match。
因此 r 维干预对一切"固定方向的相干漂移"(包括遗忘方向)一律削弱 s_match 倍,而对任务的实际推进(§2.3)并未削弱同样倍数——
这意味着任务学习不是沿某个固定方向的一阶推进,而是在受限族内由训练动力学(on-policy 采样、非线性)找到的路径。
这是 intrinsic-dimension 类现象在"逐步等范数"协议下的表现;v3 只把它写成经验命题并用 E-v3-5(字典大小)检验单调性,不做进一步理论主张。

---

# 4. 假设表与预先锁定的裁决规则

| 假设 | 预测 | 拒绝或降级条件 |
|---|---|---|
| H-1 基不特殊(C3) | 在 s_rel ∈ {1, 3, 10}、≥3 seed(s_rel=1)上,spectral 与 random 字典的 GSM8K、MMLU 差异 < margin(GSM8K 0.03、MMLU 0.02) | 任一档位差异 > margin 且跨 seed 同向 → C3 改写为条件性,并用 A5 逐矩阵代价做关联分析 |
| H-2 保谱不防遗忘(C2 上半) | exact_iso 在 lr×1 与 lr/10 上的 MMLU 与同 lr 的 full 不可分 | exact_iso 在某 lr 上 MMLU 高于 full 超 margin → 改写为"保谱在小步长区间有可测保护",C2 降级 |
| H-3 只改谱不遗忘(C2 下半) | spectrum_matched 在 s_rel=3、10 上 MMLU 仍 ≥ base−0.05,且 GSM8K 不低于 s_rel=1 | s_rel=3 即开始遗忘 → C2 下半改写为"在推进量匹配 dense 之前不遗忘",并与 dense 前沿比较 |
| H-4 dense 的崩溃可由步长解释(C4) | full 在 lr/10 或 lr/30 下,在等训练推进(SFT loss、RLVR reward)点上 MMLU 接近 r 维干预 | full 任一 lr 都无法在等推进点达到 r 维的 MMLU → 自由度有独立贡献,C4 写为正结果 |
| H-5 自由度单调(C4) | random_ext 的 MMLU 随字典大小 r/4 → r → 4r 单调下降,GSM8K 单调上升 | 无单调性 → C4 只保留前沿描述 |
| H-6 MMLU 崩溃含格式漂移 | dense ckpt 在答案位置 {A,B,C,D} 概率质量 < base 一半 | 质量不变 → 崩溃为真实能力损失;否则主文改报 generation-based MMLU |
| H-7 相干漂移(§3.2) | dense 的累计位移/累计 KL 相对 √T·单步值的比 ≥ 3× r 维 | 不成立 → §3.2 的机制段改为阴性报告 |

裁决规则(数据解封前锁定):
- 所有比较用 seed 合并均值,报告二项 SE;"不可分"= 差异 < margin 且 |t| < 2。
- 前沿以 300 步终点为主;每 50 步的中间 ckpt 用于画轨迹,不作独立样本。
- "等推进点"的定义:SFT 用训练 loss 末 50 步均值;RLVR 用训练 reward 末 50 步均值。二者与 GSM8K 矛盾时以 GSM8K 为准并写明。
- MMLU 主指标:letter-logit;若 H-6 成立,主文改报 generation-based(200 题),letter-logit 进附录。
- non-inferiority 主张按 margin 报告,不用"无显著差异"替代。

---

# 5. 实验设计(作者B;0.8B,fp32,300 步,其余参数与 P3 相同,每 50 步存 ckpt 并评测 GSM8K 500 / MMLU 1000)

## E-v3-0 零训练诊断(当天,优先级最高)

- **0a 累计位移与累计 KL。** 对 18 个 e4a run 的 ckpt_000300 与 base:逐矩阵 ‖W_300−W_0‖_F,以及在 A5 的 32 条 GSM8K 参考序列上
  KL(base ‖ ckpt)(fp32 前向,TF32 关闭;A 提供脚本 `analysis_v3/cum_kl.py`)。交付 csv(run, matrix, disp, kl_total)。
- **0b MMLU 格式检查。** base、e4a_rlvr_full_s0、e4a_rlvr_spectrum_matched_s0、e4a_sft_full_s0、e4a_sft_spectrum_matched_s0:
  200 题答案位置 {A,B,C,D} 概率质量之和、argmax 是否在四者内;各存 20 条 greedy 生成(64 token)。
- **0c 单步 KL 对照(A 做)。** 在 A 的 smoke 捕获(同 prompt 流、同 seed 的 full 与四模式,第 10/20 步)上算 KL(W_0+H) 与 KL(W_0+Hp):
  直接测每种模式"每步的功能步长"。若 KL(Hp) ≈ KL(H),则 M1(小功能步)在单步层面被排除。

## E-v3-1 dense 学习率扫描(H-4)

full,lr × {1/3, 1/10, 1/30} × {RLVR, SFT} × 1 seed = 6 run;lr×1/10 加 seed 1 = 2 run。**共 8。**

## E-v3-2 精确保谱加 seed 与降 lr(H-2)

exact_iso:lr×1 seed 1(RLVR、SFT)= 2 run;lr×1/10 seed 0(RLVR、SFT)= 2 run。**共 4。**

## E-v3-3 r 维字典的步长扫描与加 seed(H-1、H-3)

spectrum_matched 与 random_ext,新参数 `--intervention-scale s_rel`:
- s_rel ∈ {3, 10} × 2 字典 × {RLVR, SFT} × 1 seed = 8 run;
- s_rel = 1 seed 2 × 2 字典 × {RLVR, SFT} = 4 run(合计 3 seed);
- s_rel = 0.3,spectrum_matched,RLVR,1 seed = 1 run(低端锚点)。
**共 13。**

## E-v3-4 第三范式(C2/C4 的跨范式复现,可选但便宜)

OPD(teacher 2B):full、spectrum_matched、random_ext,lr×1,seed 0 = **3 run**(OPD 约 20 s/步)。
v1 显示 OPD full 把 MMLU 打到 0.247;若 r 维保住 MMLU,则 C4 在三范式成立。

## E-v3-5 字典大小(H-5;在 E-v3-1/3 出结果后决定)

random_ext,`--dict-mult k`,k ∈ {0.25, 4},s_rel=1,RLVR,seed 0 = **2 run**。

## 预算与优先级

E-v3-0 → E-v3-1 → E-v3-3(s_rel=1 加 seed 优先)→ E-v3-2 → E-v3-3(扫描)→ E-v3-4 → E-v3-5。
不含 E-v3-5 共 28 run:RLVR 14 个(约 2.5–3.5 h/卡)、SFT 11 个(0.5–1.2 h/卡)、OPD 3 个(约 1.7 h/卡);
8 卡并行约 2 天,含每 50 步评测(GSM8K+MMLU 约 10 min/ckpt)。frame_matched 不再新增。
若 09-15 前算力不足,砍序:E-v3-5 → E-v3-4 → E-v3-3 的 s_rel=10 → E-v3-1 的 lr×1/3。

---

# 6. 作者A 的工作

- **代码 `specgeom_v3/ scripts_v3/ analysis_v3/ slurm_v3/`(从 v2 复制后叠加):**
  - `intervene_engine.py`:`--intervention-scale s_rel`(在等范数 s 上再乘);`--dict-mult k`(k<1 取前 k·r 列;k>1 用 ⌈k⌉ 个独立正交字典之和);
    smoke 打印 cos(Hp, H_full) 与 ‖Hp‖/‖H‖。
  - `instrument.py`:捕获中 W 存 fp32(关闭 6.5% 自检缺口)。
  - `train.py`:`--eval-every 50`(存 ckpt 并调用两个评测脚本,评测结果写 `eval_step{n}.json`)。
  - `analysis_v3/cum_kl.py`:base 与 ckpt 之间的 KL 与逐矩阵位移(复用 A5 的参考序列与 fp32 前向)。
  - `analysis_v3/step_kl.py`:E-v3-0c,在 smoke 捕获上算单步 KL(A 侧一次 GPU 前向,约 20 分钟)。
  - `analysis_v3/mmlu_format.py`:E-v3-0b 的字母概率质量与生成 dump。
- **分析与图:**
  - Fig.1 前沿:(训练推进, MMLU)与(GSM8K, MMLU)两幅 × {RLVR, SFT}:dense lr 扫描一条线、spectral 与 random 各一条、exact_iso 两点;等推进配对连线。
  - Fig.2 双重分离:spectrum_matched 与 exact_iso 的 MMLU/GSM8K 随 seed 与步长的分布。
  - Fig.3 账目:累计位移与累计 KL 相对 √T·单步值(E-v3-0a、0c);§3.2 的预测线。
  - Fig.4 KL 代价各向异性(A5,已有)与 Fig.5 信号/噪声分解(v2 Fig.C/D,已有)。
- **写作:** 按 §8 结构;v2 §3–§7 压缩为"测量框架"一节 + 附录。

---

# 7. 数值与工程要求(继承 v2 §12.4)

1. master fp32,加载后断言(已在 680fa78);交付 README 逐 run 核实 ckpt safetensors 为 F32。
2. 捕获 G、H、W 全 fp32(v3 起);H 自检目标 ≥0.99。
3. 所有 KL 测量 fp32 前向、TF32 关闭。
4. 交付 README 必须写:transformers 版本、代码 commit、秒/步、评测口径、同口径 base。
5. 所有 run 共享同一 prompt 流;seed 只改初始化与采样;random_ext 的字典 seed 与训练 seed 分开记录。
6. 评测:GSM8K 500 题 greedy pass@1(含 stop-ids 修复);MMLU 1000 题 letter-logit;E-v3-0b 决定是否加 generation-based MMLU。

---

# 8. 论文结构与主图

题目候选:
1. *Changing the Spectrum Is Neither Necessary Nor Sufficient: A Double Dissociation Between Spectral Motion and Forgetting in LLM Post-Training*
2. *The SVD Basis Is Not Special: Random Rank-One Dictionaries Match Spectral Updates in LLM Post-Training*

正文顺序:
1. 问题与前提:谱继承 / ISO / SVD-PEFT 隐含"基有特权、保谱即保能力";我们直接检验。
2. 测量框架与 C1:可复现梯度在谱方向不富集;观测谱几何由噪声主导。
3. 功能代价:单位步长 KL 无全局谱/frame 符号(A5)。
4. 双重分离(C2)与基不特殊(C3):P3 + E-v3-2/3。
5. 什么决定折中(C4):前沿(E-v3-1/3)、账目(E-v3-0a/0c)、字典大小(E-v3-5)、第三范式(E-v3-4)。
6. 协议与数值(C5):等范数 ≠ 等推进;bf16 master 事故。
7. 讨论与局限。

主图:Fig.1 前沿;Fig.2 双重分离;Fig.3 账目;Fig.4 KL 代价;Fig.5 信号/噪声。

---

# 9. 时间线与停止条件(截止 2026-09-26)

| 日期 | A | B |
|---|---|---|
| 09-12 | v3 代码 + smoke + TASKS_A/B_v3;`cum_kl.py`、`mmlu_format.py` 交 B | E-v3-0a/0b 交付 |
| 09-13 → 09-15 | E-v3-0c;frontier 与账目脚本;§0–§3 初稿 | E-v3-1、E-v3-3(加 seed)、E-v3-2、E-v3-3(扫描)、E-v3-4 开跑并陆续交付 |
| 09-16 → 09-18 | 登记、出图、裁决 H-1 … H-7;决定 E-v3-5 | E-v3-5(若需要);补评测 |
| 09-19 → 09-23 | 正文 | 交付 README、复核 |
| 09-24 → 09-26 | 定稿 | — |

停止条件:
- H-4 成立且 H-5 不成立(dense 降 lr 即可复现 r 维的前沿):C4 收缩为"前沿不可分 + 协议警示",C2/C3 仍为主结果,论文可投。
- H-1 被拒(奇异基在某档位稳定优于随机):题目改用候选 1,C3 改条件性,加 A5 关联分析。
- H-2 被拒(exact_iso 在小 lr 下防遗忘):C2 上半改写,论文主线变为"保谱在小步长下有保护、在大步长下无",仍可投。
- B 算力不足:按 §5 砍序;C2、C3 的加 seed 与 E-v3-1 为最低必需集合(12 run)。

---

# 附录 A. v1 / v2 结果的处置

| 数据 | 处置 |
|---|---|
| v1 Phase 1 六格 G 侧(A,fp32)+ Llama 六格 | C1 主证据(经 A1 分半重算);Llama 只作附录复现 |
| v1 Phase 1 H 侧(A,fp32) | 可用,无新主张,不进主文 |
| v1 B 侧全部训练结局 | 附录"精度事故"一节一张表,标明 bf16 master、不可与 fp32 混排 |
| v1 α 收敛数值 | 对角 SNR 极低的旁证,C1 一节脚注 |
| v2 A1、A5、smoke | 主文 |
| v2 P3、P4 | 主文:C2/C3 的 s_rel=1、lr×1 点;前沿图的锚点 |
| v2 P1、P2 | P1(OPD full MMLU 0.247)进 E-v3-4 的动机;P2 G 侧作为 C1 在 B 侧复现进主文一句 |

# 附录 B. v2 → v3 的实质变化

- 核心问题从"何时开放谱方向"改为"奇异基是否有特权;谱变化与遗忘是否有因果联系;若都否,什么决定折中"。
- C2(条件谱增益)、C3(门控)撤出;新 C2(双重分离)、C3(基不特殊)、C4(自由度与推进量)、C5(协议与数值)。
- 干预协议从单点等 Frobenius 改为 s_rel 与 lr 双扫描的前沿比较,并定义"等推进点"。
- 新增理论预测 §3.2(局部二次下累计 KL 与字典无关)及其检验 E-v3-0a/0c。
- 新增 OPD 第三范式、字典大小轴、MMLU 格式检查。
- 数值要求新增:捕获 W 存 fp32;README 核实 ckpt dtype。
- v1 全部训练结局降为附录案例。
