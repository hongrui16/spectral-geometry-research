# When Should Post-Training Change the Spectrum? — v3 企划书(草案)

> 状态:**草案,2026-09-11,待用户确认后生效。** 生效后 v2 文档冻结(只改引用),
> 代码进 `*_v3/`,结果进 `results/v3/`,任务清单为 `TASKS_A_v3.md` / `TASKS_B_v3.md`。
> 本文只写 v3 相对 v2 的实质变化、证据总账、新的核心问题与实验设计;
> 符号、恒等式、C1 测量框架(v2 §3–§5、§7 的可证明部分)不重复,按需引用 v2。
> 数字的唯一来源仍是 `paper/results_draft.md`(v2 登记处,2026-09-11 已登记 B 的全部 v2 交付)。

---

# 0. 一句话结论与 v3 的核心问题

**v2 的 P3 主实验(fp32、等 Frobenius 步长、0.8B、RLVR 与 SFT、各 2 seed)给出两条清楚的经验事实:**

1. **权重奇异基不特殊。** 把更新限制在 r 个秩一方向上并匹配步长范数时,用权重的
   奇异基(spectrum_matched)与用同维数的随机正交字典(random_ext)得到的任务收益
   与能力保持几乎相同(RLVR GSM8K 0.634 vs 0.593,MMLU 0.474 vs 0.475;SFT 0.387 vs 0.393,
   0.478 vs 0.466)。
2. **分界线在自由度与有效步长,不在 spectrum 与 frame。** 三种 mn 维更新(full、frame_matched、
   exact_iso)在两个范式上都把 MMLU 打到四选一随机(0.23–0.26),GSM8K 也不高于或低于 base;
   两种 r 维更新都保住 MMLU(≈0.47,base 0.483)且 GSM8K 更高。

但第 2 条有一个 v2 协议自带的混淆:等 Frobenius 步长不等于等推进。
投影后再放大 s≈50 倍的更新与原更新的余弦恰为 1/s≈0.02(§2.2),
即 r 维干预每步沿优化器方向只推进 full 的 2%,其余 98% 的能量在正交方向上。
所以 P3 目前只能支持第 1 条,第 2 条需要在"等推进"或"整条收益—代价前沿"上重做。

**v3 的核心问题**改为:

> **在 LLM post-training 中,权重的奇异基是否是一个有特权的更新坐标系?
> 若不是,决定任务收益与能力保持折中的是什么:更新的自由度、沿梯度的有效步长,还是二者的组合?**

v2 §0.1 的问题("frame 可用时额外开放 spectrum 是否值得")在 P3 之后已经有了答案的方向:
在同维随机字典也能做到的意义上,"开放 spectrum"没有独立价值。v3 把这一阴性结论写成主结果,
并把混淆变量(有效步长、自由度)做成可控实验,这是 v2 §14"阴性结果的论文路径"第 1、3 条的落地。

---

# 1. v2 证据总账(2026-09-11)

| 编号 | v2 的主张或假设 | 证据 | 判定 |
|---|---|---|---|
| C1 / H-A | 观测梯度能量混有主导性采样噪声;v1 的 G 侧 ordering 与 H4 是耦合产物 | A1 分半重算(6 格,信号能量 <1% 噪声);P2 在 B 的 24 个 run 上复现(G 侧) | **成立,进主文** |
| v1 H5(bf16) | frame_only≈full,spectrum_only 失败 | v1 批次 master 为 bf16,谱型干预写回时被舍入,H 侧指标为量化噪声 | **作废**(不能作为证据,只在附录说明) |
| v1 E1–E3(SSD、Muon、消融、分层) | SSD 最差,消融不可分 | 同上,bf16 master;且 SSD 等效步长远小于对照 | **作废为主证据**;附录以"未做步长扫描 + bf16"注记,不再引用 |
| v1 E4(自适应 α → 0.05) | 三范式 α 收敛到接近关谱 | α 由 G 侧对角 SNR 决定(可信);训练结果 bf16 | α 的**数值**可保留为 C1 的一个旁证(对角 SNR 极低),训练结局不引用 |
| v1 4B(SFT、RLVR、Muon) | 规模趋势 | bf16 master | 只在附录报告,不作主张 |
| §9.4 caveat 3(SFT 评测脚本) | SFT 低于 base 是否评测 artifact | P0 Q1:6 个 ckpt 重评差 ≤1.6 点 | **关闭**:真实效应 |
| §9.4 caveat 4(OPD α MMLU 0.262) | 缺 OPD none 对照 | P1:OPD full 本身 MMLU 0.247 | **关闭**:OPD 自身摧毁 MMLU,与 α 无关 |
| §9.4 caveat 5 / §12.4(H 自检失败) | 捕获顺序或基错误 | P0 Q2:根因 bf16 master;fp32 后自检 0.935 | **关闭**;剩余 6.5% 为捕获中 W 以 bf16 存储,v3 改存 fp32 |
| A5(fp32 KL 探针) | 谱/frame 方向单位步长的 KL 代价 | 逐矩阵各向异性 0.26–5.2,中位 0.95,无全局符号 | **成立**;与 P3"基不特殊"一致,进主文 |
| C2(条件谱增益) | 开放谱方向相对 frame 的增益可预测干预收益 | P3:谱字典 ≈ 随机字典 ⇒ 被预测的量在该设置下为零;c 在 B=8 下不可分辨 | **降为附录**(可证明的代数保留,不做实证主张) |
| C3(PGSU 门控) | probe 门控改善折中 | 无实现;被预测的增益为零 | **删除** |
| v2 §14 第 1 项(E4a 等范数) | 把 H5 提升为主结果 | P3 完成,但协议混淆(§2.2) | **部分达成**:字典比较成立;dense vs r 维需 v3 重做 |
| v1 全部"RLVR 提升到 0.636"类数字 | 训练有效 | fp32 下同 lr 的 full 为 0.453,而 fp32 spectrum_matched 为 0.634 ≈ v1 bf16 full | v1 的 full 实际运行在**隐式截断的小有效步长区间**,不能与 fp32 数字混排 |

结论:v2 的测量框架(C1)与 KL 代价测量(A5)成立并被加强;v2 的方法主张(C2、C3)与 v1 的所有训练结局作废或降级;
P3 产生了一个 v2 没有预设的、跨两范式复现的主结果(奇异基 ≈ 随机字典),同时暴露了等范数协议的缺陷。

---

# 2. 对 P3 的解读

## 2.1 结果(均值,2 seed;完整表见登记处)

| | 自由度 | RLVR GSM8K / MMLU | SFT GSM8K / MMLU |
|---|---|---|---|
| base | — | 0.546 / 0.483 | 0.546 / 0.483 |
| full | mn | 0.453 / 0.234 | 0.301 / 0.239 |
| frame_matched(scale 1.00) | mn | 0.528 / 0.238 | 0.283 / 0.236 |
| exact_iso(1 seed) | mn | 0.600 / 0.255 | 0.290 / 0.240 |
| spectrum_matched(scale 50) | r | 0.634 / 0.474 | 0.387 / 0.478 |
| random_ext(scale 52) | r | 0.593 / 0.475 | 0.393 / 0.466 |

RLVR 训练 reward:mn 维三种在 100–150 步达峰(0.59–0.63)后回落到 0.41–0.52;r 维两种单调上升,
末窗 0.53–0.58 为全程最高。SFT 训练 loss 五种无差异(末窗 0.49–0.51),但测试 GSM8K 与 MMLU 差异巨大。

## 2.2 等范数协议的算术

设 P 为到 r 维秩一字典的正交投影,干预为 Hp = s·P(H),s 使 ‖Hp‖_F = ‖H‖_F。则

- s = ‖H‖/‖PH‖,实测 ≈50(能量比 ≈1/2500,与 §4.5 的维数基线 r/(mn) 一致);
- cos(Hp, H) = ‖PH‖/‖H‖ = 1/s ≈ 0.02;
- 沿 H 的推进 ⟨Hp,H⟩/‖H‖² = 1/s ≈ 0.02。

即 r 维干预每步沿优化器方向的推进只有 full 的 2%,其余 98% 的步长能量在与 H 正交、但限于字典内的方向上。
一个更一般的写法:令 s_rel = s/s_match(s_match 为等范数所需的 s),则沿 H 推进 = s_rel/50,
步长范数 = s_rel·‖H‖。v2 的 P3 只测了 s_rel = 1 这一点,而 full 只测了 lr×1 这一点。

## 2.3 三个候选机制(v3 要分开)

- **M1 有效步长。** 在 fp32、当前 lr(RLVR 2e-6,SFT 1e-5)、300 步下,dense 更新推进过快:
  reward 先升后落,GSM8K 与 MMLU 一起下降。r 维干预相当于 lr/50 的 dense 训练加上正交噪声;
  它的"更好"可能只是步长更小。**预测**:full 在 lr/50 下复现 r 维干预的 GSM8K/MMLU。
- **M2 自由度。** 把每步更新限制在 r 个秩一方向(mn 的 1/2000)本身保护通用能力,与步长无关。
  **预测**:在等推进(s_rel/50 = lr_mult)下,r 维前沿仍优于 dense 前沿;随字典大小(r/4、r、4r)单调变化。
- **M3 评测格式漂移。** MMLU 用 letter-logit 读首 token;若 dense 训练改变了输出格式
  (先输出推理文本),字母 logits 失去信息,"随机水平"是格式而非知识丢失。
  **预测**:dense ckpt 在答案位置对 {A,B,C,D} 的概率质量显著下降。这不改变 GSM8K 结论,但改变 MMLU 的解释。

三者不互斥。v3 的实验按能区分它们来设计(§5)。

## 2.4 与 v1 的对照

v1 在 bf16 master 下,每步更新元素(~1e-5)与 bf16 量化步长(~2e-5)同量级,大部分更新被舍入丢弃,
等价于一个未受控的隐式截断/小步长。v1 的 RLVR full(0.636 / 0.477)与 fp32 的 spectrum_matched(0.634 / 0.474)
几乎相同,这进一步支持 M1:v1 的"训练有效且不遗忘"来自小有效步长,而非任何几何性质。
v1 与 v2/v3 的数字不可混排;v1 数据只在附录作为"精度事故"案例。

---

# 3. v3 的贡献目标与不作的推断

| 编号 | 内容 | 性质 | 成功标准 |
|---|---|---|---|
| C1 | 奇异基下的可复现梯度 / 采样噪声分解与维数校正(v2 C1 原样保留) | 测量框架,已成立 | 已达成(A1、P2);v3 只补 fp32 W 捕获后的 H 侧复核 |
| C2′ | **奇异基不特殊**:同维随机秩一字典在任务收益与能力保持上与奇异基等价;A5 的 KL 代价无全局符号 | 主经验结果 | 在 ≥3 seed、两范式、整条 scale 前沿上,spectral 与 random 字典的前沿不可分(non-inferiority margin 预先锁定为 GSM8K 0.03、MMLU 0.02) |
| C3′ | **决定折中的是有效步长与自由度**:等 Frobenius ≠ 等推进;给出 dense 与 r 维更新在(GSM8K,MMLU)平面上的前沿比较 | 主经验结果 + 协议贡献 | 用 lr 扫描与 s_rel 扫描画出的前沿,能判定 M1/M2 各自的贡献;结果无论哪个方向都可报告 |

**不作的推断。**
- 奇异基 ≈ 随机字典,不等于任何低秩方法都等价,也不等于谱在其它规模/任务上不特殊;只在本文设置(0.8B、GSM8K、300 步)内主张。
- r 维更新保住 MMLU,在 M1/M2 未分开前不写成"低自由度防遗忘"。
- MMLU 降到随机在 M3 检查前不写成"知识丢失"。
- v1 的所有训练结局不作为证据。
- 不提出新的优化器或门控方法。

---

# 4. 假设表与预先锁定的裁决规则

| 假设 | 预测 | 拒绝或降级条件 |
|---|---|---|
| H-1 字典等价 | 在 s_rel ∈ {0.14, 1, 7} 与 ≥3 seed 上,spectral 与 random 字典的 GSM8K、MMLU 差异都在 margin 内 | 任一 s_rel 上差异 >margin 且跨 seed 同向 → 报告"奇异基在该步长区间有可测优势",C2′ 改写为条件性 |
| H-2 有效步长解释 dense 的崩溃 | full 在 lr/50、lr/7 下的(GSM8K,MMLU)落在 r 维干预的前沿附近 | full 在任何 lr 下都无法同时达到 r 维干预的 GSM8K 与 MMLU → M2 有独立贡献 |
| H-3 自由度有独立贡献 | 等推进配对(lr_mult = s_rel/50)下,r 维点的 MMLU 高于 dense 点;随字典大小 r/4 → r → 4r 单调 | 等推进下二者不可分 → 自由度无独立作用,C3′ 只剩协议贡献 |
| H-4 MMLU 崩溃含格式漂移 | dense ckpt 在答案位置的 {A,B,C,D} 概率质量 < base 的一半 | 质量不变 → 崩溃为真实能力损失 |

**裁决规则(数据解封前锁定):**
- 所有比较用 2–3 seed 合并的均值,报告二项 SE;"不可分"= 差异 < margin 且 |t| < 2。
- 前沿比较用 300 步终点为主,100/200 步中间 ckpt 为辅(同一 run 内的轨迹不作独立样本)。
- RLVR 同时报告训练 reward 末窗与 GSM8K,二者矛盾时以 GSM8K 为准并写明。
- MMLU 若 H-4 成立,主文改报 generation-based 或 chat-template MMLU,letter-logit 版本进附录。

---

# 5. 实验设计(作者B;0.8B,300 步,fp32,其余参数与 P3 相同)

## E-v3-0 零训练诊断(先做,当天)

- **0a 累计位移。** 对 18 个 e4a run,用 base 与 ckpt_000300 逐矩阵算 ‖W_300 − W_0‖_F,
  并除以 √300 × 该 run 各保存步 ‖H_t‖_F 的中位数(随机游走比)。预期:r 维 run 的比值明显小于 dense
  (正交噪声在固定字典内相消),这是 M2 的机制证据。交付一个 csv(run, matrix, disp, ratio)。
- **0b MMLU 格式检查。** 对 base、e4a_rlvr_full_s0、e4a_rlvr_spectrum_matched_s0、e4a_sft_full_s0,
  在 200 道 MMLU 题的答案位置记录 {A,B,C,D} 四个 token 的概率质量之和与 argmax 是否落在四者之内;
  另各存 20 条 greedy 生成(max_new_tokens 64)。交付 json + 文本。
- **0c 等推进核对。** 任取一个 spectrum_matched run 的一步捕获,算 cos(Hp, H_full)(需 H_full,
  用 v3 代码在 smoke 中打印);预期 = 1/scale。A 侧在 smoke 上做,B 不用做。

## E-v3-1 dense 学习率扫描(检验 H-2)

full,lr × {1/50, 1/7} × {RLVR, SFT} × 2 seed = **8 run**。lr×1 已有(P3 full)。
每 100 步存 ckpt 并评测 GSM8K(500)与 MMLU(1000)。

## E-v3-2 r 维字典的步长扫描(检验 H-1、H-3)

spectrum_matched 与 random_ext,新增 `--intervention-scale s_rel`,s_rel ∈ {0.14, 7} × {RLVR, SFT} × 1 seed = **8 run**;
s_rel = 1 已有(P3)。等推进配对:s_rel 7 ↔ lr×1/7,s_rel 1 ↔ lr×1/50。
s_rel = 50(等推进配 lr×1,步长范数 50 倍)作为探索性单 run 只跑 RLVR spectrum:**1 run**。

## E-v3-3 字典比较加 seed(检验 H-1 的 margin)

spectrum_matched 与 random_ext,s_rel = 1,seed 2 × {RLVR, SFT} = **4 run**(合计 3 seed)。
RLVR 上 P3 的 +0.041 GSM8K 差(≈2.6 SE)由此裁决。

## E-v3-4 字典大小(检验 H-3,若 E-v3-1/2 后仍需要)

random_ext,`--dict-mult k`,k ∈ {0.25, 4}(自由度 r/4 与 4r),s_rel = 1,RLVR,1 seed = **2 run**。
只在 H-3 在 E-v3-2 上出现正信号后再跑。

## 预算

RLVR run 约 2.5–3.5 h/卡(A100),SFT 约 0.5–1.2 h/卡;E-v3-1 至 E-v3-3 共 21 run,
其中 RLVR 11 个;8 卡并行约 1.5 天,含每 100 步评测。捕获保留(fp32 W 版本体积 +9%)。
frame_matched 与 exact_iso 不再新增(两者与 full 同为 mn 维,P3 已给出结果)。

---

# 6. 作者A 的工作

- **代码 `*_v3/`(从 v2 复制后叠加):**
  - `intervene_engine.py`:`--intervention-scale s_rel`(在等范数 s 上再乘 s_rel);`--dict-mult k` 的 random_ext
    (k<1 取前 k·r 列;k>1 用 ⌈k⌉ 个独立正交字典的投影之和,自由度 k·r);smoke 时打印 cos(Hp, H_full)。
  - `instrument.py`:捕获中 W 改存 fp32(关闭 6.5% 自检缺口)。
  - `train.py`:`--eval-every 100`(存 ckpt 并调用两个评测脚本)。
  - `modeling.py` 的 dtype 断言已在 v2(680fa78)。
- **分析 `analysis_v3/`:**
  - `frontier.py`:从 results/v3 读全部 run,画(GSM8K, MMLU)前沿:dense lr 扫描一条线、spectral 与 random 各一条线,
    等推进配对用连线标出;RLVR 与 SFT 两幅。这是主图 Fig.1。
  - 等推进算术与 E-v3-0a 的随机游走比的图(Fig.2)。
  - A5 的 KL 代价表并入 Fig.3(谱/frame 逐矩阵代价比,说明无全局符号)。
  - C1 的信号/噪声图(v2 Fig.C/D)保留为 Fig.4。
- **写作:** 按 §8 结构重写 §0/§1/§9/§10;v2 §3–§7 的理论压缩为一节"测量框架"加附录。

---

# 7. 数值与工程要求(继承 v2 §12.4 并补充)

1. master 权重 fp32,加载后断言(已实现);ckpt safetensors 头必须为 F32,交付 README 逐 run 核实。
2. 捕获中 G、H、W 全部 fp32(v3 起);H 自检目标 ≥0.99。
3. KL 类测量 fp32 前向、TF32 关闭(A5 结论)。
4. 交付 README 必须写:transformers 版本、代码 commit、每 run 秒/步、评测口径、base 同口径数字。
5. 所有 run 共享同一 prompt 流;seed 只改初始化与采样。
6. 评测:GSM8K 500 题 greedy pass@1(含 stop-ids 修复);MMLU 1000 题 letter-logit;若 H-4 成立加 generation-based MMLU(200 题)。

---

# 8. 论文结构与主图(拟)

题目候选:
- *The Spectrum Is Not Special: Random Rank-One Dictionaries Match the Weight SVD Basis in LLM Post-Training*
- *When Should Post-Training Change the Spectrum? Equal-Norm Interventions Say: The Basis Doesn't Matter, the Step Does*

正文顺序:
1. 问题:谱继承文献隐含"奇异基有特权";我们直接检验。
2. 测量:奇异基下的可复现梯度与噪声(C1);结论:可复现梯度在谱方向的能量处于随机字典基线(A1/P2)。
3. 功能代价:单位步长 KL 在谱/frame 方向无全局符号(A5)。
4. 干预:等范数下奇异基 ≈ 随机字典(P3 + E-v3-3);等范数协议的算术缺陷;前沿比较(E-v3-1/2)。
5. 什么决定折中:有效步长与自由度的分解(E-v3-1/2/4、E-v3-0a)。
6. 讨论:对谱继承/ISO 类解释的含义;精度事故(v1 bf16)作为方法论警示;局限。

主图:Fig.1 前沿(RLVR/SFT);Fig.2 等推进算术 + 随机游走比;Fig.3 KL 代价各向异性;Fig.4 信号/噪声分解。

---

# 9. 时间线与停止条件(截止 2026-09-26)

| 日期 | A | B |
|---|---|---|
| 09-12 | v3 代码(scale、dict-mult、fp32 W、eval-every)+ smoke;TASKS_A/B_v3 | E-v3-0a/0b 诊断交付 |
| 09-13 → 09-15 | frontier.py、Fig.2–4 脚本;§0/§1 初稿 | E-v3-1、E-v3-2、E-v3-3 全部开跑并交付 |
| 09-16 → 09-18 | 登记数字、出图、裁决 H-1 至 H-4;决定是否跑 E-v3-4 | E-v3-4(若需要);补评测 |
| 09-19 → 09-23 | 正文 | 交付 README、复核 |
| 09-24 → 09-26 | 定稿 | — |

停止条件:
- 若 E-v3-1 显示 full 在 lr/50 下完全复现 r 维干预(H-2 成立且 H-3 不成立),论文主张收缩为 C2′ + 协议警示,
  自由度一节改为阴性报告;仍可投。
- 若 E-v3-3 显示 spectral 在 RLVR 上稳定优于 random(H-1 被拒),题目改为条件性表述,C2′ 改写,
  并把 A5 的逐矩阵代价与该优势做关联分析。
- 若 B 的算力在 09-15 前无法完成 E-v3-1/2,优先级:E-v3-1(RLVR)> E-v3-3 > E-v3-2 > E-v3-1(SFT)。

---

# 附录 A. v1 / v2 结果的处置

| 数据 | 处置 |
|---|---|
| v1 Phase 1 六格 G 侧(A,fp32) | 保留,C1 主证据(经 A1 分半重算) |
| v1 Phase 1 H 侧 | A 侧 fp32,可用但无新主张;不进主文 |
| v1 B 侧全部训练结局(Phase 2、E1–E4、4B) | bf16 master;只在附录"精度事故"一节以一张表报告,标明不可与 fp32 混排 |
| v1 α 收敛数值 | 作为对角 SNR 极低的旁证进 C1 一节脚注 |
| v2 A1、A5、smoke | 主文 |
| v2 P3、P4 | 主文(字典比较)+ 前沿图的 s_rel = 1 / lr×1 点 |
| v2 P1、P2 | P1 进附录;P2 的 G 侧作为 C1 在 B 侧的复现进主文一句 |

# 附录 B. v2 → v3 的实质变化

- 核心问题从"何时开放谱方向"改为"奇异基是否有特权;若无,什么决定折中"。
- C2、C3 撤出主线;新增 C2′(字典等价)与 C3′(有效步长/自由度前沿)。
- 干预协议从单点等 Frobenius 改为 s_rel 与 lr 双扫描的前沿比较,并显式报告等推进配对。
- 新增 M3(评测格式)检查;MMLU 的解释以此为准。
- 数值要求新增:捕获 W 存 fp32;交付 README 必须核实 ckpt dtype。
- v1 全部训练结局降为附录案例。
