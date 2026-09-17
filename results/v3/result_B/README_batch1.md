# v3 batch 1 — 作者 B 交付说明

代码 commit `eb3f692`(与 worker.log 中记录的一致)。本文件描述 B 侧跑出的所有 v3 batch-1
结果的产生条件、实测成本与偏差。**结果表见文末,标注"待补"的部分随第二波 run 完成后补齐。**

## 1. 环境

| 项 | 值 |
|---|---|
| python | `$NAS/venv_v2/bin/python` |
| torch | 2.11.0+cu128 |
| transformers | **5.16.1** |
| GPU | A100-SXM-80GB(第二波部分 run 可能落在 H100-SXM-80GB,见 §4) |
| 模型 | Qwen3.5-0.8B |
| 环境变量 | `HAS_TT_DATA=False`(所有训练任务) |

**fp32 master 权重**:transformers 5.9.0 会静默忽略 `from_pretrained(dtype=torch.float32)`,
导致 master 权重是 bf16、`H = W_after − W_before` 退化成量化噪声。`specgeom_v3/modeling.py`
在 `from_pretrained` 之后显式 `model.to(dtype=dtype)` 并断言参数 dtype。5.16.1 已无此问题
(320 个参数全部 fp32),但断言保留。每个 run 结束后另有一道 ckpt safetensors dtype 自查,
结果写在各 run 目录的 `dtype_check.txt`,期望值 `F32`。

## 2. 训练协议

7 个 dense RLVR run,与 A 的 `slurm_v3/batch1_sft.sbatch` 对称(A 跑 SFT 六个,B 跑 RLVR):

```
scripts_v3/train.py --objective rlvr --steps 300 --save-every 20 \
  --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --eval-every 50 \
  --lr <LR> --seed <S>
```

| tag | lr | 相对基准 | seeds |
|---|---|---|---|
| lr0.33  | 6.67e-7 | ×1/3  | 0, 1 |
| lr0.1   | 2e-7    | ×1/10 | 0, 1, 2 |
| lr0.033 | 6.67e-8 | ×1/30 | 0, 1 |

基线(base 模型,同一套评测脚本):GSM8K pass@1 = **0.546**,MMLU = **0.483**。
评测:GSM8K test[:500] greedy pass@1;MMLU 1000 题 zero-shot letter-logit。

**ckpt 说明**:`--save-ckpt-every` 默认 250,`--eval-every 50` 产生的中间 ckpt 评完即删。
因此训练中途看不到 `ckpt_*` 目录是正常的,最终只保留 `ckpt_000250` 与 `ckpt_000300`。

## 3. 实测成本

| 项 | 实测值 |
|---|---|
| 训练 | 30.6 s/step → 300 步约 153 min |
| GSM8K 评测 | 336–362 s |
| MMLU 评测 | 77–83 s |
| 单次评测块 | 约 8 min(存 ckpt + 两项评测) |
| 后处理(compute_metrics + summary + 交付拷贝) | 11–16 min(GPU) |
| **单 run 端到端(领取→DONE)** | **实测 3 h 16 m / 3 h 22 m** |
| capture 体积 | `capture/step_*.pt` 每 20 步一个,单个 1.835 GB → 单 run 约 26 GB |
| 保留的 ckpt | `ckpt_000250` + `ckpt_000300`,各 2.9 GB(中间评测 ckpt 评完即删) |

**fp32 master 已在真实 run 上验证**:`v3_rlvr_full_lr0.33_s0` 的
`ckpt_000300` safetensors dtype 自查输出 `['F32'] OK`,每个 run 的
`dtype_check.txt` 都会记录该结果。`compute_metrics.py` 在真实 v3 RLVR capture 上
`METRICS_RC=0`,产出 `metrics.csv` 406 行,与 A 的 SFT run 行数一致。

## 4. 偏差与事故(必须随结果一起读)

1. **两波串行,墙钟约 7 h 而非 3.4 h**。A100-80G 配额 4/4 用满,补拉的 worker 排队 20 分钟
   未落地。7 个 run 分两波:第一波 4 个(lr0.33_s0/s1、lr0.1_s0/s1),第二波 3 个
   (lr0.033_s0/s1、lr0.1_s2)。这是**配额约束,不是计算量**。第二波可能落在 ad_llm 的
   H100 上;A 的 batch 2 同样混用了 3g.40gb / A100.40gb / A100.80gb,故先例已在。
2. **两个 run 的首轮尝试失败,原因未完全确证**。`v3_rlvr_full_lr0.33_s0`(05:41 领取、
   06:12 rc=1 失败,主机 `trial-303095066`)与 `v3_rlvr_full_lr0.33_s1`(06:17 失败),
   两者都带 Traceback,失败窗口集中在 06:12–06:17,约落在 step-50 评测附近。
   **两个目录都已被第二次尝试覆盖,traceback 均未留存**,因此无法直接判定失败原因。
   s0 换机重跑后 step-50 评测干净通过,这是"主机故障而非评测代码"的**间接证据**,
   但 s1 的失败原因没有直接证据 —— 不应写成已确认的单一主机故障。
   缓解措施:`b_cluster/worker_queue.sh` 现在扫描整个 worker.log 找
   `Found no NVIDIA driver` 签名,命中即释放锁并隔离该 slot,避免中毒主机把队列里
   剩余 run 逐个杀掉。重跑后的 7 个 run 全程未再出现该签名。
3. **主机驱动失效共 4 次(2 次确证)**,均发生在**评测子进程初始化 CUDA 的时刻**:
   训练进程持有 CUDA 上下文照常运行,新起的评测子进程却拿不到驱动
   (`torch._C._cuda_init()` 抛 `Found no NVIDIA driver`),而
   `train.py` 的 `subprocess.run(..., check=True)` 把它向上抛,整个 run 崩掉。
   估算每次评测约 5% 故障率 ⇒ 单 run(6 次评测)约 26% 被打死,实际 7 个中了 3 个。
   **B 因此修改了 `scripts_v3/train.py` 的 `run_evals`,捕获 `CalledProcessError`
   后记录一条 `eval_skipped` 并继续训练**。只改错误处理、不动训练数学;
   补丁写入于 11:43:17 UTC;本批 `lr0.033_s0`(11:45 领取)与 `lr0.033_s1`
   (11:50 领取)两个重跑运行在补丁后的代码上,其余 5 个均为补丁前,
   详见 `NOTES_batch1.md`。

4. 集群曾出现三台 i2v H100 pod **出生即坏**(`nvidia-smi` 找不到 libnvidia-ml.so),
   已排除该 quota group;A100 组正常。快检脚本 `b_cluster/cuda_probe.sh`。

## 5. 本批交付物

| 路径 | 内容 |
|---|---|
| `cum_kl/` | E-v3-0a:18 个 e4a ckpt 相对 base 的累计位移 ‖ΔW‖_F 与累计 KL |
| `mmlu_format/` | E-v3-0b:MMLU 作答格式检查(base + 4 个 e4a ckpt) |
| `health_v3_rlvr_gate.csv` | **本批主结果**:7 个 RLVR run 的 health gate,lr* = 2e-7 |
| `mmlu_stability_rlvr.txt` | 单点判定 vs 最后 3 个 ckpt 均值判定的对照(RLVR) |
| `health_v3_sft_gate.{csv,txt}` | B 独立复核 A 的 SFT batch-1 gate(结论:lr* = 1e-6,与 A 一致) |
| `mmlu_stability_sft.txt` | 同上对照(A 的 SFT);该表显示 A 的 lr* 在两种读法下判定相反 |
| `b_cluster/mmlu_stability.py` | 上述对照脚本(B 侧新增) |
| `NOTES_batch1.md` | 全部过程记录与逐条读法 |
| `v3_rlvr_full_lr*_s*/` | 7 个 RLVR dense run:`log.jsonl` / `args.json` / `summary.txt` / `metrics.csv` / `manifest.json` / `eval_*.json` / `dtype_check.txt` |

**未随仓库交付的两样东西**(体积原因,需要时找 B 取):
- `rho_hist.pt`:每个 run 约 **98 MB**(7 个共约 686 MB),`compute_metrics` 的
  rho 直方图。A 的 `result_A` 同样没有收录它。留在
  `/mnt/bn/genai-nebula/kaifan.fk/spectral/runs/results_B_v3/<run>/rho_hist.pt`。
- `capture/step_*.pt` 与 `ckpt_000250` / `ckpt_000300`:每个 run 约 32 GB,
  留在 `/mnt/bn/genai-nebula/kaifan.fk/spectral/runs/<run>/`。

**`lr0.033_s0` 的目录只有 8 个文件而不是 20**,因为它所在主机驱动失效、6 次在线评测
全部被跳过,没有 `eval_step*_*.json` 中间点;它的 `eval_gsm8k.json` / `eval_mmlu.json`
是离线从 `ckpt_000300` 补算的,`metrics.csv`(406 行)与 `dtype_check.txt` 正常。详见 §6.6。

## 6. 结果

### 6.1 E-v3-0a 累计位移与累计 KL — 已完成
见 `NOTES_batch1.md` 对应小节。要点:RLVR 上秩受限的更新在权重空间走得更远
(‖ΔW‖ 0.89–0.94 对 full 的 0.755),函数空间却动得更少(KL 0.08–0.21 对 2.0–3.7,
差 10–30 倍)。"等 Frobenius 范数"不等于"等推进"。

### 6.2 E-v3-0b MMLU 格式检查 — 已完成
见 `NOTES_batch1.md`。要点:全秩更新把 MMLU 打到 ~0.24 是**作答格式崩塌**,不是知识丢失。

### 6.3 A 的 SFT gate 复核 — 已完成
HEALTHY = {lr0.033, lr0.1},lr* = 最大 healthy = **1e-6**,与 A 写死的值一致。
唯一 FAIL 是 lr×1/3 的 MMLU(0.302)—— 结合 6.2,该判据实际筛的是格式崩塌。

### 6.4 RLVR dense lr 扫描与 health gate — 已完成

```
                 cfg  n_seeds  gsm8k  mmlu  gsm8k_spread  decline  trunc_frac_last  verdict
v3_rlvr_full_lr0.033        2  0.562 0.479         0.000    0.030            0.326  HEALTHY
  v3_rlvr_full_lr0.1        3  0.640 0.470         0.028    0.026            0.139  HEALTHY
 v3_rlvr_full_lr0.33        2  0.696 0.452         0.072    0.016            0.575  FAIL
```

**lr\* = 最大的 healthy 档 = lr×1/10 = 2e-7。** base:GSM8K 0.546 / MMLU 0.483。

三档的图景:lr×1/30 近乎空操作;lr×1/10 明显提升 GSM8K(+0.094)而 MMLU 基本不掉
(−0.013);lr×1/3 提升最大(+0.150)但 MMLU 真实下降、两个 seed 不可复现
(离散 0.072)、57.5% 回答被截断。**健康区上界落在 lr×1/10。**

**引用本表时请连余量一起引用**(理由见 §6.5):

| 判据 | lr×1/30 | lr×1/10 | lr×1/3 |
|---|---|---|---|
| MMLU ≥ 0.453 | 0.479 (+0.026) | 0.470 (+0.017) | 0.452 (**−0.001**) |
| GSM8K 离散 ≤ 0.04 | 0.000 | 0.028 | 0.072 (−0.032) |
| decline ≤ 0.030 | 0.0295 (**+0.0005**) | 0.026 (+0.004) | 0.016 |

### 6.5 方法论:这套 gate 的阈值精度超出了实验规模的分辨率

本项目中**四次**判定落在阈值 0.001 附近:A 的 SFT lr×1/10(MMLU,+0.003)、
B 的 RLVR lr×1/3(MMLU,−0.001)、B 的 RLVR lr×1/10 在只有 2 个 seed 时
(decline,+0.001)、B 的 RLVR lr×1/30(decline,+0.0005)。四次独立的贴线不是巧合。

三条具体证据:
1. **同一个 run 内**相邻 ckpt 的 MMLU 就能摆动 0.057(3.6 个抽样 SE),且这种不稳定
   **随 lr 单调增大**(A 的完整 SFT 数据:×1/30 最大 0.6 SE、×1/10 0.8 SE、×1/3 5.8 SE)。
2. **单点读法与"最后 3 个 ckpt 均值"读法会给出相反判定**:A 的 SFT lr\*
   单点 0.456 PASS / 均值 0.452 FAIL。B 的 RLVR lr\* 两种读法都 PASS(0.470 / 0.472),
   **即 RLVR 侧的 lr\* 稳健,SFT 侧的不稳健**,并列引用时应说明这一差别。
3. **跨 seed 平均会掩盖单个 seed 的失败**:A 的 `v3_sft_full_lr0.1_s1` 终值 0.450
   已低于判据,被 s0 的 0.462 拉过线。

建议:gate 的 MMLU 改用最后若干 ckpt 的均值并报告其波动;任何 HEALTHY/FAIL 标签
都附带余量。脚本见 `b_cluster/mmlu_stability.py`。

### 6.6 本批的已知局限

1. **`lr0.033_s0` 没有中间评测轨迹**。它所在主机驱动失效,6 次在线评测全部被跳过
   (训练本身完好,补丁使其跑满 300 步)。终值 0.562 / 0.477 是离线从 `ckpt_000300`
   补算的,**已进入 gate 表**;但 `mmlu_stability` 那一行因此只有 1 个 seed 的轨迹。
2. **lr×1/3 只有 2 个 seed**,而它恰恰是 seed 离散最大的一档(0.072)。
3. lr×1/30 的 `gsm8k_spread = 0.000` 是巧合(两 seed 都是 281/500),其 MMLU 并不相同
   (0.477 / 0.481);应表述为"低于分辨率"而非"为零"。
