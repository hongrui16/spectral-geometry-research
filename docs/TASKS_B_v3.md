# 作者B 任务清单 v3(唯一入口,逐条命令版)

对应企划书 `docs/unified_paper_document_v3.md`(§4 假设与裁决、§5 实验设计),结果交付到 `results/v3/result_B/`(与 v1/v2 同构:每个 run 一个目录直接放在这里,不再套一层)。
v2 清单 `docs/TASKS_B_v2.md` 已全部交付,不再执行。**代码用 main 最新提交的 `*_v3/` 文件夹**(`specgeom_v3/ analysis_v3/ scripts_v3/ slurm_v3/`);v2 文件夹冻结,不再使用。

```bash
cd spectral-geometry-research && git checkout main && git pull && git rev-parse --short HEAD   # 把这个 hash 写进交付 README
```
环境变量同 v2(`HF_HOME`、`HF_HUB_OFFLINE=1`、`RUNS`、`PY`)。**transformers ≥ 5.16**(README 版本表);`load_model` 会断言参数 dtype,若报错先对齐环境。
铁律不变:除 `--out`/`CUDA_VISIBLE_DEVICES`/`--seqs-per-microbatch 1` 以及本清单明确写出的参数外不改任何参数。

**组织方式:三个批次,每批一天;每批交付后 A 当天出裁决,再开下一批。批次一是闸门,没有 A 的裁决不要开批次二。**

v3 代码相对 v2 的变化(与 B 有关的):
- `--intervention-scale s_rel`:等范数匹配后再乘 s_rel(步长范数 = s_rel·‖H‖_F);默认 1.0 = v2 行为。
- `--eval-every 50`:每 50 步存 ckpt 并同步跑 GSM8K(500)+ MMLU(1000),写 `eval_step000050_gsm8k.json` 等;末步另存为 `eval_gsm8k.json / eval_mmlu.json`(summary.py 用)。中间 ckpt 评测后自动删除(`--keep-eval-ckpts` 可保留),末步与 `--save-ckpt-every` 倍数的 ckpt 保留。**每 run 因此多约 1 小时评测时间。**
- `--kl-beta`:RLVR 的 k3 KL 惩罚(冻结 bf16 参考模型),**只在批次一闸门失败、A 指定时用**。
- 日志新增 `resp_len_mean`、`trunc_frac`(RLVR/OPD 每步)、`cos_mean`(干预 run 的保存步)。
- SVD 基每步刷新(B 的 v2 修复已合并);捕获中 G/H/W 存 fp32(capture 体积约 +9%)。

---

## 批次一(闸门):健康 dense 基线 + 零训练诊断

### 1.0 零训练诊断(先做,当天交付;用 v2 的 18 个 e4a ckpt,不训练)

**E-v3-0a 累计位移与累计 KL**(1 张卡,约 20 分钟):
```bash
$PY analysis_v3/cum_kl.py --runs-root $RUNS --glob 'e4a_*' --step 300 --out $RUNS/results_B_v3/cum_kl
```
交付 `results/v3/result_B/cum_kl/{cum_kl_runs.csv,cum_kl_matrices.csv,manifest.json}`。

**E-v3-0b MMLU 格式检查**(每个 ckpt 约 3 分钟):
```bash
for r in e4a_rlvr_full_s0 e4a_rlvr_spectrum_matched_s0 e4a_sft_full_s0 e4a_sft_spectrum_matched_s0; do
  $PY analysis_v3/mmlu_format.py --model $RUNS/$r/ckpt_000300 --out $RUNS/results_B_v3/mmlu_format/$r.json
done
$PY analysis_v3/mmlu_format.py --model Qwen/Qwen3.5-0.8B --out $RUNS/results_B_v3/mmlu_format/base.json
```
交付 `results/v3/result_B/mmlu_format/*.json`。

### 1.1 E-v3-1 dense 学习率扫描(12 run,全可并行)

> **分工更新(2026-09-12):SFT 的 6 个 run 由 A 在本地集群空闲的 A100.40gb 上跑(`slurm_v3/batch1_sft.sbatch`,已提交),B 只跑下面 RLVR 的 6 个。** 若 B 的卡有富余,可加跑 RLVR 的第 3 个 seed(`--seed 2`)而不是重复 SFT。

lr×1 = RLVR 2e-6 / SFT 1e-5(P3 用的值)。三档 × 两范式 × 2 seed。**每 run 都加 `--eval-every 50`。**

RLVR 6 个(模板;按下表替换 `--lr`、`--seed`、`--out`):
```bash
$PY scripts_v3/train.py --objective rlvr --steps 300 --save-every 20 --lr 6.67e-7 \
  --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 \
  --eval-every 50 --out $RUNS/v3_rlvr_full_lr0.33_s0
```
| --lr | 档 | --seed | --out |
|---|---|---|---|
| 6.67e-7 | 1/3 | 0 / 1 | v3_rlvr_full_lr0.33_s0 / _s1 |
| 2e-7 | 1/10 | 0 / 1 | v3_rlvr_full_lr0.1_s0 / _s1 |
| 6.67e-8 | 1/30 | 0 / 1 | v3_rlvr_full_lr0.033_s0 / _s1 |

SFT 6 个(模板去掉 `--rollouts`;目录 `v3_sft_full_lr*`):
```bash
$PY scripts_v3/train.py --objective sft --steps 300 --save-every 20 --lr 3.33e-6 \
  --prompts-per-step 8 --max-new-tokens 384 --seed 0 \
  --eval-every 50 --out $RUNS/v3_sft_full_lr0.33_s0
```
| --lr | 档 | --seed | --out |
|---|---|---|---|
| 3.33e-6 | 1/3 | 0 / 1 | v3_sft_full_lr0.33_s0 / _s1 |
| 1e-6 | 1/10 | 0 / 1 | v3_sft_full_lr0.1_s0 / _s1 |
| 3.33e-7 | 1/30 | 0 / 1 | v3_sft_full_lr0.033_s0 / _s1 |

每 run 跑完:
```bash
$PY analysis_v3/compute_metrics.py $RUNS/<run> --out-root $RUNS/results_B_v3
$PY scripts_v3/summary.py $RUNS/<run>
```
交付每 run 目录:`log.jsonl、args.json、eval_gsm8k.json、eval_mmlu.json、eval_step*_{gsm8k,mmlu}.json、metrics.csv、manifest.json、summary.txt`。
自查(A 也会跑):
```bash
$PY analysis_v3/health.py $RUNS/v3_rlvr_full_lr*_s* $RUNS/v3_sft_full_lr*_s* --base-gsm8k 0.546 --base-mmlu 0.483
```
健康判据(H-0):reward 末 100 步均值 ≥ 峰值 −0.03;MMLU ≥ 0.453;两 seed GSM8K 差 ≤ 0.04。**lr\* 由 A 定,写在批次二的清单里。**

### 1.2 闸门失败时的备选(只在 A 指定后跑)
若三档都不健康:RLVR 加 `--kl-beta 0.01`(lr×1/10),或 `--prompts-per-step 32`;A 会给出具体两条命令。

---

## 批次二 + 批次三(RLVR;2026-09-13 A 裁决后一次性放开,14 run 全可并行)

> **RLVR lr\* = 2e-7(×1/10)**,采用 B 批次一报告(三判据 HEALTHY;lr×1/3 在 MMLU 与 GSM8K 离散上双失败)。A 收到 run 目录后用 `health.py` 复核,若翻案会另发通知,但不阻塞开跑。
> SFT 侧三批已全部由 A 跑完并裁决(H-0 ✅ lr\*=1e-6,H-1 ✅ 3 seed,H-2 ✅,H-3 ✗ → C5 SFT 阴性);B **不跑任何 SFT**。批次二裁决没有改变批次三的设计,所以 RLVR 的批次二和批次三合并一次放开。

模板(与批次一 RLVR 完全相同,只改 `--lr / --seed / --intervention / --intervention-scale / --out`;**每 run 都加 `--eval-every 50`**):
```bash
$PY scripts_v3/train.py --objective rlvr --steps 300 --save-every 20 --lr 2e-7 \
  --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 \
  --intervention frame_matched --intervention-scale 1 \
  --eval-every 50 --out $RUNS/v3_rlvr_frame_matched_lr0.1_s0
```

| # | 实验 | --intervention | --intervention-scale | --lr | --seed | --out |
|---|---|---|---|---|---|---|
| 1–2 | E-v3-2 | frame_matched | 1 | 2e-7 | 0 / 1 | v3_rlvr_frame_matched_lr0.1_s0 / _s1 |
| 3–4 | E-v3-2 | exact_iso | 1 | 2e-7 | 0 / 1 | v3_rlvr_exact_iso_lr0.1_s0 / _s1 |
| 5–6 | E-v3-3 | spectrum_matched | 1 | 2e-7 | 0 / 1 | v3_rlvr_spectrum_matched_lr0.1_s0 / _s1 |
| 7–8 | E-v3-3 | random_ext | 1 | 2e-7 | 0 / 1 | v3_rlvr_random_ext_lr0.1_s0 / _s1 |
| 9 | E-v3-4 | spectrum_matched | 1 | 2e-6(lr×1) | 2 | v3_rlvr_spectrum_matched_lr1_s2 |
| 10 | E-v3-4 | random_ext | 1 | 2e-6(lr×1) | 2 | v3_rlvr_random_ext_lr1_s2 |
| 11 | E-v3-5 | spectrum_matched | 3 | 2e-7 | 0 | v3_rlvr_spectrum_matched_lr0.1_srel3_s0 |
| 12 | E-v3-5 | random_ext | 3 | 2e-7 | 0 | v3_rlvr_random_ext_lr0.1_srel3_s0 |
| 13 | E-v3-5 | spectrum_matched | 10 | 2e-7 | 0 | v3_rlvr_spectrum_matched_lr0.1_srel10_s0 |
| 14 | E-v3-5 | random_ext | 10 | 2e-7 | 0 | v3_rlvr_random_ext_lr0.1_srel10_s0 |

E-v3-6(dense lr\*×3)不用跑:批次一的 lr×1/3 = 6.67e-7 已是该点。E-v3-7/8 暂不开。
优先级:1–8 先起(裁决 H-2 / H-1);9–14 随后。SFT 上 5–8 与 11–14 的结果全部落在同一点,RLVR 预期相同,但这是 C5 的最终判据,必须跑。

每 run 跑完同批次一:`compute_metrics.py` + `summary.py`;交付同批次一目录清单。
**scratch 清理规则(A 已执行)**:每 run `compute_metrics` 跑完后删 `capture/` 与中间 ckpt,只留 `ckpt_000300` + 小文件。

### 对 B 批次一报告的两点回复(A,2026-09-13)
1. **A 的 SFT lr\* 余量确实只有 0.003**(full lr\* MMLU 均值 0.456 vs 阈值 0.453;s1 终值 0.450;"末三点均值"读法 s0 0.458 / s1 0.447,合并 0.4525 → FAIL)。已登记。这不改变 SFT 三批的裁决:H-1/H-2/H-3 都是**同一 lr 下**各族之间的比较,而且 lr×1/30(MMLU 0.461,余量 0.008)同样健康并落在与 r 维干预相同的前沿点上;正文引用 lr\* 时连余量一起给,SFT dense 参照点同时报 lr×1/10 与 lr×1/30。
2. **step_kl.py 的 seed 没有不一致**:smoke 目录名 `smoke_v3_<obj>_<int>_s<N>` 里的 `_s<N>` 是 `--intervention-scale`(s_rel),不是 seed;`slurm_v3/smoke.sbatch` 五个 smoke 全部 `--seed 0`,同一 prompt 流。`spectrum_matched_s3` = s_rel 3、seed 0。E-v3-0c 前提成立。命名歧义已在登记处注明。
3. B 对 `scripts_v3/train.py` 的 run_evals 容错补丁(异常→记 `eval_skipped` 继续训练)A 接受;请随批次一交付把 diff 一并给出,A 合入 main。

---

## 交付 README 必须包含
transformers 版本、代码 commit、每类 run 的秒/步、评测口径与同口径 base、每 run ckpt 的 safetensors dtype 核实(F32)、与清单的任何偏离。
