# Spectral Geometry of Post-Training Gradients

Code, documents and results for "When Should Post-Training Change the Spectrum?".

## 版本、分工与文件对应关系

只有一个分支 `main`。**企划书、任务清单、代码、结果共用同一个版本号**;
每个版本下按人分 A / B。规则:旧版本的文件夹和文档一律不再改动,新版本
用后缀新建。

| | v1(2026-09-06 → 09-09,已完结) | v2(2026-09-09 → 09-11,已冻结) |
|---|---|---|
| **v3(2026-09-11 起,进行中)** | 企划书 `docs/unified_paper_document_v3.md`;任务 `docs/TASKS_A_v3.md` / `docs/TASKS_B_v3.md`;结果 `results/v3/{result_A,result_B}`;代码 `specgeom_v3/ analysis_v3/ scripts_v3/ slurm_v3/`(从 v2 复制后叠加:`--intervention-scale`、每步刷新基、fp32 捕获、`--eval-every`、`--kl-beta`、E-v3-0 诊断脚本)。**v2 自此冻结。** | 主线:奇异基不是特权坐标系;dense 先找健康 lr\*,三个小批次 |
| 企划书(理论 + 实验设计,唯一叙事来源) | `docs/unified_paper_document_v1.md`(credit-assignment 决定谱几何 + SSD 优化器,H1–H7) | `docs/unified_paper_document_v2.md`(谱更新的条件增益;§9.4 登记 v1 结果与判定;§14 优先级) |
| 实验总账 | `docs/EXPERIMENTS_v1.md` | 并入企划书 §11、§14 |
| 作者A 任务清单 | (无单独文档;A 的 v1 工作记录在 `EXPERIMENTS_v1.md`) | `docs/TASKS_A_v2.md`:代码、CPU 重算、GPU smoke、分析与写作 |
| 作者B 任务清单 | `docs/TASKS_B_v1.md`(P0–P6,已全部交付) | `docs/TASKS_B_v2.md`:P0 先决问题、P1 补评测、P2 重算指标、P3 等范数 H5、P4 exact_iso、P5 4B、P6 可选 |
| 作者A 结果 | `results/v1/result_A/`(Phase 1 六格、Llama 六格、h3power、ssd_trial) | `results/v2/result_A/`(v2 指标重算、smoke) |
| 作者B 结果 | `results/v1/result_B/`(Phase 2、E1–E4、4B,36 条) | `results/v2/result_B/`(与 v1 同构,run 目录直接放这里;B 的文字答复也放这里,如 `B_P0_answers_v2.md`) |
| 数值登记处 | `paper/results_draft.md`(v1 登记处) | `paper/results_draft.md`(追加"v2 登记处"小节) |
| 文档索引 | `docs/INDEX.md`(所有文档、交付说明与登记处的入口,含每份 B 答复的一行摘要) | 同左 |
| 代码 | `specgeom/ analysis/ scripts/ slurm/`,与 git tag `v1` 逐字节一致 | `specgeom_v2/ analysis_v2/ scripts_v2/ slurm_v2/`:从 v1 复制后叠加 v2 改动,自包含,不引用 v1 文件夹 |

分工原则(2026-09-09 定):Hopper GPU 紧张,**作者A 只做代码、CPU 计算、极小的
GPU smoke、分析与写作;所有批量 GPU 实验归作者B**。每个 run 的交付物是五个小
文件(`log.jsonl、args.json、eval*.json、metrics.csv、manifest.json`),大文件不进 git。

v2 相对 v1 的代码改动(其余文件为原样复制):
- `specgeom_v2/interventions.py`、`specgeom_v2/intervene_engine.py`:四个等范数干预
  `spectrum_matched / frame_matched / random_ext / exact_iso`(企划书 v2 §11 E4a)
- `scripts_v2/train.py`:新的 `--intervention` 值、`--intervention-seed`、日志 `scale_mean/scale_max`
- `analysis_v2/compute_metrics.py`:微批次分半的交叉样本 SNR、`R_sigma_cross`、
  split-half Spearman、H 自检、`manifest.json`、`--out-root`
- `scripts_v2/summary.py`:交付摘要含新列
- `analysis_v2/unit_test_cpu_v2.py`:v2 的 CPU 测试;`analysis_v2/unit_test_cpu.py` 为 v1 测试在 v2 包上的复跑
- `slurm_v2/`:`recompute.sbatch`(CPU 重算 v1 捕获)、`purge_v1_bulk.sbatch`(重算成功后删 scratch 大文件)、`smoke.sbatch`(四个 v2 模式的 GPU smoke)

## 代码布局(v1 与 v2 同构)
- `specgeom*/` — library: metrics (R_spectrum, SNR, rho), instrumentation,
  Muon, SSD (M1), intervention engine
- `scripts*/train.py` — unified SFT / OPD / RLVR trainer with G/H/W capture
- `scripts*/eval_gsm8k.py`, `eval_mmlu.py` — greedy pass@1 / letter-logit eval
- `analysis*/` — offline metric computation, CPU tests, figures
- `slurm*/` — sbatch scripts (partition gpuq, qos gpu; CPU jobs on normal)
- `runs/` — slurm logs;训练输出在 `/scratch/rhong5/spectral_runs/<run>`

## Environment
- venv: `~/envs_spectral`。主要库的版本固定如下(`requirements.txt` 同步维护;作者B 的机器需对齐):

| 库 | 版本 | 备注 |
|---|---|---|
| python | 3.10.13 | |
| torch | 2.11.0+cu128 | |
| transformers | 5.16.1 | **必须 ≥5.16**。B 实测 5.9.0 忽略 `from_pretrained(dtype=torch.float32)`,模型以 bf16 加载,master 权重变成 bf16,捕获的 H 与谱型干预都被舍入污染(见 `results/v2/result_B/B_P0_answers_v2.md` Q2)。`specgeom_v2/modeling.py` 在加载后断言参数 dtype,不符即报错 |
| tokenizers | 0.23.2 | |
| safetensors | 0.8.0 | |
| huggingface_hub | 1.30.0 | |
| datasets | 5.0.1 | |
| accelerate | 1.14.0 | |
| trl | 1.12.0 | 未在训练循环中使用,仅作依赖 |
| triton | 3.6.0 | |
| numpy | 2.2.6 | |
| pandas | 2.3.3 | |
| matplotlib | 3.10.9 | |
| tqdm | 4.70.0 | |

- 数值约定:fp32 master 权重 + bf16 autocast 前向/反向;KL 类功能代价测量必须 fp32 前向且关闭 TF32(v2 §12.4)。
- HF cache: `/scratch/rhong5/dataset/hf_home` (models pre-downloaded;
  jobs run with `HF_HUB_OFFLINE=1`)
- Models: Qwen/Qwen3.5-0.8B (main), Qwen/Qwen3.5-2B (OPD teacher);
  family axis Llama-3.2-1B/3B-Instruct; scale axis Qwen3.5-4B (作者B)

## Quick start (v3)
```bash
~/envs_spectral/bin/python analysis_v3/unit_test_cpu.py && \
~/envs_spectral/bin/python analysis_v3/unit_test_cpu_v2.py && \
~/envs_spectral/bin/python analysis_v3/unit_test_cpu_v3.py            # all three must PASS
sbatch --partition=contrib-gpuq --gres=gpu:A100.80gb:1 --array=0-3 slurm_v3/smoke.sbatch
python analysis_v3/health.py $RUNS/v3_*_full_lr*_s* --base-gsm8k 0.546 --base-mmlu 0.483   # batch-1 gate
```

## Quick start (v2, frozen)
```bash
~/envs_spectral/bin/python analysis_v2/unit_test_cpu.py && \
~/envs_spectral/bin/python analysis_v2/unit_test_cpu_v2.py      # both must PASS
sbatch --array=0-3 slurm_v2/smoke.sbatch                          # 4 v2 modes, 20 steps SFT
sbatch --array=0-8 slurm_v2/recompute.sbatch                      # v2 metrics from v1 captures
python analysis_v2/compute_metrics.py $RUNS/<run> --out-root results/v2/result_A
```
v1 复现:`git checkout v1` 后按 `docs/TASKS_B_v1.md` 的命令执行。
