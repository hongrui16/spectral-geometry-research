# 作者A 任务清单 v3

对应企划书 `docs/unified_paper_document_v3.md`;B 的清单 `docs/TASKS_B_v3.md`。A 只做代码、CPU 计算、极小 GPU smoke、分析与写作;批量 GPU 归 B。

## 版本管理
v3 = `docs/unified_paper_document_v3.md` + `TASKS_A_v3 / TASKS_B_v3` + `results/v3/{result_A,result_B}` + 代码 `specgeom_v3/ analysis_v3/ scripts_v3/ slurm_v3/`(从 v2 复制后叠加改动,自包含)。v2 全部冻结,只改引用。

## A0:代码(2026-09-11)

| 项 | 文件 | 状态 |
|---|---|---|
| `--intervention-scale s_rel`;`cos_log`;SVD 基每步刷新;`observe_grad` 首步基初始化;random_ext 短路 | `specgeom_v3/intervene_engine.py` | ✅ CPU 测试通过 |
| 捕获 G/H/W 存 fp32 | `specgeom_v3/instrument.py` | ✅ |
| `--eval-every`(存 ckpt + 同步评测 + 末步 eval_*.json);`--kl-beta`(k3,冻结参考);`resp_len_mean/trunc_frac/cos_mean` 日志 | `scripts_v3/train.py` | ✅ 语法与 CPU 路径;GPU smoke 见 A1 |
| E-v3-0a `cum_kl.py`、E-v3-0b `mmlu_format.py`、E-v3-0c `step_kl.py`、闸门 `health.py` | `analysis_v3/` | ✅ 写好;0a/0b 由 B 跑,0c 由 A 在 smoke 捕获上跑 |
| v3 CPU 测试(scale/cos、每步刷新、exact_iso、fp32 捕获) | `analysis_v3/unit_test_cpu_v3.py` | ✅ 三套测试(v1、v2、v3)在 v3 包上全部 PASS |
| smoke sbatch(4 路:sft spectrum s_rel=3、sft random、rlvr none kl-beta、rlvr spectrum;各 8 步,含 eval-every) | `slurm_v3/smoke.sbatch` | 已提交 contrib-gpuq |

## A1:GPU smoke —— ✅ 通过 2026-09-12(jobs 9913754 / 9914443,contrib-gpuq;五路:sft spectrum s_rel=3、sft random、sft none、rlvr none+kl-beta、rlvr spectrum)
通过标准:四路无 NaN;`scale_mean` 谱/随机约 50、`cos_mean` ≈ 0.02;s_rel=3 的 run 里 `‖Hp‖/‖H‖` 应为 3(compute_metrics 的 H 自检仍 OK);`eval_step000008_*.json` 与 `eval_gsm8k.json` 生成;kl-beta 路径不报错;捕获 W/H 为 fp32(sbatch 末尾断言);`resp_len_mean/trunc_frac` 出现在 rlvr 日志。第一次 RLVR 两路因 128 token 全截断、末步无有效 group 而跳过评测 → 已修 train.py(跳过步也执行 ckpt/eval)。五路复跑全过:cos_mean 0.021–0.024、scale≈50、s_rel=3 记录正确、H 自检 1.0、捕获 fp32、eval_step/eval_* 齐全、kl-beta 路径正常。已 push(da2d160 起),B 可开批次一。

## A1b:批次一 SFT 六个 run 由 A 跑(2026-09-12 提交 gpuq A100.40gb,`slurm_v3/batch1_sft.sbatch`;交付到 `results/v3/result_A/v3_sft_full_lr*_s*`)

## A2:批次一裁决 —— SFT 侧 ✅ 2026-09-12:lr\* = 1e-6(×1/10 HEALTHY,×1/3 MMLU 衰减,×1/30 健康但学得少);dense@lr/10 与 r 维干预同前沿 → SFT 上 H-4 成立、C5 初判阴性。SFT 批次二 8 个 run 已由 A 提交(`slurm_v3/batch2_sft.sbatch`)。RLVR 侧等 B。

## A2(原文):批次一裁决(B 交付当天)
1. `analysis_v3/health.py` 跑 12 个 dense run → 定 lr\*(最大的 HEALTHY 档);写进 `TASKS_B_v3.md` 批次二并 push。
2. E-v3-0a:累计位移/KL 相对 √T·单步值的比(单步值用 A5 或 E-v3-0c),判 H-5 初值;E-v3-0b:判 H-4(字母概率质量)。
3. 登记 `paper/results_draft.md` v3 登记处;若三档都不健康,给 B 两条备选命令(kl-beta / 大 batch)。

## A3:E-v3-0c —— ✅ 第 8 步已测(job 9914606):full 6.5e-3 vs spectrum(s_rel=3)1.3e-4 vs random 5.1e-5 nats/token;登记处与 v3 §3.2 已改写。第 4 步复核(job 9914627)一致:full 3.1e-2 vs 4.5e-4 / 7.1e-5。RLVR 捕获复测可选。

## A4:分析与出图(批次二、三交付后)
- `analysis_v3/frontier.py`:Fig.1 前沿(GSM8K, MMLU)× {RLVR, SFT},dense lr 线、两字典 s_rel 线、frame_matched/exact_iso 点;Fig.2 六种更新在 lr\* 与 lr×1 的带 SE 对比;Fig.3 崩溃诊断 + 累计 KL 比。
- 裁决 H-1 … H-5,按 v3 §4 规则写结论;决定批次三范围。

## A5:写作
按 v3 §8 结构;v2 §3–§7 压缩为"测量框架"一节 + 附录;附录"精度事故"表。

## 时间线
- D0(09-12):A1 通过、push、B 开批次一。
- D1:A2 裁决、批次二清单。
- D2–D3:批次二裁决、批次三清单;A4 脚本。
- D4–D5:批次三裁决、出图、登记。
- D6 起:正文。
