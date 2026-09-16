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

## A2:批次一裁决 —— SFT 侧 ✅ 2026-09-12:lr\* = 1e-6(×1/10 HEALTHY,×1/3 MMLU 衰减,×1/30 健康但学得少);dense@lr/10 与 r 维干预同前沿 → SFT 上 H-4 成立、C5 初判阴性。SFT 批次二 8 个 run 已由 A 提交(`slurm_v3/batch2_sft.sbatch`)。RLVR 侧等 B。 2026-09-13:A100.40gb 全满、原 job 9916589 排队 >14h 未起,改为同一数组按 GPU 类型分投(gpuq 3g.40gb=27170、gpuq A100.80gb=27169、contrib-gpuq 3g.40gb=27171,sbatch 内 `$OUT.lock` 原子认领防重复);frame_matched×2 / exact_iso×2 已在 MIG 上跑。

## A2b:批次二裁决 —— SFT 侧 ✅ 2026-09-13:8 run 全部完成(MIG 约 2 h/run)。**H-2 成立**(full / frame_matched / exact_iso 两两差 ≤0.015 / 0.005,|t|<1)→ C4 成立;**H-1 在 lr\* 复现**(spectrum − random +0.014 / −0.001);r 维干预的任务指标对 lr 十倍变化不敏感;无一方占优(r 维 MMLU +0.021 但 GSM8K −0.03,B 复核) → C5 SFT 维持阴性初判。登记处 E-v3-2/3 已填,`figs/v3/fig1_*` 已出。批次三 SFT(E-v3-4 第三 seed ×2 + E-v3-5 s_rel {3,10} ×4)已提交 jobs 38326/38327(`slurm_v3/batch3_sft.sbatch`)。RLVR 侧仍等 B 批次一。

## A2c:批次三裁决 —— SFT 侧 ✅ 2026-09-13:6 run 完成。**H-1 成立(3 seed,lr×1:spectrum 0.391/0.476 vs random 0.391/0.469)→ C2 成立**;**H-3 阴性(终判):s_rel 1→3→10 r 维两字典都不沿前沿移动(GSM8K 0.37–0.39,MMLU 0.46–0.48),与 dense lr×1/30 同点,dense lr\* 在其右下 → C5 SFT 阴性**。登记处 E-v3-4/5 已填,图已更新。**SFT 侧 v3 实验全部完成**(H-0/1/2 ✅,H-3 ✗)。scratch 清理规则(2026-09-13,已执行,627G→约 60G):每个 run 只保留 ckpt_000300 + log/args/eval json(E-v3-0a/H-5 要用终点 ckpt 对 base);capture/ 与中间 ckpt 在 compute_metrics 跑完后即删;smoke 目录删。B 的 RLVR 批次照同一规则。剩余:RLVR 侧(等 B 批次一 → lr\* → 批次二/三命令,可直接复用 batch2/3_sft.sbatch 改 objective)、A4 Fig.2/3、A5 写作。

## A2d:RLVR 侧 —— A 自跑 ✅ 2026-09-16(B 的 RLVR run 目录始终未到;`slurm_v3/batch_rlvr.sbatch` 0–15,MIG 3g.40gb,约 5 h/run,全部 16 个 run 09-16 10:00 UTC 前完成;评测 OOM 补丁 f0a37a3,无 run 触发降 batch)
**H-0 ✅** lr\*=2e-7 本地复核 HEALTHY(0.645/0.472,回落 0.018)。**H-1 ✅**(spectrum − random +0.008/0.000)→ C2 两范式成立。**H-2 ✗:exact_iso GSM8K 0.696 > dense 0.645 (t=2.4) > frame_matched 0.619,MMLU 0.464/0.472/0.473 在 1 SE 内**;reward 轨迹从第 100 步起分开 → C4 只在 SFT 成立,RLVR 上谱拉平占优。**H-3 ✗**:r 维在 lr\* 被 dense 占优(GSM8K −0.08/−0.09,t≈−4,MMLU 同),s_rel/lr 增大时 r 维响应(与 SFT 相反)但到 lr×1 仍在 dense 之下 → C5 阴性(RLVR,dense 占优)。全部登记在 `paper/results_draft.md` E-v3-2/3/4/5 RLVR 节;`analysis_v3/verdict_rlvr.py` 生成表与检验;Fig.1 已含 RLVR。第三 seed(full/frame/iso s2)jobs 274505/274506 在跑。**待用户决定**:v3 §4 裁决表 C4 改"范式依赖"及主线措辞;A4 Fig.2/3 设计随之调整。

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
