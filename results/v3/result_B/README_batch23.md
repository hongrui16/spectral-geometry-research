# RLVR 批次二 + 批次三 交付说明(作者 B)

14 个 run 全部完成:0 失败、0 次评测被跳过、0 次驱动故障(与批次一的五次故障不同)。

## 环境

| 项 | 值 |
|---|---|
| GPU | A100-SXM4-80GB |
| torch | 2.11.0+cu128 |
| transformers | **5.16.1** |
| 代码 | eb3f692 + `run_evals` 容错补丁(1ddc9f9);14 个 run 均在补丁之后启动,NAS 上的代码与仓库 HEAD 一致 |
| 环境变量 | `HAS_TT_DATA=False`(`launch.sh --envs` 与 `env.sh` 两层) |
| 模板 | rlvr、300 步、save-every 20、prompts 8、rollouts 8、max-new-tokens 384、eval-every 50 |


## 每个 run 的终值与 ckpt dtype 核实

base:GSM8K 0.546 / MMLU 0.483。`dtype` 列为 `ckpt_000300` safetensors 自查结果。

| run | GSM8K | MMLU | dtype |
|---|---|---|---|
| v3_rlvr_frame_matched_lr0.1_s0 | 0.650 | 0.471 | F32 OK |
| v3_rlvr_frame_matched_lr0.1_s1 | 0.644 | 0.471 | F32 OK |
| v3_rlvr_exact_iso_lr0.1_s0 | 0.706 | 0.466 | F32 OK |
| v3_rlvr_exact_iso_lr0.1_s1 | 0.646 | 0.478 | F32 OK |
| v3_rlvr_spectrum_matched_lr0.1_s0 | 0.554 | 0.480 | F32 OK |
| v3_rlvr_spectrum_matched_lr0.1_s1 | 0.546 | 0.478 | F32 OK |
| v3_rlvr_random_ext_lr0.1_s0 | 0.556 | 0.475 | F32 OK |
| v3_rlvr_random_ext_lr0.1_s1 | 0.558 | 0.478 | F32 OK |
| v3_rlvr_spectrum_matched_lr1_s2 | 0.614 | 0.478 | F32 OK |
| v3_rlvr_random_ext_lr1_s2 | 0.624 | 0.475 | F32 OK |
| v3_rlvr_spectrum_matched_lr0.1_srel3_s0 | 0.550 | 0.481 | F32 OK |
| v3_rlvr_random_ext_lr0.1_srel3_s0 | 0.560 | 0.479 | F32 OK |
| v3_rlvr_spectrum_matched_lr0.1_srel10_s0 | 0.608 | 0.473 | F32 OK |
| v3_rlvr_random_ext_lr0.1_srel10_s0 | 0.588 | 0.470 | F32 OK |

**14 / 14 为 F32。** 每个交付目录:21 个文件、14 个评测文件、`metrics.csv` 406 行、`TRAIN_RC=0`、`METRICS_RC=0`。

## health gate(`health_v3_rlvr_batch23_gate.csv`)

批次一 dense `full_lr0.1` 三个 seed 一并列入作对照。

```
cfg,n_seeds,gsm8k,mmlu,gsm8k_spread,reward_peak,reward_last100,decline,trunc_frac_last,groups_kept_last,verdict
v3_rlvr_exact_iso_lr0.1,2,0.6759999999999999,0.472,0.05999999999999994,0.6378125,0.62,0.01781250000000001,0.225625,4.0,FAIL: spread 0.060
v3_rlvr_frame_matched_lr0.1,2,0.647,0.471,0.006000000000000005,0.5671875,0.5490625,0.018125000000000058,0.13453125,4.83,HEALTHY
v3_rlvr_full_lr0.1,3,0.64,0.4696666666666667,0.028000000000000025,0.5855208333333333,0.5598958333333334,0.025625000000000047,0.13916666666666666,4.82,HEALTHY
v3_rlvr_random_ext_lr0.1,2,0.557,0.4765,0.0020000000000000018,0.3678125,0.347109375,0.020703125000000017,0.4703125,5.2,HEALTHY
v3_rlvr_random_ext_lr0.1_srel10,1,0.588,0.47,,0.514375,0.49515625,0.019218750000000007,0.18375,5.06,HEALTHY
v3_rlvr_random_ext_lr0.1_srel3,1,0.56,0.479,,0.4175,0.4003125,0.017187499999999967,0.3203125,5.66,HEALTHY
v3_rlvr_random_ext_lr1,1,0.624,0.475,,0.59875,0.57984375,0.018906250000000013,0.1028125,5.22,HEALTHY
v3_rlvr_spectrum_matched_lr0.1,2,0.55,0.479,0.008000000000000007,0.36343749999999997,0.34484375,0.018593749999999992,0.47515625,5.17,HEALTHY
v3_rlvr_spectrum_matched_lr0.1_srel10,1,0.608,0.473,,0.5390625,0.5071875,0.03187499999999999,0.15,5.02,FAIL: decline 0.032
v3_rlvr_spectrum_matched_lr0.1_srel3,1,0.55,0.481,,0.41875,0.40296875,0.015781250000000024,0.3203125,5.48,HEALTHY
v3_rlvr_spectrum_matched_lr1,1,0.614,0.478,,0.588125,0.57765625,0.010468750000000027,0.1375,5.04,HEALTHY
```

- `exact_iso_lr0.1`:**FAIL**,GSM8K seed 离散 0.060(见下方引擎缺陷)。
- `spectrum_matched_lr0.1_srel10`:**FAIL**,decline 0.032,仅超阈值 0.002,且只有 1 个 seed。
- 其余 9 个配置 HEALTHY;单 seed 配置的离散判据未被检验。
- MMLU 在单点与末三点均值两种读法下全部 PASS(`mmlu_stability_rlvr_batch23.txt`)。

## 与批次一流程的差异

1. **`exact_iso` 引擎有数值缺陷,该臂不能当作等谱对照**,详见 `docs/B_review_rlvr_batch23.md`。
2. `spectrum_matched_lr0.1_s0` 在 step 129 没有可用 rollout 组,实际做了 299 次更新(与批次一 lr×1/3 的情况相同)。
3. 未执行 A 的 scratch 清理规则:`capture/` 与 `ckpt_000250` 保留在 NAS(每 run 约 26 G + 2.9 G),exact_iso 的 capture 是缺陷的证据。
4. `rho_hist.pt`(每 run 约 98 MB)未放进仓库,留在 NAS `results_B_v3/<run>/`。
5. `worker.log` 里的 `commit=eb3f692` 是 `env.sh` 的默认值,实际代码还含 1ddc9f9 补丁。

