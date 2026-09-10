# results/ — 小文件归档(每 run:log.jsonl、args.json、eval*.json、metrics.csv[、manifest.json])

按版本 × 作者组织,与 README 顶部的对应表一致:

```
results/v1/result_A/   作者A 的 v1 run(Phase 1 六格、Llama 六格、h3power、ssd_trial、eval_base_0.8b)
results/v1/result_B/   作者B 的 v1 交付(Phase 2、E1–E4、4B,36 条;README 为 B 的原始说明)
results/v2/result_A/   作者A 的 v2 输出(slurm_v2/recompute.sbatch 重算的 v2 指标、smoke)
results/v2/result_B/   作者B 的 v2 交付(results_B_v2.zip 解压于此)
```

捕获张量(capture/)与 checkpoint 不进 git:A 的在 /scratch/rhong5/spectral_runs/,B 的留在 B 本机。
图在 figs/v1、figs/v2。数值登记处:paper/results_draft.md。
