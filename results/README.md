# results/ — 参考数据(小文件)

每个 run 的 log.jsonl(训练曲线,作者B 自查健康度用)、args.json(精确配置)、
metrics.csv(谱指标汇总)、eval json。**捕获张量(capture/)不在 git**——
体积太大,B→A 方向按 runbook 打包传输。figs/ 为当前图版本。

作者B 对照用:phase2 对照 run 的 rlvr 曲线参考 phase1_rlvr_adamw/log.jsonl
(reward 应从 ~0.15 爬升);SSD 参考 ssd_trial/log.jsonl。
