# results/v1/result_B/ — 作者B 的 v1 交付小文件（原目录名 results_B/，2026-09-09 归档）(每 run:log.jsonl、args.json、eval*.json、metrics.csv、summary.txt)

硬件:0.8B 训练在 1×H100 80GB/run(P0、E1 大部分)或 1×A100 80GB/run(E2、E3、E4、E1 SFT/OPD s1、4B);
Qwen3.5 linear-attn 走 transformers 的 torch fallback,RLVR 约 30 s/step(H100)/40 s/step(A100)。
参数与 `docs/TASKS_B_v1.md` 完全一致,未改动。评测口径:GSM8K test 前 500 题 greedy pass@1(`--limit 500`);
MMLU 为 1000 题 zero-shot letter-logit。**base(0.8B,同口径 500 题)= 0.546**(A 的前 200 题口径为 0.575);base MMLU 见 `eval_base_0.8b/eval_mmlu.json`。

## 已完成(2026-09-09 15:45)

### P0 Phase 2(0.8B,300 步,ckpt_000300)
| intervention | RLVR | SFT |
|---|---|---|
| none(full) | **0.636** | **0.412** |
| frame_only | **0.636** | 0.406 |
| spectrum_only | 0.566 | 0.354 |
| snr_topq q=0.1 | 0.544 | 0.384 |
| mag_topq q=0.1 | 0.558 | 0.382 |

### E1 主表(0.8B,500 步,ckpt_000500)
| objective | AdamW | Muon | SSD | SSD-Muon |
|---|---|---|---|---|
| RLVR | 0.670 / 0.694 (s1/s2) | 0.604 / 0.624 | 0.568 / 0.564 / 0.560 (s0/s1/s2) | 0.584 (s0) |
| SFT | 0.438 (s1) | 0.356 (s1) | 0.420 / 0.404 (s0/s1) | — |
| OPD | 0.544 (s1) | 0.470 (s1) | 0.558 / 0.556 (s0/s1) | — |
判定与 caveat 见 `docs/B_results_phase2_E1E4.md`。

### E2 SSD 消融(rlvr 300 步):k64 0.552 · signvariant 0.558 · noalign 0.574 · notail 0.552
### E3 分层 SSD(rlvr 300 步):layers0-5 0.584 · 6-11 0.580 · 12-17 0.618 · 18-23 0.600
### E4 自适应 α(300 步):sft 0.416(MMLU 0.365) · opd 0.556(MMLU 0.262) · rlvr 0.626
对照 MMLU:base 0.483;SFT none 0.366 / frame_only 0.369(SFT 本身掉 12 点,α 没有减少遗忘);RLVR 对照 MMLU 评测中
### P5 4B:base 0.862;SFT AdamW 0.818;RLVR AdamW/Muon/SSD 与 base 排队(A100-80GB,seqs-per-microbatch 1);OPD-4B 需 2 卡尚未跑
