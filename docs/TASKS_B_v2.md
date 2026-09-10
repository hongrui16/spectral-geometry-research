# 作者B 任务清单 v2(唯一入口,逐条命令版)

对应企划书 `docs/unified_paper_document_v2.md`(§9.4、§11 E4a、§14 优先级),结果交付到 `results/v2/result_B/`。
v1 清单 `docs/TASKS_B_v1.md` 已完成,不再执行。**代码用 main 分支最新提交**;v1 结果对应 tag `v1`。

```bash
cd spectral-geometry-research && git checkout main && git pull && git rev-parse --short HEAD
```
每个 shell 的环境变量与 v1 相同(`HF_HOME`、`HF_HUB_OFFLINE=1`、`RUNS`、`PY`);铁律不变:除 `--out`/`CUDA_VISIBLE_DEVICES`/`--seqs-per-microbatch 1` 外不改参数。
本轮优先级:P0 → P1 → P2 → P3 → P4 → P5 → P6。**P3 是论文主结果的唯一缺口,P0–P2 都是小活,请先做掉。**

---

## P0:三个先决问题(不跑训练,当天回复)

1. **代码版本**:v1 批次跑的是哪个 commit?`git log -1 --format=%h` 于当时的工作目录,或 `args.json` 旁若有记录。特别要确认 `scripts/eval_gsm8k.py` 是否含 commit `adf1786`(greedy 非终止修复,加显式 stop ids)。这决定所有 SFT 行(0.412、0.418、4B 0.818 等,均低于 base)是否可用。
2. **捕获顺序**:在你的 `scripts/train.py` 中确认 `self.engine.post_step()` 在 `self.instr.post_optimizer()` **之前**(v1 tag 中即如此)。然后用新脚本对一个 spectrum_only run 做自检,把打印的 `H self-check` 行发给 A:
   ```bash
   $PY analysis/compute_metrics_v2.py $RUNS/phase2_sft_spectrum_only --out-root $RUNS/results_B_v2
   ```
   期望 `median R_sigma(H)` 接近 1。v1 交付的 metrics 里它只有随机基线的 1.3–9 倍,A 已排除 bf16 存储和基漂移两种解释,剩下的在你那边。
3. **硬件与 wall-clock**:每类 run 的实际 秒/步(H100 与 A100 各一行),写进交付 README。

## P1:补评测(只评测,不训练)

```bash
# OPD 的遗忘对照(e4_opd_alpha 的 MMLU 只有 0.262,缺 none 对照)
$PY scripts/eval_mmlu.py --model $RUNS/e1_opd_adamw_s1/ckpt_000500 --limit 1000 --out $RUNS/e1_opd_adamw_s1/eval_mmlu.json
# RLVR none 的 MMLU(v1 README 里写"评测中",若已完成直接交付)
$PY scripts/eval_mmlu.py --model $RUNS/phase2_rlvr_none/ckpt_000300 --limit 1000 --out $RUNS/phase2_rlvr_none/eval_mmlu.json
# frame_only 的 MMLU(与 none 配对,RLVR 缺)
$PY scripts/eval_mmlu.py --model $RUNS/phase2_rlvr_frame_only/ckpt_000300 --limit 1000 --out $RUNS/phase2_rlvr_frame_only/eval_mmlu.json
```

## P2:用 v2 脚本重算全部 v1 捕获(CPU 或 GPU 均可,每 run 几分钟)

新增列:`R_sigma_cross`、`sig_energy_*`、`noise_energy`、`snr_cross_*`、`spearman_absC_snr_cross`、`spearman_absC_split`,以及 `manifest.json`(含 H 自检)。
```bash
for r in phase2_rlvr_none phase2_rlvr_frame_only phase2_rlvr_spectrum_only phase2_rlvr_snr_topq phase2_rlvr_mag_topq \
         phase2_sft_none  phase2_sft_frame_only  phase2_sft_spectrum_only  phase2_sft_snr_topq  phase2_sft_mag_topq \
         e1_rlvr_adamw_s1 e1_rlvr_adamw_s2 e1_rlvr_muon_s1 e1_rlvr_muon_s2 e1_rlvr_ssd_s0 e1_rlvr_ssd_s1 \
         e1_sft_adamw_s1 e1_sft_muon_s1 e1_sft_ssd_s0 e1_opd_ssd_s0 e4_sft_alpha e4_opd_alpha e4_rlvr_alpha p1_4b_sft_adamw; do
  $PY analysis/compute_metrics_v2.py $RUNS/$r --out-root $RUNS/results_B_v2
done
```
交付 `results_B_v2/<run>/{metrics.csv,manifest.json}`。**v1 的 metrics.csv 不要覆盖**(它们已归档在 `results/v1/result_B`)。

## P3:E4a 等范数干预 —— 14 run(0.8B,300 步,单卡/run,全可并行)

四种模式的定义(`specgeom/interventions_v2.py`):
- `spectrum_matched`:只保留谱分量,但放大到与完整更新相同的 Frobenius 范数(v1 的 spectrum_only 只保留了约 0.03% 的更新能量,这是 v1 H5 的步长混淆);
- `frame_matched`:只保留 frame 分量,同样匹配范数;
- `random_ext`:同维数(r 个秩一方向)的固定随机字典上的投影,匹配范数,作为"随机方向也能学吗"的对照;
- `full`:不加 `--intervention`(seed 0 直接复用 v1 的 `phase2_*_none`,只需新跑 seed 1)。

RLVR 7 个(模板;按下表替换 `--intervention`、`--seed`、`--out`):
```bash
$PY scripts/train.py --objective rlvr --lr 2e-6 --steps 300 --save-every 20 \
  --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 \
  --intervention spectrum_matched --out $RUNS/e4a_rlvr_spectrum_matched_s0
```
| --intervention | --seed | --out |
|---|---|---|
| (不加) | 1 | e4a_rlvr_full_s1 |
| spectrum_matched | 0 / 1 | e4a_rlvr_spectrum_matched_s0 / _s1 |
| frame_matched | 0 / 1 | e4a_rlvr_frame_matched_s0 / _s1 |
| random_ext | 0 / 1 | e4a_rlvr_random_ext_s0 / _s1 |

SFT 7 个(同上七种;模板去掉 `--lr/--rollouts`,目录 `e4a_sft_*`):
```bash
$PY scripts/train.py --objective sft --steps 300 --save-every 20 \
  --prompts-per-step 8 --max-new-tokens 384 --seed 0 \
  --intervention spectrum_matched --out $RUNS/e4a_sft_spectrum_matched_s0
```
每 run 评测两项 + 重算指标:
```bash
$PY scripts/eval_gsm8k.py --model $RUNS/<run>/ckpt_000300 --limit 500  --out $RUNS/<run>/eval_gsm8k.json
$PY scripts/eval_mmlu.py  --model $RUNS/<run>/ckpt_000300 --limit 1000 --out $RUNS/<run>/eval_mmlu.json
$PY analysis/compute_metrics_v2.py $RUNS/<run> --out-root $RUNS/results_B_v2
$PY scripts/summary_v2.py $RUNS/<run>
```
**健康自检(开跑 30 分钟内看 log.jsonl)**:
- 每个保存步会记录 `scale_mean`/`scale_max`(投影部分被放大的倍数)。spectrum_matched 约 30–80,random_ext 约 30–80,frame_matched 约 1.0。`scale_max` 若超过 1000 或出现 NaN/loss 发散,停掉并把 log 发给 A。
- rlvr 的 spectrum_matched / random_ext:这次 reward **应该会动**(v1 的 spectrum_only 不动是步长太小)。是否能追上 full 就是本实验要回答的问题,任何方向的结果都是结果,跑满别杀。
- 干预 run 比对照慢 20–40% 属正常。

## P4:exact_iso 基线 —— 2 run(0.8B,300 步)

每步对每个矩阵做一次 SVD 把奇异值重置为初始谱(ISO 的精确保谱版本),比 P3 慢约 2 倍。
```bash
$PY scripts/train.py --objective rlvr --lr 2e-6 --steps 300 --save-every 20 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 --intervention exact_iso --out $RUNS/e4a_rlvr_exact_iso_s0
$PY scripts/train.py --objective sft            --steps 300 --save-every 20 --prompts-per-step 8            --max-new-tokens 384 --seed 0 --intervention exact_iso --out $RUNS/e4a_sft_exact_iso_s0
```
评测与交付同 P3。

## P5:4B RLVR 三行(v1 P5 已排队的,照跑;不再新增)

`p1_4b_rlvr_adamw`、`e1_4b_rlvr_muon`、`e1_4b_rlvr_ssd` 命令见 `docs/TASKS_B_v1.md` P5,不变。**4B OPD 与 9B 全部取消。**

## P6(可选,有空卡再做):SSD 学习率扫描 —— 3 run

目的只是让 E1 基线表公平(v1 的 SSD 等效步长过小),不影响主结论。
```bash
# 在 e1_rlvr_ssd_s0 命令上改 --steps 300 与 --muon-lr,--out 对应:
--muon-lr 5e-5 → $RUNS/e6_rlvr_ssd_lr5e-5
--muon-lr 1e-4 → $RUNS/e6_rlvr_ssd_lr1e-4
--muon-lr 2e-4 → $RUNS/e6_rlvr_ssd_lr2e-4
```
各配 GSM8K eval(ckpt_000300)。

---

## 交付方式

- 每 run 五个小文件:`log.jsonl、args.json、eval*.json、metrics.csv、manifest.json`,放在 `results_B_v2/<run>/`,打包成 `results_B_v2.zip` 发给 A(A 解压到 `results/v2/result_B/`)。
- 大文件(capture/、checkpoint)留在本机不删,**P3 的 capture 尤其要留**(等范数干预的 H 自检要用)。
- 每批开跑 30 分钟内先发一次 `tail -5 log.jsonl`;P0 的三个回答单独发,不要等 P3。
- README 里写:硬件、秒/步、代码 commit、任何偏离清单的地方。

## 时间线(与 docs/unified_paper_document_v2.md §14 一致)
- 09-10:P0、P1、P2
- 09-11 到 09-14:P3(14 run)、P4
- 09-15 起:P5、P6 视卡而定
