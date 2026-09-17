# 作者B 任务清单 v4(2026-09-17 生效;逐条命令版,与 v3 同格式)

> **2026-09-17 暂停通知(A):v4 项目暂停,本清单所有任务(B1 起)不必再做。** 原因与结论见企划书 `docs/unified_paper_document_v4.md` 第三部分:因果检验成立,方法 E-v4-4 判负,A 认为目前工作不足以投主会,先暂停做其他工作,后面再看。已跑出的东西若有,照旧交付到 `results/v4/result_B/` 即可。

企划书 `docs/unified_paper_document_v4.md` §2–§4。代码用 main 最新提交的 `*_v4/`(从 v3 复制,含 exact_iso 修复:CUDA SVD driver gesvda + 等范数匹配;`analysis_v4/kl_probe.py` 有任务外参考集)。交付到 `results/v4/result_B/`(每 run 一个目录,格式同 v3;README 写 transformers 版本、commit、GPU、每类 run 秒/步、ckpt dtype)。A、B 的 run 按独立样本合并(v3 已证同 seed 跨硬件不复现)。

```bash
cd spectral-geometry-research && git checkout main && git pull && git rev-parse --short HEAD
```
环境同 v3(transformers 5.16.1,fp32 主权重)。铁律:除 `--out` / `CUDA_VISIBLE_DEVICES` 与本清单写明的参数外不改任何参数。

## B1(第 1 周):2B 单元的闸门 —— Qwen3.5-2B × GSM8K × RLVR,dense lr 扫描
目的:企划书 §2.2 的"新设定前瞻":A 先用 8 步探针预测 2B 的健康 lr 档位并写进登记处,B 再跑闸门验证预测。**B 先跑下面的 8 步捕获(5 个,每个约 20 分钟,A100.80gb),把 capture 目录交给 A;A 出预测后 B 再开闸门六个。**

8 步捕获(与 v3 smoke 同设定,模型换 2B):
```bash
for INT in none frame_matched exact_iso spectrum_matched random_ext; do
$PY scripts_v4/train.py --objective rlvr --model Qwen/Qwen3.5-2B --steps 8 --save-every 4 --save-ckpt-every 1000 \
  --lr 2e-6 --prompts-per-step 8 --rollouts 8 --max-new-tokens 384 --seed 0 \
  --intervention $INT --intervention-scale 1 --eval-every 1000 --out $RUNS/klcap_v4_2b_rlvr_$INT
done
```
(`--model` 已是 train.py 的参数;2B 不加 `--kl-beta`,不会加载 teacher。)

闸门六个(A 给出预测后开):lr ∈ {×1/30, ×1/10, ×1/3} 以 2B 的 lr×1 = 2e-6 为基准,seed 0/1,`--eval-every 50`,目录 `v4_2b_rlvr_full_lr{0.033,0.1,0.33}_s{0,1}`。命令模板同 v3 批次一,加 `--model Qwen/Qwen3.5-2B`。跑完 `analysis_v4/health.py`(判据同 v3)。

## B2(第 2 周):2B 单元的方向臂(在 B1 的 lr\* 上)
frame_matched / exact_iso / spectrum_matched / random_ext × 2 seed = 8 run,同模板改 `--intervention`。目的:企划书 §2.1 的因果检验在第二个模型上重复。

## B3(第 2–3 周):方法对照(与 A 分工,A 定后通知)
企划书 §2.3 的对照里 FINCH 与 Anchored Learning 的 2B 版本,或 A 的功能步长 SFT 的 2B 版本,视 A 在 0.8B 上的结果决定是否开。

## 交付与复核
- 每批交付后 A 当天裁决;B 沿用 v3 的独立复核方式复核 A 的 0.8B 裁决(§2.1、§2.2)。
- 若 8 步捕获或闸门任何一步出现 OOM / 评测失败,按 v3 的 `run_evals` 降 batch 重试逻辑处理并在 README 注明。
