# 作者 B 任务清单 v6(2026-10-01,A 写)

企划书 `docs/unified_paper_document_v6.md`;数字登记 `paper/results_draft.md` "v6 登记处"。
v6 的问题:已发表的"RL 比 SFT 遗忘少"(Retaining by Doing,arXiv 2510.18874)差距里,多少是答题格式、多少是知识。
**分工(2026-10-01 更新)**:剩下的实验**全部由 B 跑**(B 那边也要用这些结果,跨组重跑本身就是独立复现);A 不再占 GPU,只做代码支持、核对 B 的数字、写短文 / 技术报告。
A 已有的结果(供 B 核对,登记处 v6 节):Qwen2.5-1.5B、训练 MMLU——SFT 非目标平均降 16.4 个点,GRPO 不降;格式提醒测试:Countdown 恢复 109%(纯格式),MATH 恢复 22%(大部分是真实损失)。

**执行顺序**:B2(训练,Qwen 与 Llama 的 SFT + GRPO,共 4 个)→ B1(Qwen 三步)→ B3(Llama 三步)。B1 是对 A 结果的独立复现;**B3 是新信息,决定写短文时结论能否推广到第二个模型**。

---

## B0 环境与数据(一次性)

**1. 作者代码 + 我们的补丁**
```bash
git clone https://github.com/princeton-pli/retaining-by-doing.git rbd_code
cd rbd_code && git checkout 1228541
git apply <本仓库>/third_party/rbd_patches.diff     # 未定义变量 data_dir;flash_attention_2 -> sdpa
```

**2. 数据**(作者 Google Drive,约 340 MB 压缩)
```bash
gdown 1IBhBH5TdVp1gxeg3BT8kgZ1j8PyCGI1l -O rbd_data.zip && unzip rbd_data.zip   # 得到 data/
ln -s $(pwd)/data rbd_code/data          # 作者代码按相对路径 data/ 读取,必须在 rbd_code 目录下运行
```

**3. 两个 Python 环境**(A 实测的版本)
| 用途 | 版本 | 说明 |
|---|---|---|
| 评测、SFT、few-shot 测试 | torch 2.8.0、vllm 0.11.0、transformers 4.57.0、ray 2.50 | 另装 `fire addict pynvml` |
| **RL 训练专用** | vllm **0.7.3**、torch 2.5.1、transformers 4.49.0、ray 2.40 | 作者的 RL 代码通过 vLLM V0 引擎内部接口同步权重(`driver_worker`、进程内 torch.distributed),vllm ≥0.8 的 V1 引擎下会报 "Default process group has not been initialized"。另装 `fire addict pynvml wandb rich pyyaml datasets` |

**4. 模型**:`Qwen/Qwen2.5-1.5B-Instruct`、`meta-llama/Llama-3.2-1B-Instruct`(HF 下载到本地)。

**已知坑**
- vllm 0.11 在 MIG 切片上会因 `CUDA_VISIBLE_DEVICES` 是 MIG UUID 而报 `invalid literal for int()`;在作业里 `export CUDA_VISIBLE_DEVICES=0` 即可(A 已验证)。
- 作者 RL 脚本默认每 20 步存一个 3.4 GB ckpt,750 步约 130 GB;只需 `final`(可选每 100 步),其余可删。

---

## B1(Qwen,独立复现 A 的结果):格式提醒测试

**目的**:SFT(训练 MMLU)后,MATH 与 Countdown 大幅下降;宽松抽取答案只救回 2–3 个点。测试在提示里放 3 道格式正确的示范题(基座自己答对的题与答案)后,SFT 模型能恢复多少。恢复一大半 → 下降主要是格式/表达;恢复很少 → 主要是真实能力损失。

**需要的 ckpt**:Qwen2.5-1.5B 的 SFT(MMLU)与 GRPO(MMLU),由 B 按 B2 自己训(SFT 2×A100 约 40 分钟,RL 2×A100 约 4.5 小时)。
**A 的对照数字**(MATH 前 1000 题去掉 3 道示范、Countdown 全部,n=997;作者打分):
| | 基座 零样本 / 示范 | RL 零样本 / 示范 | SFT 零样本 / 示范 |
|---|---|---|---|
| Countdown | 0.143 / 0.271 | 0.155 / 0.274 | 0.001 / 0.296 |
| MATH | 0.735 / 0.741 | 0.746 / 0.748 | 0.525 / 0.573 |
B 的数字与之相差 2–3 个点以内算复现成功(训练有随机性;方向和量级一致即可)。

**步骤**(都在 `rbd_code` 目录下,用评测环境):
```bash
# 1) 基座、SFT、RL 的零样本评测(作者评测;也是 few-shot 选示范题的来源)
for m in <Qwen2.5-1.5B-Instruct 路径> <SFT final> <RL final>; do
  for ds in mmlu countdown ifeval_verify math; do
    python -m core.evaluation.run --model_name "$m" --dataset_name $ds --run_dir <EVAL>/<tag> \
      --batch_size 128 --model_module_path core.vllm_utils.vLLMCausalLM --dataset_split eval --temperature 0.0
  done
done
# 2) 重新打分(严格 / 宽松 / 写完率 / 写完的题中正确率)
PYTHONPATH=$(pwd) python <本仓库>/analysis_v6/rescore.py <EVAL>/base <EVAL>/sft <EVAL>/rl --out rescore.json
# 3) 格式提醒测试(每个模型一次,示范题取自基座的零样本输出)
PYTHONPATH=$(pwd) python <本仓库>/analysis_v6/fewshot_eval.py --model <ckpt> --tag sft_qwen15 \
    --base-eval <EVAL>/base --zero-eval <EVAL>/sft --out fewshot_sft_qwen15.json
#    对 base(--zero-eval <EVAL>/base)与 RL(--zero-eval <EVAL>/rl)各跑一次
```
**MIG 上跑 vLLM**:在作业里 `export CUDA_VISIBLE_DEVICES=0`(A 已验证可用,见 `slurm_v6/fewshot_mig.sbatch`)。

**交付**:`results/v6/result_B/` 下放 `fewshot_{sft,base,rl}_qwen15.json`、`rescore.json`、各评测目录下的 `eval_metrics.json`;README 写环境版本、GPU、用的 ckpt 是自训还是 A 给的。

**判读(预先写好)**:看 SFT 的 `fewshot_strict` 相对 `zeroshot_strict_same_items` 与基座 few-shot 的差距——
恢复比例 = (SFT few-shot − SFT zero-shot) / (基座 few-shot − SFT zero-shot)。≥ 0.5 → 继续(格式为主);< 0.3 → 停(能力损失为主);中间 → 补 Llama 再定。

---

## B2(必做,先做):训练(作者脚本,只改路径)

`<本仓库>/slurm_v6/rbd_train.sbatch` 是 A 的包装:把作者的 `scripts/{sft,self_sft,rl}.sh` 复制一份,只替换代码 / 输出路径、模型路径、`model_name_short`、`dataset_name`、环境,**超参全用作者脚本默认值**;为排卡改成 2 卡,并把梯度累积 ×2 保持总批量不变(SFT 256、RL 32)。B 的集群不同的话按同样原则改路径即可。
- 4 个 run:Qwen2.5-1.5B 与 Llama-3.2-1B 各一个 `ALGO=sft DS=MMLUSFTDataset` 与 `ALGO=rl DS=MMLUDataset`(`MS=qwen-2.5-1.5b-inst` / `llama-3.2-1b-inst`;RL 用 vllm 0.7.3 环境)。
- 已知作者脚本与论文的差异:SFT 总批量脚本 256 / 论文 128;RL 学习率脚本 5e-6 / 论文写 1e-4(1B/1.5B)。A 全按脚本。

## B3(必做,新信息):Llama-3.2-1B 的同样流程

对 Llama 的基座、SFT、RL 做 B1 的同样三步。`fewshot_eval.py` 的 `--base-eval` 改指 **Llama 基座**的零样本评测目录(示范题取自同一模型的基座)。交付 `fewshot_{sft,base,rl}_llama1b.json` 与 rescore 结果。
判读:Llama 上若也是"Countdown 恢复 ≥ 0.5、MATH < 0.3",短文结论写成"格式与能力损失的占比因任务而异,跨两个模型一致";若不一致,如实写成"依模型而异"。
