# Spectral Geometry of Post-Training Gradients

Code for "When Should Post-Training Change the Spectrum?".

Versioning (docs, results, tasks and code share the same version label):
- **v1** — `docs/unified_paper_document_v1.md` (credit-assignment + SSD
  narrative, H1–H7), `docs/TASKS_B_v1.md`, `docs/EXPERIMENTS_v1.md`,
  results in `results/v1/{result_A,result_B}`, code at git tag `v1`.
- **v2** — `docs/unified_paper_document_v2.md` (conditional value of spectral
  motion; live document), `docs/TASKS_A_v2.md`, `docs/TASKS_B_v2.md`,
  results in `results/v2/{result_A,result_B}`, code in the suffixed folders
  `specgeom_v2/ analysis_v2/ scripts_v2/ slurm_v2/` (self-contained copies of
  the v1 folders plus the v2 changes; the v1 folders stay byte-identical to
  tag `v1`). Single branch: `main`.

## Layout
- `specgeom/` — library: metrics (R_spectrum, SNR, rho), instrumentation,
  Muon, SSD (M1), intervention engine (Phase 2)
- `scripts/train.py` — unified SFT / OPD / RLVR trainer with G/H/W capture
- `scripts/eval_gsm8k.py` — greedy pass@1 eval
- `analysis/` — offline metric computation + Phase 1 figures
- `slurm/` — sbatch scripts (partition gpuq, qos gpu, 1x A100.80gb each)
- `runs/` — outputs (capture files, logs, checkpoints, metrics.csv)

## Environment
- venv: `~/envs_spectral` (python 3.10, torch 2.8 cu128, transformers 5.x)
- HF cache: `/scratch/rhong5/dataset/hf_home` (models pre-downloaded;
  jobs run with `HF_HUB_OFFLINE=1`)
- Models: Qwen/Qwen3.5-0.8B (main), Qwen/Qwen3.5-2B (OPD teacher, later scale-up)

## Quick start
```bash
sbatch slurm/smoke.sbatch          # 3-objective pipeline smoke test
sbatch --array=0-3 slurm/phase1.sbatch   # Phase 1 main runs
python analysis/plot_phase1.py --runs runs/phase1_{sft,opd,rlvr}_adamw \
    --labels SFT OPD RLVR --out figs/
```
