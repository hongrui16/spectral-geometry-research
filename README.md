# Spectral Geometry of Post-Training Gradients

Code for "When Should LLM Training Change the Spectrum?" — see
`unified_paper_document.md` for theory, hypotheses (H1–H7) and experiment
design (E1–E7).

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
