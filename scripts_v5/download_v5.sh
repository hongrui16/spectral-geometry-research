#!/bin/bash
# v5 stage 0: retention suite + second-task candidates + stage-3 teacher. Run once on the login
# node (internet); everything else runs with HF_HUB_OFFLINE=1.
export HF_HOME=/scratch/rhong5/dataset/hf_home
unset HF_HUB_OFFLINE
$HOME/envs_spectral/bin/python - <<'PY'
from datasets import load_dataset
def show(name, *a, **k):
    d = load_dataset(name, *a, **k); print(name, a, {s: len(v) for s, v in d.items()}, flush=True)
show("allenai/ai2_arc", "ARC-Challenge")
show("Rowan/hellaswag")
show("mandarjoshi/trivia_qa", "rc.nocontext")
show("openlifescienceai/medmcqa")
PY
echo "V5_DATA_OK"
