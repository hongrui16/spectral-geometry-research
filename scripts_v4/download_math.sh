#!/bin/bash
# v4 second task (MATH, Hendrycks et al.). Run ONCE on a node with internet, then everything runs offline.
# Usage: bash scripts_v4/download_math.sh
export HF_HOME=/scratch/rhong5/dataset/hf_home
unset HF_HUB_OFFLINE
$HOME/envs_spectral/bin/python - <<'PY'
from datasets import load_dataset
for name in ("EleutherAI/hendrycks_math",):   # configs: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus
    for cfg in ("algebra","counting_and_probability","geometry","intermediate_algebra","number_theory","prealgebra","precalculus"):
        d = load_dataset(name, cfg); print(name, cfg, {k: len(v) for k, v in d.items()})
d = load_dataset("HuggingFaceH4/MATH-500"); print("MATH-500", {k: len(v) for k, v in d.items()})
PY
echo "MATH cached under $HF_HOME; set HF_HUB_OFFLINE=1 again for training."
