#!/bin/bash
# Pre-download models + datasets to shared scratch (login node has internet;
# compute nodes run with HF_HUB_OFFLINE=1).
set -e
export HF_HOME=/scratch/rhong5/dataset/hf_home
HF=$HOME/envs_spectral/bin/hf

$HF download Qwen/Qwen3.5-0.8B
$HF download Qwen/Qwen3.5-2B

$HOME/envs_spectral/bin/python - <<'EOF'
import os
from datasets import load_dataset
for split in ["train", "test"]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    print(split, len(ds))
EOF
echo "DOWNLOAD_OK"
