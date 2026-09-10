# Shared environment for all jobs. Source this at the top of every sbatch script.
export PROJ=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research
export HF_HOME=/scratch/rhong5/dataset/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PY=$HOME/envs_spectral/bin/python
export RUNS=/scratch/rhong5/spectral_runs
mkdir -p $RUNS
cd $PROJ
