# v6 shared environment: external reproduction of Retaining by Doing (2510.18874) with the
# authors' code in /scratch/rhong5/rbd/code (upstream rev in /scratch/rhong5/rbd/UPSTREAM_REV).
# vllm/ray env = ~/envs_py310_new (not modified); 3 small deps live in /scratch/rhong5/rbd/pydeps.
export PROJ=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research
export RBD=/scratch/rhong5/rbd
export HF_HOME=/scratch/rhong5/dataset/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$RBD/pydeps:$RBD/code
export PY=$HOME/envs_py310_new/bin/python
export RUNS=/scratch/rhong5/rbd/runs
export WANDB_MODE=disabled
mkdir -p $RUNS
export REV=$(cut -c1-7 $PROJ/.git/$(cut -d' ' -f2 $PROJ/.git/HEAD) 2>/dev/null)
snap() { ls -d $HF_HOME/hub/models--${1//\//--}/snapshots/* | head -1; }
