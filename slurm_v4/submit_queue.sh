#!/bin/bash
# Serial submitter that respects QOS MaxSubmitPU=40 array tasks: tries each pending command in order,
# submits when it fits, and loops until all are in. Run under Monitor or nohup.
cd /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research
Q=( "sbatch --parsable --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=10-19 slurm_v4/mmlu_full.sbatch"
    "sbatch --parsable --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=20-29 slurm_v4/mmlu_full.sbatch"
    "sbatch --parsable --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=30-44 slurm_v4/mmlu_full.sbatch"
    "sbatch --parsable --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=0-15 slurm_v4/probe8_configs.sbatch"
    "sbatch --parsable --partition=contrib-gpuq --gres=gpu:3g.40gb:1 --array=16-31 slurm_v4/probe8_configs.sbatch" )
i=0
while [ $i -lt ${#Q[@]} ]; do
  J=$(${Q[$i]} 2>/dev/null)
  if [ -n "$J" ]; then echo "SUBMITTED[$i] $J :: ${Q[$i]}"; i=$((i+1)); else sleep 180; fi
done
echo QUEUE_DRAINED
