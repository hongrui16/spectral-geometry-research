#!/bin/bash
# User rule (2026-09-27): at most 4 of my slurm jobs (array tasks count individually) at once,
# running or pending. Submits each queued command only when it fits under the cap.
# Usage: bash slurm_v5/submit_limited.sh "sbatch ... a.sbatch" "sbatch --array=0-5%4 ... b.sbatch" ...
#   an --array=a-b command is counted as its number of tasks and must carry a %N throttle <= 4.
MAX=${MAX_JOBS:-4}
cd /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research
njobs() { squeue -u rhong5 -h -r | wc -l; }
for cmd in "$@"; do
  need=1
  if [[ $cmd =~ --array=([0-9]+)-([0-9]+)(%([0-9]+))? ]]; then
    thr=${BASH_REMATCH[4]}
    if [ -z "$thr" ] || [ "$thr" -gt "$MAX" ]; then echo "REFUSED (array needs %N<=$MAX): $cmd"; continue; fi
    need=$(( BASH_REMATCH[2] - BASH_REMATCH[1] + 1 )); [ $need -gt $MAX ] && need=$MAX
  fi
  until [ $(( $(njobs) + need )) -le $MAX ]; do sleep 60; done
  echo "[$(date +%H:%M)] SUBMIT $(eval "$cmd") :: $cmd"
  sleep 5
done
echo QUEUE_DRAINED
