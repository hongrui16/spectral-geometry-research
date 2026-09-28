#!/bin/bash
# User rule (2026-09-27, standing): at most 4 of THIS task's slurm jobs at once (running or pending;
# array tasks count individually). Only jobs named v5-* are counted -- the user's other projects
# (evRet9*, g44a*, llmGloss2, stitch9*, ...) are not ours. Every command must submit exactly ONE
# job or ONE array task (--array=i-i); anything else is refused (a multi-task array would put
# several pending tasks in the queue at once).
# Usage: bash slurm_v5/submit_limited.sh "sbatch --array=3-3 ... a.sbatch" "sbatch ... b.sbatch" ...
MAX=${MAX_JOBS:-4}
cd /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research
njobs() { squeue -u rhong5 -h -r -o "%j" | grep -c '^v5-'; }
for cmd in "$@"; do
  need=1
  if [[ $cmd =~ --array[=\ ]+([^ ]+) ]]; then
    spec=${BASH_REMATCH[1]%%%*}
    if ! [[ $spec =~ ^([0-9]+)-([0-9]+)$ ]] || [ "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]; then
      echo "REFUSED (one array task per command, use --array=i-i): $cmd"; continue
    fi
  fi
  if [[ ! $cmd =~ job-name=v5- ]] && ! grep -q "^#SBATCH --job-name=v5-" $(echo $cmd | grep -oE "slurm_v[0-9]+/[a-z0-9_]+\.sbatch" | head -1) 2>/dev/null; then
    echo "REFUSED (job name must start with v5- so the cap can count it): $cmd"; continue
  fi
  # check-and-submit under a lock so several submitters can run side by side safely
  while true; do
    exec 9>/scratch/rhong5/tmp/v5_submit.lock
    flock 9
    if [ $(( $(njobs) + need )) -le $MAX ]; then
      echo "[$(date +%H:%M)] SUBMIT $(eval "$cmd") :: $cmd"
      sleep 5; flock -u 9; break
    fi
    flock -u 9; sleep 60
  done
done
echo QUEUE_DRAINED
