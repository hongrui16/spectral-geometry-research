#!/bin/bash
# Delete the bulk v1 artefacts on scratch (capture/ tensors and checkpoints)
# for runs whose v2 metrics have been written to results/v2/result_A/<run>/
# and whose v1 small files are archived in results/v1/result_A/<run>/.
# Small files (args/log/eval/metrics) stay on scratch. Intended to be chained:
#   sbatch --dependency=afterok:<recompute job id> slurm/purge_v1_bulk.sbatch
#SBATCH --job-name=spec-purge
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=6:00:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm/common.sh

for d in $RUNS/*/; do
  run=$(basename $d)
  v1=$PROJ/results/v1/result_A/$run
  v2=$PROJ/results/v2/result_A/$run
  if [ -s $v1/metrics.csv ] && [ -s $v2/manifest.json ]; then
    before=$(du -sh $d | cut -f1)
    rm -rf $d/capture $d/ckpt_*
    echo "purged $run: $before -> $(du -sh $d | cut -f1)"
  else
    echo "kept   $run (v1 archive: $([ -s $v1/metrics.csv ] && echo yes || echo no), v2 manifest: $([ -s $v2/manifest.json ] && echo yes || echo no))"
  fi
done
df -h /scratch/rhong5 | tail -1
echo PURGE_DONE
