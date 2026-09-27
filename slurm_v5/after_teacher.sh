#!/bin/bash
# Wait for both 2B-teacher shards, merge them into teacher2b.jsonl, then feed the SFT pilot
# through the 4-job-cap submitter.
D=/scratch/rhong5/spectral_runs/v5/data
cd /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research
until [ -f $D/teacher2b.shard0.jsonl.stats.json ] && [ -f $D/teacher2b.shard1.jsonl.stats.json ]; do sleep 120; done
cat $D/teacher2b.shard0.jsonl $D/teacher2b.shard1.jsonl > $D/teacher2b.jsonl
echo "merged $(wc -l < $D/teacher2b.jsonl) rows"
cp $D/teacher2b.shard*.stats.json results/v5/result_A/
CMDS=()
for i in 0 1 2 3; do
  CMDS+=("sbatch --parsable --array=$i-$i%1 --time=8:00:00 --export=ALL,CFG=slurm_v5/configs/pilot_sft.txt slurm_v5/train.sbatch")
done
bash slurm_v5/submit_limited.sh "${CMDS[@]}"
