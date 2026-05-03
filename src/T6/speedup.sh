#!/bin/bash
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613
# log showing timg form bash schedular perspective
# while /usr/bin/time or just time in a normal linux machine
# times the specific commands within that script considered higher fidelity
# souce this script dont run as a bsub < ....sh
QUEUE="hpc"
WALLTIME="2:00"
MEM="1.5GB"
MODEL="XeonGold6226R"

for N in {1..16}; do
  bsub \
    -q "$QUEUE" \
    -W "$WALLTIME" \
    -n "$N" \
    -R "rusage[mem=$MEM]" \
    -R "select[model==$MODEL]" \
    -R "span[hosts=1]" \
    -J "chunked_parallel_n$N" \
    -o "job_%J_n$N.out" \
    -e "job_%J_n$N.err" \
    "/usr/bin/time -v -o required_logs/T6/plans_10/parallel_core_$N.log python3 src/T6/parallel_imap_dynamic.py 32 $N"
done
