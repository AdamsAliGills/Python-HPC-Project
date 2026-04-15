#!/bin/bash
#BSUB -J parallel_imap_dynamic
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -n 10
#BSUB -o required_logs/T6/parallel_imap_dynamic_%J.out
#BSUB -e required_logs/T6/parallel_imap_dynamic_%J.err
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
/usr/bin/time -v -o required_logs/T6/parallel_imap_dynamic.log python3 src/T6/parallel_imap_dynamic.py 10 10
