#!/bin/bash
#BSUB -J parallel_map_static
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -n 16
#BSUB -o required_logs/T5/parallel_map_static_%J.out
#BSUB -e required_logs/T5/parallel_map_static_%J.err
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
/usr/bin/time -v -o required_logs/T5/parallel_map_static.log python3 src/T5/parallel_map_static.py 30 16
