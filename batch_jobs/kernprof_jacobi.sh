#!/bin/bash
#BSUB -J sleeper
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -n 1
#BSUB -o sleeper_%J.out
#BSUB -e sleeper_%J.err
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
kernprof -lb src/kernprof_jacobi.py 1
