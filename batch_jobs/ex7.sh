#!/bin/bash
#BSUB -J ex7_timing
#BSUB -q hpc
#BSUB -W 20
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o hpc_logs/ex7_timing_%J.out
#BSUB -e hpc_logs/ex7_timing_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

{ time python3 src/main.py 32 jit --plot; } 2>&1