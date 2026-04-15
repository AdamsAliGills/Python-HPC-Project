#!/bin/bash
#BSUB -J ex6_timing
#BSUB -q hpc
#BSUB -W 20
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 16
#BSUB -o hpc_logs/ex6_timing_%J.out
#BSUB -e hpc_logs/ex6_timing_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

for workers in 1 2 4 8 12 16; do
    echo "=== $workers workers ==="
    { time python3 src/main.py 32 parallel --workers $workers --dynamic --plot; } 2>&1
done