#!/bin/bash
#BSUB -J ex8_timing
#BSUB -q c02613
#BSUB -W 20
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=5GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o hpc_logs/ex8_timing_%J.out
#BSUB -e hpc_logs/ex8_timing_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

{ time python3 src/main.py 32 cuda --plot; } 2>&1