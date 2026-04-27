#!/bin/bash
#BSUB -J q9_cuda
#BSUB -q c02613
#BSUB -W 20
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=5GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o hpc_logs/q9_%J.out
#BSUB -e hpc_logs/q9_%J.err

export PYTHONPATH=$PWD/src:$PYTHONPATH

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

echo "--- CUDA timing (NO plotting) ---"
{ time python3 src/T9/gpu.py 32 cuda; } 2>&1

echo "--- CuPy timing (optional but good) ---"
{ time python3 src/T9/gpu.py 32 cupy; } 2>&1