#!/bin/bash
#BSUB -J ex12_run_all
#BSUB -q hpc
#BSUB -W 240
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o hpc_logs/ex12_run_all_%J.out
#BSUB -e hpc_logs/ex12_run_all_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p hpc_logs output/step12

{ time python3 src/step12_run_all.py --mode jit --output output/step12/all_buildings_results.csv; } 2>&1
