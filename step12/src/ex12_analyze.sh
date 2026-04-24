#!/bin/bash
#BSUB -J ex12_analyze
#BSUB -q hpc
#BSUB -W 10
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -o hpc_logs/ex12_analyze_%J.out
#BSUB -e hpc_logs/ex12_analyze_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p hpc_logs output/step12

{ time python3 src/step12_analyze.py \
    --input output/step12/all_buildings_results.csv \
    --plot output/step12/mean_temperature_histogram.png \
    --summary output/step12/step12_summary.txt; } 2>&1
