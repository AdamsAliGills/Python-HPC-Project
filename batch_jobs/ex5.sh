#!/bin/bash
#BSUB -J ex5
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 10
#BSUB -o hpc_logs/ex5_%J.out
#BSUB -e hpc_logs/ex5_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
python3 src/ex5.py 100
