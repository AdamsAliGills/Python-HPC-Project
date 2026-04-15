#!/bin/bash
#BSUB -J sleeper
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -n 1
#BSUB -o required_logs/sleeper_%J.out
#BSUB -e sleeper_%J.err
python3 src/T1/specific_ids.py 156 1573
