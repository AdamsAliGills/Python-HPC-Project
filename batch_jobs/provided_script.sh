#!/bin/bash
#BSUB -J provided_script
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -n 1
#BSUB -o provided_script_%J.out
#BSUB -e provided_script_%J.err
python3 src/provided_script.py 20
