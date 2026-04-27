#!/bin/bash
#BSUB -J T9_profile_cuda
#BSUB -q c02613
#BSUB -W 10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=5GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o hpc_logs/profile_cuda_%J.out
#BSUB -e hpc_logs/profile_cuda_%J.err

export LSB_JOB_REPORT_MAIL=N
export PYTHONPATH=$PWD/src:$PYTHONPATH

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

echo "Profiling CUDA version"
nvidia-smi

nsys profile \
  --trace=cuda,osrt \
  --stats=true \
  -o hpc_logs/t9_cuda_profile \
  python3 src/T9/gpu.py 32 cuda