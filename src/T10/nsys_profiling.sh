#!/bin/bash
#BSUB -J T10_profile_cupy_fix
#BSUB -q c02613
#BSUB -W 10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=5GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o hpc_logs/profile_cupy_fix_%J.out
#BSUB -e hpc_logs/profile_cupy_fix_%J.err

export LSB_JOB_REPORT_MAIL=N
export PYTHONPATH=$PWD/src:$PYTHONPATH

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

echo "Profiling Cupy version"

PROFILE_OUT="hpc_logs/t10_cupy_fix_profile_${LSB_JOBID}"

nsys profile \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  -o "$PROFILE_OUT" \
  python3 src/T10/cp_fix.py 32

nsys stats --report cudaapisum    "$PROFILE_OUT.nsys-rep"
nsys stats --report gpukernsum    "$PROFILE_OUT.nsys-rep"
nsys stats --report gpumemtimesum "$PROFILE_OUT.nsys-rep"
nsys stats --report gpumemsizesum "$PROFILE_OUT.nsys-rep"