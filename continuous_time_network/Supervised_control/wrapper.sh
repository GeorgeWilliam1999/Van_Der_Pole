#!/bin/bash
# HTCondor job wrapper: one supervised-control run.
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /data/bfys/gscriven/Van_Der_Pole/continuous_time_network/Supervised_control
exec /data/bfys/gscriven/conda/envs/TE/bin/python supervised.py \
    --depth "$1" --width "$2" --density "$3" --seed "$4"
