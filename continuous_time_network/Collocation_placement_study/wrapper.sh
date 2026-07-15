#!/bin/bash
# HTCondor job wrapper: one collocation-placement-study run.
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /data/bfys/gscriven/Van_Der_Pole/continuous_time_network/Collocation_placement_study
exec /data/bfys/gscriven/conda/envs/TE/bin/python run_one.py "$@"
