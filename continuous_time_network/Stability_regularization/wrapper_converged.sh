#!/bin/bash
# HTCondor job wrapper: one convergence-assured stability-regularisation run.
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /data/bfys/gscriven/Van_Der_Pole/continuous_time_network/Stability_regularization
exec /data/bfys/gscriven/conda/envs/TE/bin/python stability_converged.py \
    --arm "$1" --horizon "$2" --seed "$3"
