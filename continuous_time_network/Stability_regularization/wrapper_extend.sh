#!/bin/bash
# HTCondor job wrapper: continue one convergence-assured run under the stricter rule
# (Adam to at least 60k epochs, plateau tolerance 1e-4 over 10k, polish cap 60 restarts).
set -e
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /data/bfys/gscriven/Van_Der_Pole/continuous_time_network/Stability_regularization
exec /data/bfys/gscriven/conda/envs/TE/bin/python stability_converged.py \
    --arm "$1" --horizon "$2" --seed "$3" \
    --extend --min-adam 60000 --plateau-tol 1e-4 --lbfgs-cap 60
