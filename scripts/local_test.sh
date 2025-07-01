#!/bin/bash

NUM_SHOTS=${1:-4}
GPU_ID=${2:-0}

EXPERIMENT_NAME=gp_test

# Grids
GP_LRS=(0.003 0.005 0.01)
BETAS=(0.05 0.1 0.2)
LAMBDA_RES=(0.001 0.01)

# Shorter training schedule
MAX_EPOCH=100

#bash scripts/adapt.sh 1 caltech101 baseline_1template    $NUM_SHOTS RN50 0.0 0.0 0.01 10 $EXPERIMENT_NAME $GPU_ID
#bash scripts/adapt.sh 1 caltech101 baseline_10templates  $NUM_SHOTS RN50 0.0 0.0 $EXPERIMENT_NAME $GPU_ID $MAX_EPOCH

# Loop over all combinations of length-scale, LR, and Beta
for lr in ${GP_LRS[@]}; do
  for beta in ${BETAS[@]}; do
    for lam in ${LAMBDA_RES[@]}; do
      bash scripts/adapt.sh 1 caltech101 GP_rbf $NUM_SHOTS RN50 ${lr} ${beta} ${lam} $EXPERIMENT_NAME $GPU_ID $MAX_EPOCH
    done
  done
done
