#!/bin/bash

NUM_SHOTS=${1:-4}
GPU_ID=${2:-0}

EXPERIMENT_NAME=gp_test3

# Hyper-parameter grids
GP_LRS=(0.1)
GP_BETAS=(0.01)

bash scripts/adapt.sh 3 caltech101 baseline_1template    $NUM_SHOTS RN50 0.0 0.0 $EXPERIMENT_NAME $GPU_ID
bash scripts/adapt.sh 3 caltech101 baseline_10templates  $NUM_SHOTS RN50 0.0 0.0 $EXPERIMENT_NAME $GPU_ID

# Loop over all combinations of length-scale, LR, and Beta
for lr in ${GP_LRS[@]}; do
  for beta in ${GP_BETAS[@]}; do
    bash scripts/adapt.sh 3 caltech101 GP_rbf $NUM_SHOTS RN50 ${lr} ${beta} $EXPERIMENT_NAME $GPU_ID
  done
done
