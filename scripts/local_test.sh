#!/bin/bash

DATASET=${1:-caltech101}
GPU_ID=${2:-0}

EXPERIMENT_NAME=gp_test9

# Grids
SHOTS=(1 4 8 16)
GP_LRS=(0.1)
BETAS=(0.01)
W_REGS=(0.01 0.001 0.0001)

for shot in ${SHOTS[@]}; do
  bash scripts/adapt.sh 3 $DATASET baseline_10templates  $shot RN50 0.0 0.0 0.0 $EXPERIMENT_NAME $GPU_ID

  for lr in ${GP_LRS[@]}; do
    for beta in ${BETAS[@]}; do
      for w_reg in ${W_REGS[@]}; do
        bash scripts/adapt.sh 3 $DATASET GP_rbf $shot RN50 ${lr} ${beta} ${w_reg} $EXPERIMENT_NAME $GPU_ID
      done
    done
  done
done
