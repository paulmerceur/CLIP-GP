#!/bin/bash

DATASET=${1:-caltech101}
GPU_ID=${2:-0}

EXPERIMENT_NAME=gp_test3

# Grids
SHOTS=(1 4 8 16)
GP_LRS=(0.1)
WREGS=(500.0)
BETAS=(0.0)
# Temperature values for GP template weights
TEMPS=(5.0)

for shot in ${SHOTS[@]}; do
  #bash scripts/adapt.sh 3 $DATASET baseline_10templates  $shot RN50 0.0 0.0 0.0 0.0 $EXPERIMENT_NAME $GPU_ID

  for lr in ${GP_LRS[@]}; do
    for beta in ${BETAS[@]}; do
      for temp in ${TEMPS[@]}; do
        for wreg in ${WREGS[@]}; do
          bash scripts/adapt.sh 1 $DATASET GP_rbf $shot RN50 ${lr} ${beta} ${wreg} 0.0 $EXPERIMENT_NAME $GPU_ID
        done
      done
    done
  done
done



# Good hyperparams

# Oxford Pets
# bash scripts/adapt.sh 3 oxford_pets GP_rbf $shot RN50 0.1 0.001 500.0 5.0 $EXPERIMENT_NAME $GPU_ID

# Oxford Flowers
# bash scripts/adapt.sh 3 oxford_flowers GP_rbf $shot RN50 0.1 0.001 200.0 5.0 $EXPERIMENT_NAME $GPU_ID

# Caltech101
# bash scripts/adapt.sh 3 caltech101 GP_rbf $shot RN50 0.1 0.001 500.0 5.0 $EXPERIMENT_NAME $GPU_ID

# DTD
# bash scripts/adapt.sh 3 dtd GP_rbf $shot RN50 0.1 0.001 1000.0 5.0 $EXPERIMENT_NAME $GPU_ID

# FGVC Aircraft
# bash scripts/adapt.sh 3 fgvc_aircraft GP_rbf $shot RN50 0.1 0.001 500.0 5.0 $EXPERIMENT_NAME $GPU_ID
