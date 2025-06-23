#!/bin/bash

NUM_SHOTS=${1:-4}
GPU_ID=${2:-0}

EXPERIMENT_NAME=gp_test1

GP_LR=(0.01)
GP_BETA=(0.01)

#bash scripts/adapt.sh 1 caltech101 baseline_1template    $NUM_SHOTS RN50 0.0 0.0 $EXPERIMENT_NAME $GPU_ID
#bash scripts/adapt.sh 1 caltech101 baseline_10templates  $NUM_SHOTS RN50 0.0 0.0 $EXPERIMENT_NAME $GPU_ID
for i in ${!GP_LR[@]}; do
    for j in ${!GP_BETA[@]}; do
        #bash scripts/adapt.sh 1 caltech101 GP_rbf_length1    $NUM_SHOTS RN50 ${GP_LR[$i]} ${GP_BETA[$j]} $EXPERIMENT_NAME $GPU_ID
        #bash scripts/adapt.sh 1 caltech101 GP_rbf_length1e-1 $NUM_SHOTS RN50 ${GP_LR[$i]} ${GP_BETA[$j]} $EXPERIMENT_NAME $GPU_ID
        bash scripts/adapt.sh 1 caltech101 GP_rbf_length1e-2 $NUM_SHOTS RN50 ${GP_LR[$i]} ${GP_BETA[$j]} $EXPERIMENT_NAME $GPU_ID
    done
done
