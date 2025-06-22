#!/bin/bash

NUM_SHOTS=${1:-4}
GPU_ID=${2:-0}

bash scripts/adapt.sh 1 caltech101 baseline_10templates $NUM_SHOTS RN50 gp_test3 $GPU_ID # 89.9 / 3.11
bash scripts/adapt.sh 1 caltech101 GP_rbf_length5 $NUM_SHOTS RN50 gp_test3 $GPU_ID # 88.9 / 2.64
bash scripts/adapt.sh 1 caltech101 GP_rbf_length10 $NUM_SHOTS RN50 gp_test3 $GPU_ID # 88.9 / 2.68
bash scripts/adapt.sh 1 caltech101 GP_rbf_length1 $NUM_SHOTS RN50 gp_test3 $GPU_ID # 88.9 / 2.64
bash scripts/adapt.sh 1 caltech101 GP_rbf_length1e-2 $NUM_SHOTS RN50 gp_test3 $GPU_ID # 88.9 / 2.64
