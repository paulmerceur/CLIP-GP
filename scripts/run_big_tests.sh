#!/bin/bash

# Run comprehensive baseline and GP experiments
# Usage: ./run_big_tests.sh <experiment_name> [GPU_ID]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [GPU_ID]"
    echo "Example: $0 big_test_v1 0"
    exit 1
fi

EXPERIMENT_NAME=$1
GPU_ID=${2:-0}  # Default to GPU 0 if not provided

DATASETS=("oxford_pets" "oxford_flowers" "caltech101" "dtd" "fgvc_aircraft" "eurosat" "food101" "stanford_cars" "ucf101")

for dataset_name in "${DATASETS[@]}"; do
    ./scripts/run_baseline.sh "${EXPERIMENT_NAME}" "$dataset_name" "$GPU_ID"
    baseline_exit_code=$?
    
    if [ $baseline_exit_code -ne 0 ]; then
        log_with_timestamp "WARNING: Baseline experiment failed for $dataset_name (exit code: $baseline_exit_code)"
    else
        log_with_timestamp "✓ Baseline experiment completed for $dataset_name"
    fi
    
    ./scripts/run_gp.sh "${EXPERIMENT_NAME}" "$dataset_name" "$GPU_ID"
    gp_exit_code=$?
    
    if [ $gp_exit_code -ne 0 ]; then
        log_with_timestamp "WARNING: GP experiment failed for $dataset_name (exit code: $gp_exit_code)"
    else
        log_with_timestamp "✓ GP experiment completed for $dataset_name"
    fi
done