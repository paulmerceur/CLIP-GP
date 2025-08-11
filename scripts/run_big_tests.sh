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

# Dataset configurations: dataset_name:regularization_value
DATASETS=(
    "oxford_pets:50.0"
    "oxford_flowers:10.0"
    "caltech101:20.0"
    "dtd:20.0"
    "fgvc_aircraft:20.0"
    "eurosat:20.0"
    "food101:20.0"
    "stanford_cars:50.0"
    "ucf101:20.0"
)

for dataset_config in "${DATASETS[@]}"; do
    # Parse dataset name and regularization value
    dataset_name=$(echo $dataset_config | cut -d: -f1)
    reg_value=$(echo $dataset_config | cut -d: -f2)
    
    ./scripts/run_baseline.sh "${EXPERIMENT_NAME}" "$dataset_name" "$reg_value" "$GPU_ID"
    baseline_exit_code=$?
    
    if [ $baseline_exit_code -ne 0 ]; then
        log_with_timestamp "WARNING: Baseline experiment failed for $dataset_name (exit code: $baseline_exit_code)"
    else
        log_with_timestamp "✓ Baseline experiment completed for $dataset_name"
    fi
    
    ./scripts/run_gp.sh "${EXPERIMENT_NAME}" "$dataset_name" "$reg_value" "$GPU_ID"
    gp_exit_code=$?
    
    if [ $gp_exit_code -ne 0 ]; then
        log_with_timestamp "WARNING: GP experiment failed for $dataset_name (exit code: $gp_exit_code)"
    else
        log_with_timestamp "✓ GP experiment completed for $dataset_name"
    fi
done