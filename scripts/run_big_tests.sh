#!/bin/bash

# Run comprehensive baseline and GP experiments
# Usage: ./run_big_tests.sh <experiment_name> [GPU_ID]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [GPU_ID]"
    echo "Example: $0 big_test_v1 0"
    echo ""
echo "This script will run both baseline and GP experiments for:"
echo "  - oxford_pets (reg: 0.1)"
echo "  - oxford_flowers (reg: 0.1)" 
echo "  - caltech101 (reg: 0.1)"
echo "  - dtd (reg: 0.1)"
echo "  - fgvc_aircraft (reg: 0.1)"
    exit 1
fi

EXPERIMENT_NAME=$1
GPU_ID=${2:-0}  # Default to GPU 0 if not provided

# Dataset configurations: dataset_name:regularization_value
DATASETS=(
    "oxford_pets:0.1"
    "oxford_flowers:0.1"
    "caltech101:0.1"
    "dtd:0.1"
    "fgvc_aircraft:0.1"
    "eurosat:0.1"
    "food101:0.1"
    "stanford_cars:0.1"
    "ucf101:0.1"
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