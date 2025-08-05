#!/bin/bash

# Run both baseline and GP experiments for comparison
# Usage: ./run_comparison.sh <experiment_name> [L2_LAMBDA] [GPU_ID]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [L2_LAMBDA] [GPU_ID]"
    echo "Example: $0 comparison_test 0.1 0"
    exit 1
fi

EXPERIMENT_NAME=$1
L2_LAMBDA=${2:-0.1}
GPU_ID=${3:-0}

echo "Running comparison experiments..."
echo "This will run both baseline and GP methods"
echo ""

# Run baseline first
echo "=== Running Baseline ==="
./scripts/run_baseline.sh "${EXPERIMENT_NAME}" "$L2_LAMBDA" "$GPU_ID"

echo ""
echo "=== Running GP Method ==="
./scripts/run_gp.sh "${EXPERIMENT_NAME}" "$L2_LAMBDA" "$GPU_ID"

echo ""
echo "=== Analyzing Results ==="
python analyze_experiment.py "$EXPERIMENT_NAME"

echo ""
echo "Comparison completed!"
echo "Check plots/ directory for visualizations"
