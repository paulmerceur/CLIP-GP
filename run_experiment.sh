#!/bin/bash

# Helper script to run CLAP experiments with proper experiment naming
# Usage: ./run_experiment.sh <experiment_name> <script_type>
# 
# Example:
#   ./run_experiment.sh test few_shot_baseline

EXPERIMENT_NAME=$1
SCRIPT_TYPE=${2:-"few_shot_baseline"}

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: $0 <experiment_name> [script_type]"
    echo ""
    echo "Available script types:"
    echo "  few_shot_baseline   - Original few-shot baseline comparison (scripts/few_shot_baseline.sh)"
    echo "  cross_dataset       - Cross-dataset transfer evaluation (scripts/cross_dataset_transfer.sh)"
    echo ""
    echo "Example:"
    echo "  $0 my_clap_test few_shot_baseline"
    exit 1
fi

echo "Running experiment: $EXPERIMENT_NAME"
echo "Script type: $SCRIPT_TYPE"
echo ""

# Create log directory for this experiment
mkdir -p "logs/$EXPERIMENT_NAME"

case $SCRIPT_TYPE in
    "few_shot_baseline")
        echo "Submitting scripts/few_shot_baseline.sh with experiment name: $EXPERIMENT_NAME"
        # Update SLURM output path and submit
        sed "s|#SBATCH --output=logs/%x_%A_%a.out|#SBATCH --output=logs/$EXPERIMENT_NAME/%x_%A_%a.out|" scripts/few_shot_baseline.sh > temp_few_shot_baseline.sh
        sbatch temp_few_shot_baseline.sh "$EXPERIMENT_NAME"
        rm temp_few_shot_baseline.sh
        ;;
    "cross_dataset")
        echo "Submitting scripts/cross_dataset_transfer.sh with experiment name: $EXPERIMENT_NAME"
        sed "s|#SBATCH --output=logs/%x_%A_%a.out|#SBATCH --output=logs/$EXPERIMENT_NAME/%x_%A_%a.out|" scripts/cross_dataset_transfer.sh > temp_cross_dataset_transfer.sh
        sbatch temp_cross_dataset_transfer.sh "$EXPERIMENT_NAME"
        rm temp_cross_dataset_transfer.sh
        ;;
    *)
        echo "Error: Unknown script type '$SCRIPT_TYPE'"
        echo "Available: few_shot_baseline, cross_dataset"
        exit 1
        ;;
esac

echo ""
echo "To monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "To parse results when complete:"
echo "  python parse_baseline_logs.py $EXPERIMENT_NAME" 