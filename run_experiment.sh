#!/bin/bash

# Helper script to run CLAP experiments with proper experiment naming
# Usage: ./run_experiment.sh <experiment_name> <script_type>
# 
# Example:
#   ./run_experiment.sh clap_fix1 baseline_improved
#   ./run_experiment.sh clap_variants baseline_clap_variants

EXPERIMENT_NAME=$1
SCRIPT_TYPE=${2:-"baseline_improved"}

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: $0 <experiment_name> [script_type]"
    echo ""
    echo "Available script types:"
    echo "  baseline           - Original baseline comparison (scripts/baseline.sh)"
    echo "  baseline_improved  - Improved CLAP with better hyperparams (scripts/baseline_improved.sh) [default]"
    echo "  baseline_clap_variants - Test different CLAP constraint variants (scripts/baseline_clap_variants.sh)"
    echo ""
    echo "Example:"
    echo "  $0 my_clap_test baseline_improved"
    exit 1
fi

echo "Running experiment: $EXPERIMENT_NAME"
echo "Script type: $SCRIPT_TYPE"
echo ""

# Create log directory for this experiment
mkdir -p "logs/$EXPERIMENT_NAME"

case $SCRIPT_TYPE in
    "baseline")
        echo "Submitting scripts/baseline.sh with experiment name: $EXPERIMENT_NAME"
        # Update SLURM output path and submit
        sed "s|#SBATCH --output=logs/%x_%A_%a.out|#SBATCH --output=logs/$EXPERIMENT_NAME/%x_%A_%a.out|" scripts/baseline.sh > temp_baseline.sh
        sbatch temp_baseline.sh "$EXPERIMENT_NAME"
        rm temp_baseline.sh
        ;;
    "baseline_improved")
        echo "Submitting scripts/baseline_improved.sh with experiment name: $EXPERIMENT_NAME"
        sed "s|#SBATCH --output=logs/%x_%A_%a.out|#SBATCH --output=logs/$EXPERIMENT_NAME/%x_%A_%a.out|" scripts/baseline_improved.sh > temp_baseline_improved.sh
        sbatch temp_baseline_improved.sh "$EXPERIMENT_NAME"
        rm temp_baseline_improved.sh
        ;;
    "baseline_clap_variants")
        echo "Submitting scripts/baseline_clap_variants.sh with experiment name: $EXPERIMENT_NAME"
        sed "s|#SBATCH --output=logs/%x_%A_%a.out|#SBATCH --output=logs/$EXPERIMENT_NAME/%x_%A_%a.out|" scripts/baseline_clap_variants.sh > temp_baseline_clap_variants.sh
        sbatch temp_baseline_clap_variants.sh "$EXPERIMENT_NAME"
        rm temp_baseline_clap_variants.sh
        ;;
    *)
        echo "Error: Unknown script type '$SCRIPT_TYPE'"
        echo "Available: baseline, baseline_improved, baseline_clap_variants"
        exit 1
        ;;
esac

echo ""
echo "To monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "To parse results when complete:"
echo "  python parse_baseline_logs.py $EXPERIMENT_NAME" 