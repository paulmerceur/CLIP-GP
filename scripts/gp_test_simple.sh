#!/bin/bash

#SBATCH --job-name=test_gp_fix
#SBATCH --account=def-josedolz
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-2
#SBATCH --output=logs/gp_fix_test/%x_%A_%a.out

source .venv/bin/activate

# Quick test with fewer epochs to verify the fix
seeds=1
dataset="caltech101"
shots=4
init="ZS"
constraint="none"
backbone="RN50"
templates=10
experiment_name="gp_fix_test"

echo "Testing GP fix with dataset: $dataset"

# Run different tests based on array task ID
case $SLURM_ARRAY_TASK_ID in
    1)
        echo "=== Test 1: Regular averaging with 10 templates ==="
        bash scripts/adapt.sh "$seeds" "$dataset" "SGD_lr1e-1_B256_ep300" "$shots" "$init" "$constraint" "$backbone" "$templates" "$experiment_name"
        ;;
    2)
        echo "=== Test 2: GP with 10 templates (FIXED) ==="
        bash scripts/adapt.sh "$seeds" "$dataset" "SGD_lr1e-1_B256_ep300_GP" "$shots" "$init" "$constraint" "$backbone" "$templates" "$experiment_name"
        ;;
esac

echo "Test $((SLURM_ARRAY_TASK_ID)) completed. Check output logs for results."