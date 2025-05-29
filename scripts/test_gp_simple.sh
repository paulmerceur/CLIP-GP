#!/bin/bash

#SBATCH --job-name=test_gp_simple
#SBATCH --account=def-josedolz
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-12
#SBATCH --output=logs/test_gp/%x_%A_%a.out

source .venv/bin/activate

# Fixed parameters for quick testing
seeds=3
dataset="caltech101"  # Small dataset for quick testing
init="ZS"
constraint="none"
backbone="RN50"
experiment_name="test_gp"
optim="SGD_lr1e-1_B256_ep300"
optim_gp="SGD_lr1e-1_B256_ep300_GP"

# Get the current array task ID
task_id=$SLURM_ARRAY_TASK_ID

# Calculate which configuration to run based on array task ID
# Each shot level has 3 configurations (1 template, 5 templates baseline, 5 templates GP)
shots_per_config=3
shot_level=$(( (task_id - 1) / shots_per_config + 1 ))
config_type=$(( (task_id - 1) % shots_per_config + 1 ))

# Map shot level to actual number of shots
case $shot_level in
    1) shots=1 ;;
    2) shots=2 ;;
    3) shots=4 ;;
    4) shots=8 ;;
esac

# Map config type to template count and optimizer
case $config_type in
    1) templates=1; current_optim=$optim ;;      # Single template baseline
    2) templates=10; current_optim=$optim ;;      # 10 templates baseline
    3) templates=10; current_optim=$optim_gp ;;   # 10 templates GP
esac

# Run the selected configuration
bash scripts/adapt.sh "$seeds" "$dataset" "$current_optim" "$shots" "$init" "$constraint" "$backbone" "$templates" "$experiment_name"