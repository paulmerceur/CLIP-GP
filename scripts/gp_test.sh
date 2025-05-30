#!/bin/bash

#SBATCH --job-name=test_gp_simple
#SBATCH --account=def-josedolz
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-150
#SBATCH --output=logs/gp_test/%x_%A_%a.out

source .venv/bin/activate

# Fixed parameters for quick testing
seeds=3
datasets=(caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets sun397 ucf101 stanford_cars)
optim_base="SGD_lr1e-1_B256_ep300"
optim_gp="SGD_lr1e-1_B256_ep300_GP"
shots=(1 2 4 8 16)
init="ZS"
constraint="none"
backbone="RN50"
experiment_name="gp_test"

# Build configurations for testing
declare -a cfg

for ds in "${datasets[@]}"; do
  for N in "${shots[@]}"; do
    # Test 1: optim with 1 template
    cfg+=("$seeds $ds $optim_base $N $init $constraint $backbone 1")
    # Test 2: optim with 10 templates  
    cfg+=("$seeds $ds $optim_base $N $init $constraint $backbone 10")
    # Test 3: optim_gp with 10 templates
    cfg+=("$seeds $ds $optim_gp $N $init $constraint $backbone 10")
  done
done

# Select configuration based on array task ID
IFS=' ' read -r seed ds optim N init constraint bb nb_templates <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"

# Run the selected configuration
bash scripts/adapt.sh "$seed" "$ds" "$optim" "$N" "$init" "$constraint" "$bb" "$nb_templates" "$experiment_name"