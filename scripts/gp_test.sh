#!/bin/bash

#SBATCH --job-name=gp_test_v3
#SBATCH --account=rrg-josedolz
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-3
#SBATCH --output=logs/gp_test_v3/%x_%A_%a.out

source .venv/bin/activate

# Fixed parameters for quick testing
seeds=1
#datasets=(caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets sun397 ucf101 stanford_cars)
datasets=(caltech101)
optim_base="SGD_lr1e-1_B128_ep300"
optim_gp="GP_linear"
optim_gp2="GP_linear2"
shots=(4)
init="ZS"
constraint="none"
backbone="RN50"
experiment_name="gp_test_v3"

# Build configurations for testing
declare -a cfg

for ds in "${datasets[@]}"; do
	for N in "${shots[@]}"; do
		# Baseline (no GP) with 1 template
		#cfg+=("$seeds $ds $optim_base $N $init $constraint $backbone 1")
		# Baseline (no GP) with 10 templates  
		cfg+=("$seeds $ds $optim_base $N $init $constraint $backbone 10")
		# GP Linear kernel with 10 templates
		cfg+=("$seeds $ds $optim_gp $N $init $constraint $backbone 10")
		# GP Linear kernel with 10 templates
		cfg+=("$seeds $ds $optim_gp2 $N $init $constraint $backbone 10")
	done
done

# Select configuration based on array task ID
IFS=' ' read -r seeds ds optim N init constraint bb nb_templates <<< "${cfg[$SLURM_ARRAY_TASK_ID-1]}"

# Run the selected configuration
bash scripts/adapt.sh "$seeds" "$ds" "$optim" "$N" "$init" "$constraint" "$bb" "$nb_templates" "$experiment_name"