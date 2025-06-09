#!/bin/bash

#SBATCH --job-name=cross_dataset_eval
#SBATCH --account=rrg-josedolz
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-12
#SBATCH --output=logs/cross_dataset_eval/%x_%A_%a.out

source .venv/bin/activate

# Fixed parameters for evaluation
source_exp_name="gp_test_v2"
target_exp_name="cross_dataset_eval"
#datasets=(caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets sun397 ucf101 stanford_cars)
datasets=(caltech101 eurosat)
#shots=(1 2 4 8 16)
shots=(1 4)
init="ZS"
constraint="none"
backbone="RN50"
seeds=3

# Method definitions
# Method 1: Baseline with 1 curated template
cfg_base_1t="SGD_lr1e-1_B128_ep300"
templates_base_1t=1

# Method 2: Baseline with 10 averaged templates
cfg_base_10t="SGD_lr1e-1_B128_ep300"
templates_base_10t=10

# Method 3: GP with 10 templates
cfg_gp_10t="GP_opt_linear"
templates_gp_10t=10


# Build all configurations for the transfer evaluation
# 10 datasets x 9 target datasets x 5 shots x 3 methods = 1350 configurations
declare -a cfg

for source_ds in "${datasets[@]}"; do
    for target_ds in "${datasets[@]}"; do
        # Skip evaluating on the same source and target dataset
        if [ "$source_ds" == "$target_ds" ]; then
            continue
        fi

        for N in "${shots[@]}"; do
            # Config for Baseline (1 template)
            cfg+=("$seeds $source_ds $target_ds $N $cfg_base_1t $init $constraint $backbone $templates_base_1t")
            # Config for Baseline (10 templates)
            cfg+=("$seeds $source_ds $target_ds $N $cfg_base_10t $init $constraint $backbone $templates_base_10t")
            # Config for GP (10 templates)
            cfg+=("$seeds $source_ds $target_ds $N $cfg_gp_10t $init $constraint $backbone $templates_gp_10t")
        done
    done
done

# Select configuration based on array task ID
IFS=' ' read -r seeds source_ds target_ds N cfg_name init constraint bb nb_templates <<< "${cfg[$SLURM_ARRAY_TASK_ID-1]}"

# Run the selected configuration using the generalized eval script
bash scripts/eval.sh "$seeds" "$source_ds" "$target_ds" "$cfg_name" "$N" "$init" "$constraint" "$bb" "$nb_templates" "$source_exp_name" "$target_exp_name"