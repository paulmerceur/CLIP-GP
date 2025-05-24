#!/bin/bash
#SBATCH --job-name=clap_cross_dataset
#SBATCH --account=def-josedolz
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1                 # one GPU per array task
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-461%20            # 462 tasks (7 source × 6 target × 11 config combinations), run ≤20 concurrently
#SBATCH --output=logs/%x_%A_%a.out   # %A = job-ID, %a = array-index

# ──────────────────────────────────────────────────────────────────────────────
# Cross-Dataset Transfer Evaluation Script
# 
# This script systematically evaluates cross-dataset transfer performance by:
# 1. Training models on source datasets with different few-shot configurations
# 2. Evaluating trained models on all other target datasets
# 
# For each source dataset, we test:
# - Zero-shot baseline
# - 5 few-shot settings (1,2,4,8,16) × 2 methods (ZS-LP, CLAP) = 10 configurations
# Total: 11 configurations per source dataset
# 
# For 7 source datasets × 6 target datasets × 11 configurations = 462 tasks
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────
# 0. Parse arguments and environment set-up  
# ──────────────────────────
EXPERIMENT_NAME=${1:-"cross_dataset_transfer"}
echo "Running cross-dataset transfer experiment: $EXPERIMENT_NAME"

# Change to project root directory (SLURM runs in different dir)
cd "$SLURM_SUBMIT_DIR"

# Create experiment directories
mkdir -p "output/$EXPERIMENT_NAME" "metrics/$EXPERIMENT_NAME" "logs/$EXPERIMENT_NAME"

source "${PWD}/.venv/bin/activate"

export DATA="/scratch/pmerceur/data"
export DATASETROOT="/scratch/pmerceur/data"
export TORCH_HOME="${PWD}/.cache"
export PYTHONHASHSEED=0
export EXPERIMENT_NAME="$EXPERIMENT_NAME"

# ──────────────────────────
# 1. Parameter space
# ──────────────────────────
datasets=(caltech101 dtd eurosat)

shots=(1 2 4 8 16)                     # Few-shot settings
optim="SGD_lr1e-1_B256_ep300"           # Training configuration
backbone="RN50"                         # CLIP backbone

# ──────────────────────────
# 2. Build all cross-dataset transfer configurations
#    Total: 7 source × 6 target × 11 configurations = 462 tasks
# ──────────────────────────
declare -a cfg                         # cfg[i] holds one full configuration

# For each source dataset
for source_ds in "${datasets[@]}"; do
    # For each target dataset (excluding source itself)
    for target_ds in "${datasets[@]}"; do
        if [ "$source_ds" != "$target_ds" ]; then
            # Zero-shot baseline
            cfg+=("$source_ds $target_ds SGD_lr1e-3_B1_ep1 0 ZS none $backbone")
            
            # Few-shot configurations (ZS-LP and CLAP)
            for N in "${shots[@]}"; do
                cfg+=("$source_ds $target_ds $optim $N ZS none $backbone")   # ZS-LP
                cfg+=("$source_ds $target_ds $optim $N ZS l2   $backbone")   # CLAP
            done
        fi
    done
done

echo "Total configurations: ${#cfg[@]}"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# ──────────────────────────
# 3. Parse and execute the selected configuration
# ──────────────────────────
if [ $SLURM_ARRAY_TASK_ID -ge ${#cfg[@]} ]; then
    echo "ERROR: Task ID $SLURM_ARRAY_TASK_ID exceeds available configurations (${#cfg[@]})"
    exit 1
fi

IFS=' ' read -r source_ds target_ds optim_flag N split reg bb <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"

echo "Configuration $SLURM_ARRAY_TASK_ID:"
echo "  Source dataset: $source_ds"
echo "  Target dataset: $target_ds"
echo "  Optimization: $optim_flag"
echo "  Shots: $N"
echo "  Split: $split"
echo "  Regularization: $reg"
echo "  Backbone: $bb"

# ──────────────────────────
# 4. Cross-dataset evaluation workflow
# ──────────────────────────

# Step 1: Ensure source model exists (train if needed)
echo "Step 1: Checking/training source model on $source_ds..."

# Use adapt.sh to train/verify source model exists
bash scripts/adapt.sh "0" "$source_ds" "$optim_flag" "$N" "$split" "$reg" "$bb" "$EXPERIMENT_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to train/find source model for $source_ds"
    exit 1
fi

# Step 2: Cross-dataset evaluation on target dataset
echo "Step 2: Evaluating on target dataset $target_ds..."

# Set up paths for cross-dataset evaluation
TRAINER=ADAPTER

for SEED in 1 2 3; do
    # Source model directory
    if [ "$N" -eq 0 ]; then
        # Zero-shot case
        MODELDIR="scratch/pmerceur/CLAP/output/$EXPERIMENT_NAME/${source_ds}/${optim_flag}_${split}Init_${reg}Constraint_${N}shots/seed${SEED}"
    else
        # Few-shot case
        MODELDIR="scratch/pmerceur/CLAP/output/$EXPERIMENT_NAME/${source_ds}/${optim_flag}_${split}Init_${reg}Constraint_${N}shots/seed${SEED}"
    fi
    
    # Target evaluation directory
    OUTDIR="scratch/pmerceur/CLAP/output/$EXPERIMENT_NAME/transfer_${source_ds}_to_${target_ds}/${optim_flag}_${split}Init_${reg}Constraint_${N}shots/seed${SEED}"
    
    if [ -d "$OUTDIR" ]; then
        echo "Results already exist at ${OUTDIR}, skipping..."
    else
        echo "Evaluating seed ${SEED}: ${source_ds} → ${target_ds}"
        
        # Create output directory
        mkdir -p "$OUTDIR"
        
        # Run cross-dataset evaluation
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${target_ds}.yaml \
            --config-file configs/trainers/${optim_flag}.yaml \
            --output-dir ${OUTDIR} \
            --model-dir ${MODELDIR} \
            --load-epoch 300 \
            --eval-only \
            --backbone ${bb} \
            DATASET.NUM_SHOTS ${N} \
            TRAINER.ADAPTER.INIT ${split} \
            TRAINER.ADAPTER.CONSTRAINT ${reg}
        
        if [ $? -ne 0 ]; then
            echo "WARNING: Cross-dataset evaluation failed for seed ${SEED}"
        else
            echo "Successfully completed evaluation for seed ${SEED}"
        fi
    fi
done

echo "Cross-dataset transfer evaluation completed: ${source_ds} → ${target_ds}" 