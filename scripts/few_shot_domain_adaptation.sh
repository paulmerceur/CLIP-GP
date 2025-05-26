#!/bin/bash
#SBATCH --job-name=clap_domain_adaptation
#SBATCH --account=def-josedolz
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-59%20    # 3 datasets * 2 target datasets * 5 shots * 2 methods = 60 combinations
#SBATCH --output=logs/%x_%A_%a.out

# ──────────────────────────
# 0. Parse arguments and environment set-up  
# ──────────────────────────
EXPERIMENT_NAME=${1:-"domain_adaptation_experiment"}
echo "Running domain adaptation experiment: $EXPERIMENT_NAME"

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
# 1. Parameter space for domain adaptation
# ──────────────────────────
#datasets=(imagenet caltech101 dtd eurosat fgvc_aircraft oxford_flowers oxford_pets sun397 food101 imagenet_a imagenet_r imagenet_sketch)
datasets=(caltech101 dtd eurosat)
shots=(1 2 4 8 16) # Few-shot settings
optim="SGD_lr1e-1_B256_ep300"
backbone="RN50"

# ──────────────────────────
# 2. Build all source->target combinations
#    Each configuration: "seed source_dataset target_dataset optim shots method backbone"
# ──────────────────────────
declare -a cfg

for source_ds in "${datasets[@]}"; do
  for target_ds in "${datasets[@]}"; do
    # Skip same dataset (not domain adaptation)
    if [ "$source_ds" = "$target_ds" ]; then
      continue
    fi
    
    # For each source->target pair, test both methods across all shot settings
    for N in "${shots[@]}"; do
      cfg+=("0 $source_ds $target_ds $optim $N ZS none $backbone")   # ZS-LP
      cfg+=("0 $source_ds $target_ds $optim $N ZS l2   $backbone")   # CLAP
    done
  done
done

# ──────────────────────────
# 3. Launch the selected configuration
# ──────────────────────────
IFS=' ' read -r seed source_ds target_ds optim_flag N split reg bb <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"

echo "Configuration $SLURM_ARRAY_TASK_ID: Training on $source_ds, testing on $target_ds with $N shots using method $split-$reg"

# First, train the model on the source dataset
echo "Step 1: Training adapter on source dataset: $source_ds"
for SEED in 1 2 3; do
    SOURCE_DIR=output/${EXPERIMENT_NAME}/transfer_${source_ds}_to_${target_ds}/${optim_flag}_${split}Init_${reg}Constraint_${N}shots/seed${SEED}
    
    if [ -d "$SOURCE_DIR" ]; then
        echo "Source model already exists at ${SOURCE_DIR}, skipping training"
    else
        echo "Training source model: $source_ds -> $SOURCE_DIR"
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ADAPTER \
        --dataset-config-file configs/datasets/${source_ds}.yaml \
        --config-file configs/trainers/${optim_flag}.yaml \
        --output-dir ${SOURCE_DIR} \
        --backbone ${bb} \
        DATASET.NUM_SHOTS ${N} \
        TRAINER.ADAPTER.INIT ${split} \
        TRAINER.ADAPTER.CONSTRAINT ${reg}
    fi
done

# Second, evaluate the trained model on the target dataset
echo "Step 2: Evaluating on target dataset: $target_ds"
for SEED in 1 2 3; do
    SOURCE_DIR=output/${EXPERIMENT_NAME}/transfer_${source_ds}_to_${target_ds}/${optim_flag}_${split}Init_${reg}Constraint_${N}shots/seed${SEED}
    TARGET_DIR=output/${EXPERIMENT_NAME}/transfer_${source_ds}_to_${target_ds}/${optim_flag}_${split}Init_${reg}Constraint_${N}shots/seed${SEED}
    
    # Check if evaluation has already been done (look for evaluation results in log)
    if [ -f "${TARGET_DIR}/log.txt" ] && grep -q "acc_test" "${TARGET_DIR}/log.txt"; then
        echo "Target evaluation already completed at ${TARGET_DIR}, skipping"
    else
        echo "Evaluating: $source_ds -> $target_ds at $TARGET_DIR"
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ADAPTER \
        --dataset-config-file configs/datasets/${target_ds}.yaml \
        --config-file configs/trainers/${optim_flag}.yaml \
        --output-dir ${TARGET_DIR} \
        --model-dir ${SOURCE_DIR} \
        --load-epoch 300 \
        --eval-only \
        --backbone ${bb} \
        DATASET.NUM_SHOTS ${N} \
        TRAINER.ADAPTER.INIT ${split} \
        TRAINER.ADAPTER.CONSTRAINT ${reg}
    fi
done

echo "Completed domain adaptation: $source_ds -> $target_ds with $N shots using $split-$reg" 