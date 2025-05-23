#!/bin/bash
#SBATCH --job-name=clap_variants
#SBATCH --account=def-josedolz
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1                 # one GPU per array task
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-83%15              # 84 tasks, run ≤15 concurrently
#SBATCH --output=logs/%x_%A_%a.out   # %A = job-ID, %a = array-index
# ──────────────────────────
# 0. Parse arguments and environment set-up
# ──────────────────────────
EXPERIMENT_NAME=${1:-"variants_experiment"}
echo "Running experiment: $EXPERIMENT_NAME"

# Change to project root directory (since we're in scripts/)
cd "$(dirname "$0")/.."

# Create experiment directories
mkdir -p "output/$EXPERIMENT_NAME" "metrics/$EXPERIMENT_NAME" "logs/$EXPERIMENT_NAME"

source "${PWD}/.venv/bin/activate"

export DATA="/scratch/pmerceur/data"
export DATASETROOT="/scratch/pmerceur/data"
export TORCH_HOME="${PWD}/.cache"
export PYTHONHASHSEED=0
export EXPERIMENT_NAME="$EXPERIMENT_NAME"

# ──────────────────────────
# 1. Parameter space for CLAP analysis
# ──────────────────────────
datasets=(caltech101 dtd eurosat fgvc_aircraft oxford_flowers oxford_pets sun397)
shots=(1 2 4 8 16)                         # zero-shot handled separately
optim_zs="SGD_lr1e-1_B256_ep300"           # Original config for ZS-LP
optim_clap="SGD_lr1e-2_B256_ep300_CLAP"    # Lower LR config for CLAP
backbone="RN50"

# Test different CLAP constraint types
constraint_types=("l2" "l2_constant")  # Basic CLAP vs constant constraint

# ──────────────────────────
# 2. Build configurations
# ──────────────────────────
declare -a cfg

for ds in "${datasets[@]}"; do
  # Zero-shot baseline
  cfg+=("0 $ds SGD_lr1e-3_B1_ep1 0 ZS none $backbone")
  
  for N in "${shots[@]}"; do
    # ZS-LP baseline
    cfg+=("0 $ds $optim_zs $N ZS none $backbone")
    
    # Test different CLAP variants
    for constraint in "${constraint_types[@]}"; do
      cfg+=("0 $ds $optim_clap $N ZS $constraint $backbone")
    done
  done
done

# ──────────────────────────
# 3. Launch configuration
# ──────────────────────────
IFS=' ' read -r seed ds optim_flag N split reg bb <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"
bash adapt.sh "$seed" "$ds" "$optim_flag" "$N" "$split" "$reg" "$bb" "$EXPERIMENT_NAME" 