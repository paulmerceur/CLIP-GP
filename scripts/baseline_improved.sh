#!/bin/bash
#SBATCH --job-name=clap_improved
#SBATCH --account=def-josedolz
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1                 # one GPU per array task
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-76%20              # 77 tasks, run ≤20 concurrently
#SBATCH --output=logs/%x_%A_%a.out   # %A = job-ID, %a = array-index
# ──────────────────────────
# 0. Parse arguments and environment set-up
# ──────────────────────────
EXPERIMENT_NAME=${1:-"improved_experiment"}
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
# 1. Parameter space
# ──────────────────────────
datasets=(caltech101 dtd eurosat fgvc_aircraft oxford_flowers oxford_pets sun397)
shots=(1 2 4 8 16)                         # zero-shot handled separately
# Test different configurations for CLAP
optim_zs="SGD_lr1e-1_B256_ep300"           # Original config for ZS-LP
optim_clap="SGD_lr1e-2_B256_ep300_CLAP"    # Improved config for CLAP
backbone="RN50"

# ──────────────────────────
# 2. Build the 77 configurations
#    (index == $SLURM_ARRAY_TASK_ID)
# ──────────────────────────
declare -a cfg                         # cfg[i] holds one full CLI line

for ds in "${datasets[@]}"; do
  # 2-A  Zero-shot (same as before)
  cfg+=("0 $ds SGD_lr1e-3_B1_ep1 0 ZS none $backbone")

  # 2-B  Few-shot with improved configurations
  for N in "${shots[@]}"; do
    cfg+=("0 $ds $optim_zs $N ZS none $backbone")    # ZS-LP with original config
    cfg+=("0 $ds $optim_clap $N ZS l2   $backbone")  # CLAP with improved config
  done
done

# ──────────────────────────
# 3. Launch *exactly* the selected configuration
# ──────────────────────────
IFS=' ' read -r seed ds optim_flag N split reg bb <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"
bash adapt.sh "$seed" "$ds" "$optim_flag" "$N" "$split" "$reg" "$bb" "$EXPERIMENT_NAME" 