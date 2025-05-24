#!/bin/bash
#SBATCH --job-name=clap_baselines
#SBATCH --account=def-josedolz
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1                 # one GPU per array task
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-43%22
#SBATCH --output=logs/%x_%A_%a.out   # %A = job-ID, %a = array-index
# ──────────────────────────
# 0. Parse arguments and environment set-up  
# ──────────────────────────
EXPERIMENT_NAME=${1:-"default_experiment"}
echo "Running experiment: $EXPERIMENT_NAME"

# Change to project root directory (SLURM runs in different dir)
cd "$SLURM_SUBMIT_DIR"

# Create experiment directories
mkdir -p "output/$EXPERIMENT_NAME" "metrics/$EXPERIMENT_NAME" "logs/$EXPERIMENT_NAME"

# Update SLURM output path for this experiment
#SBATCH --output=logs/$EXPERIMENT_NAME/%x_%A_%a.out

source "${PWD}/.venv/bin/activate"

export DATA="/scratch/pmerceur/data"
export DATASETROOT="/scratch/pmerceur/data"
export TORCH_HOME="${PWD}/.cache"
export PYTHONHASHSEED=0
export EXPERIMENT_NAME="$EXPERIMENT_NAME"

# ──────────────────────────
# 1. Parameter space
# ──────────────────────────
#datasets=(caltech101 dtd eurosat fgvc_aircraft oxford_flowers oxford_pets sun397)
datasets=(imagenet_a imagenet_r imagenet_sketch)
shots=(1 2 4 8 16)                         # zero-shot handled separately
optim="SGD_lr1e-1_B256_ep300"
backbone="RN50"

# ──────────────────────────
# 2. Build the 77 configurations
#    (index == $SLURM_ARRAY_TASK_ID)
# ──────────────────────────
declare -a cfg                         # cfg[i] holds one full CLI line

for ds in "${datasets[@]}"; do
  # 2-A  Zero-shot
  cfg+=("0 $ds SGD_lr1e-3_B1_ep1 0 ZS none $backbone")

  # 2-B  Few-shot (ZS-LP and CLAP)
  for N in "${shots[@]}"; do
    cfg+=("0 $ds $optim $N ZS none $backbone")   # ZS-LP
    cfg+=("0 $ds $optim $N ZS l2   $backbone")   # CLAP
  done
done

# ──────────────────────────
# 3. Launch *exactly* the selected configuration
# ──────────────────────────
IFS=' ' read -r seed ds optim_flag N split reg bb <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"
bash scripts/adapt.sh "$seed" "$ds" "$optim_flag" "$N" "$split" "$reg" "$bb" "$EXPERIMENT_NAME"
