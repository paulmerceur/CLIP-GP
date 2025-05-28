#!/bin/bash
#SBATCH --job-name=clap_baselines
#SBATCH --account=def-josedolz
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-100    # 10 datasets * 5 shots per config * 2 configs
# ──────────────────────────
# 0. Parse arguments and environment set-up  
# ──────────────────────────
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <experiment_name>" >&2
  exit 1
fi

EXPERIMENT_NAME=$1
echo "Running experiment: $EXPERIMENT_NAME"

# Update SLURM output path for this experiment
#SBATCH --output=logs/$EXPERIMENT_NAME/%x_%A_%a.out

source .venv/bin/activate

export TORCH_HOME="${PWD}/.cache"
export PYTHONHASHSEED=0

# ──────────────────────────
# 1. Parameter space
# ──────────────────────────
datasets=(caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets sun397 ucf101 stanford_cars)
shots=(1 2 4 8 16)
optim="SGD_lr1e-1_B256_ep300"
backbone="RN50"
nb_templates=(1 10)

# ──────────────────────────
# 2. Build the configurations
# ──────────────────────────
declare -a cfg

for ds in "${datasets[@]}"; do
  for N in "${shots[@]}"; do
    for nt in "${nb_templates[@]}"; do
      cfg+=("0 $ds $optim $N ZS none $backbone $nt")
    done
  done
done

# ──────────────────────────
# 3. Launch *exactly* the selected configuration
# ──────────────────────────
IFS=' ' read -r seed ds optim N init constraint bb nb_templates <<< "${cfg[$SLURM_ARRAY_TASK_ID]}"
bash scripts/adapt.sh "$seed" "$ds" "$optim" "$N" "$init" "$constraint" "$bb" "$nb_templates" "$EXPERIMENT_NAME"

# ──────────────────────────
# 4. Analyze the results
# ──────────────────────────
python parse_experiment_results.py "$EXPERIMENT_NAME"
