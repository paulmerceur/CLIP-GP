#!/bin/bash

#SBATCH --job-name=quick_gp_test              # Job name
#SBATCH --account=def-josedolz
#SBATCH --time=01:00:00                   # Short wall-clock time
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-2                       # 1=baseline, 2=GP
#SBATCH --output=logs/quick_gp_test/%x_%A_%a.out

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
source .venv/bin/activate                 # Your existing venv
set -e                                    # Exit on first error
seed=3

# ------------------------------------------------------------------
# Fixed experiment settings
# ------------------------------------------------------------------
dataset="caltech101"
shots=4
init="ZS"
constraint="none"
backbone="RN50"
templates=10
experiment_name="quick_gp_test"

# ------------------------------------------------------------------
# Pick config according to array index
# ------------------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    # ---------- Baseline (no GP) ----------
    USE_GP="False"
    GP_LR=0.1
    GP_INIT_STD=1e-2
    QUALITY_SCALE=2.0
    TEMP_INIT=1.5
    REG_SCALE=1.0
    DESC="baseline_no_gp"
else
    # ---------- GP-Weighted ----------
    USE_GP="True"
    GP_LR=0.01           # learning rate for GP parameters
    GP_INIT_STD=0.1         # initialization std for template weights
    QUALITY_SCALE=2.0       # scaling factor for quality scores
    TEMP_INIT=1.5           # initial temperature
    REG_SCALE=1.0           # regularization scale
    DESC="with_gp_weighting"
fi

echo "=== GAUSSIAN PROCESS TEMPLATE WEIGHTING TEST ==="
echo "Run: $DESC   (array id $SLURM_ARRAY_TASK_ID)"
echo "Dataset: $dataset   Shots: $shots   GP Enabled: $USE_GP"

# ------------------------------------------------------------------
# Create a tiny on-the-fly YAML config
# ------------------------------------------------------------------
CONFIG_FILE="configs/trainers/quick_gp_test_${SLURM_ARRAY_TASK_ID}.yaml"
cat > "$CONFIG_FILE" <<EOF
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 128
  TEST:
    BATCH_SIZE: 256
  NUM_WORKERS: 4

OPTIM:
  NAME: "sgd"
  LR: 0.05
  MAX_EPOCH: 30          # just enough epochs to verify learning
  LR_SCHEDULER: "cosine"
  WEIGHT_DECAY: 0.0

TRAINER:
  ADAPTER:
    USE_GP: $USE_GP
    GP_LR: $GP_LR
    GP_BETA: 0.1
    GP_KERNEL_TYPE: "rbf"
    GP_LENGTHSCALE: 1.0
    GP_OUTPUTSCALE: 1.0
    GP_NOISE_VAR: 1e-4
    GP_NUM_MC_SAMPLES: 3
    GP_USE_DIAGONAL_COV: True
EOF

# ------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------
python train.py \
  --root /scratch/pmerceur/data \
  --seed $seed \
  --trainer ADAPTER \
  --dataset-config-file configs/datasets/${dataset}.yaml \
  --config-file "$CONFIG_FILE" \
  --output-dir output/${experiment_name}/${dataset}/${shots}shot_${DESC}/seed${seed} \
  --backbone $backbone \
  DATASET.NUM_SHOTS $shots \
  TRAINER.ADAPTER.INIT $init \
  TRAINER.ADAPTER.CONSTRAINT $constraint \
  TRAINER.ADAPTER.NUM_TEMPLATES $templates

# ------------------------------------------------------------------
# Clean-up
# ------------------------------------------------------------------
rm "$CONFIG_FILE"
echo "=== $DESC completed ==="
