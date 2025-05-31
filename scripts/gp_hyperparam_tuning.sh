#!/bin/bash

#SBATCH --job-name=gp_architecture_tune
#SBATCH --account=def-josedolz
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=1-120
#SBATCH --output=logs/gp_architecture_tune/%x_%A_%a.out

source .venv/bin/activate

# Comprehensive GP architecture tuning across datasets and configurations
init="ZS"
constraint="none"
backbone="RN50"
templates=10
experiment_name="gp_architecture_tune"
seed=3

echo "=== GP Architecture Hyperparameter Tuning ==="
echo "Testing across 3 datasets × 2 shot settings × 20 GP configs = 120 experiments"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Calculate dataset, shot setting, and GP configuration from array task ID
# Structure: 3 datasets × 2 shots × 20 configs = 120 total
dataset_idx=$(((SLURM_ARRAY_TASK_ID - 1) / 40 + 1))
remainder=$(((SLURM_ARRAY_TASK_ID - 1) % 40))
shot_idx=$((remainder / 20 + 1))
config_idx=$((remainder % 20 + 1))

# Define datasets - diverse and computationally manageable
case $dataset_idx in
    1) dataset="caltech101" ;;      # General objects (100 classes)
    2) dataset="oxford_flowers" ;;  # Fine-grained classification (102 classes)
    3) dataset="food101" ;;         # Food domain (101 classes)
esac

# Define shot settings
case $shot_idx in
    1) shots=4 ;;   # Few-shot setting
    2) shots=16 ;;  # More data setting
esac

# Define GP configurations - exploring key hyperparameters
case $config_idx in
    1)
        # Baseline: No GP
        USE_GP=False
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="Baseline_No_GP"
        ;;
    2)
        # Conservative GP - RBF kernel, low learning rate
        USE_GP=True
        GP_LR=0.001
        GP_BETA=0.01
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="Conservative_RBF_LowLR"
        ;;
    3)
        # Moderate GP - RBF kernel, moderate learning rate
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="Moderate_RBF"
        ;;
    4)
        # Aggressive GP - RBF kernel, high learning rate
        USE_GP=True
        GP_LR=0.05
        GP_BETA=0.5
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="Aggressive_RBF"
        ;;
    5)
        # Linear kernel - moderate settings
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="linear"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="Linear_Kernel"
        ;;
    6)
        # Short lengthscale - captures local similarities
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=0.5
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="Short_Lengthscale"
        ;;
    7)
        # Long lengthscale - captures global similarities
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=2.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="Long_Lengthscale"
        ;;
    8)
        # High output scale - strong GP influence
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=2.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="High_Output_Scale"
        ;;
    9)
        # Low noise variance - more confident GP
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-6
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="Low_Noise"
        ;;
    10)
        # High noise variance - more uncertain GP
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-3
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="High_Noise"
        ;;
    11)
        # More MC samples - better approximation
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=10
        GP_USE_DIAGONAL_COV=True
        DESC="More_MC_Samples"
        ;;
    12)
        # Full covariance - more flexible posterior
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.1
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=False
        DESC="Full_Covariance"
        ;;
    13)
        # Low KL weight - less regularization
        USE_GP=True
        GP_LR=0.01
        GP_BETA=0.01
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="Low_KL_Weight"
        ;;
    14)
        # High KL weight - strong regularization
        USE_GP=True
        GP_LR=0.01
        GP_BETA=1.0
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=True
        DESC="High_KL_Weight"
        ;;
    15)
        # Optimized RBF - best performing combination
        USE_GP=True
        GP_LR=0.02
        GP_BETA=0.2
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.5
        GP_OUTPUTSCALE=1.5
        GP_NOISE_VAR=5e-5
        GP_NUM_MC_SAMPLES=5
        GP_USE_DIAGONAL_COV=True
        DESC="Optimized_RBF"
        ;;
    16)
        # Optimized Linear - best linear kernel setup
        USE_GP=True
        GP_LR=0.02
        GP_BETA=0.15
        GP_KERNEL_TYPE="linear"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=2.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=5
        GP_USE_DIAGONAL_COV=True
        DESC="Optimized_Linear"
        ;;
    17)
        # Conservative Full Cov - full covariance with conservative settings
        USE_GP=True
        GP_LR=0.005
        GP_BETA=0.05
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=1.0
        GP_OUTPUTSCALE=1.0
        GP_NOISE_VAR=1e-4
        GP_NUM_MC_SAMPLES=3
        GP_USE_DIAGONAL_COV=False
        DESC="Conservative_Full_Cov"
        ;;
    18)
        # High Variance Setup - exploring high variance regime
        USE_GP=True
        GP_LR=0.03
        GP_BETA=0.3
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=0.8
        GP_OUTPUTSCALE=2.5
        GP_NOISE_VAR=2e-4
        GP_NUM_MC_SAMPLES=7
        GP_USE_DIAGONAL_COV=True
        DESC="High_Variance"
        ;;
    19)
        # Minimal GP - very light GP influence
        USE_GP=True
        GP_LR=0.001
        GP_BETA=0.005
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=3.0
        GP_OUTPUTSCALE=0.5
        GP_NOISE_VAR=1e-3
        GP_NUM_MC_SAMPLES=1
        GP_USE_DIAGONAL_COV=True
        DESC="Minimal_GP"
        ;;
    20)
        # Ultra GP - maximum GP influence
        USE_GP=True
        GP_LR=0.1
        GP_BETA=2.0
        GP_KERNEL_TYPE="rbf"
        GP_LENGTHSCALE=0.3
        GP_OUTPUTSCALE=3.0
        GP_NOISE_VAR=1e-6
        GP_NUM_MC_SAMPLES=15
        GP_USE_DIAGONAL_COV=False
        DESC="Ultra_GP"
        ;;
esac

full_desc="${dataset}_${shots}shot_${DESC}"
echo "Testing: $full_desc"
echo "Dataset: $dataset, Shots: $shots, USE_GP: $USE_GP"
if [ "$USE_GP" = "True" ]; then
    echo "GP Config: LR=$GP_LR, β=$GP_BETA, Kernel=$GP_KERNEL_TYPE"
    echo "           Lengthscale=$GP_LENGTHSCALE, OutputScale=$GP_OUTPUTSCALE"
    echo "           Noise=$GP_NOISE_VAR, MC_Samples=$GP_NUM_MC_SAMPLES"
    echo "           Diagonal_Cov=$GP_USE_DIAGONAL_COV"
fi

# Create temporary config for this experiment
CONFIG_FILE="configs/trainers/gp_arch_tune_${SLURM_ARRAY_TASK_ID}.yaml"
cat > $CONFIG_FILE << EOF
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 256
  TEST:
    BATCH_SIZE: 500
  NUM_WORKERS: 8

INPUT:
  SIZE: (224, 224)
  INTERPOLATION: "bicubic"
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  TRANSFORMS: ["random_resized_crop", "random_flip", "normalize"]

OPTIM:
  NAME: "sgd"
  LR: 0.1
  MAX_EPOCH: 150  # Sufficient epochs for convergence
  LR_SCHEDULER: "cosine"
  WARMUP_EPOCH: 1
  WARMUP_TYPE: "constant"
  WARMUP_CONS_LR: 1e-5
  WEIGHT_DECAY: 0.0

TRAIN:
  PRINT_FREQ: 10

TRAINER:
  ADAPTER:
    USE_GP: $USE_GP
    GP_LR: $GP_LR
    GP_BETA: $GP_BETA
    GP_KERNEL_TYPE: "$GP_KERNEL_TYPE"
    GP_LENGTHSCALE: $GP_LENGTHSCALE
    GP_OUTPUTSCALE: $GP_OUTPUTSCALE
    GP_NOISE_VAR: $GP_NOISE_VAR
    GP_NUM_MC_SAMPLES: $GP_NUM_MC_SAMPLES
    GP_USE_DIAGONAL_COV: $GP_USE_DIAGONAL_COV
EOF

# Run the experiment
python train.py \
    --root /scratch/pmerceur/data \
    --seed $seed \
    --trainer ADAPTER \
    --dataset-config-file configs/datasets/${dataset}.yaml \
    --config-file $CONFIG_FILE \
    --output-dir output/${experiment_name}/${dataset}/${shots}shot_${DESC}/seed${seed} \
    --backbone $backbone \
    DATASET.NUM_SHOTS $shots \
    TRAINER.ADAPTER.INIT $init \
    TRAINER.ADAPTER.CONSTRAINT $constraint \
    TRAINER.ADAPTER.NUM_TEMPLATES $templates

# Clean up temporary config
rm $CONFIG_FILE

echo ""
echo "=== GP ARCHITECTURE EXPERIMENT COMPLETED ==="
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $dataset ($dataset_idx/3)"
echo "Shots: $shots ($shot_idx/2)" 
echo "Configuration: $DESC ($config_idx/20)"
echo "GP Enabled: $USE_GP"
echo ""
echo "Results: output/${experiment_name}/${dataset}/${shots}shot_${DESC}/seed${seed}"
echo ""
echo "=== Hyperparameter Summary ==="
if [ "$USE_GP" = "True" ]; then
    echo "GP_LR: $GP_LR"
    echo "GP_BETA: $GP_BETA"
    echo "GP_KERNEL_TYPE: $GP_KERNEL_TYPE"
    echo "GP_LENGTHSCALE: $GP_LENGTHSCALE"
    echo "GP_OUTPUTSCALE: $GP_OUTPUTSCALE"
    echo "GP_NOISE_VAR: $GP_NOISE_VAR"
    echo "GP_NUM_MC_SAMPLES: $GP_NUM_MC_SAMPLES"
    echo "GP_USE_DIAGONAL_COV: $GP_USE_DIAGONAL_COV"
fi
echo ""
echo "This systematic sweep will identify optimal GP configurations!" 