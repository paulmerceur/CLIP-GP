#!/bin/bash
#SBATCH --job-name=general_optim
#SBATCH --account=rrg-josedolz
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-137              # 3 datasets × 2 shots × 23 configs
#SBATCH --output=logs/general_optim/%x_%A_%a.out

# ---------- 1.  Static choices -----------------------------------------------
source .venv/bin/activate
export TORCH_HOME=$SCRATCH/torch-cache     # keeps models off $HOME

init="ZS"
constraint="none"
backbone="RN50"
templates=10
seeds=3
root=/scratch/pmerceur/data

# Fixed GP configuration (opt_linear from previous tuning)
use_gp=True
gp_kernel="linear"
gp_lr=0.02
gp_beta=0.15
gp_lengthscale=1.0
gp_outputscale=2.0
gp_noise=1e-4
gp_mc_samples=5
gp_diag=True

# ---------- 2.  Search space --------------------------------------------------
datasets=(caltech101 oxford_flowers food101)   # 3 datasets
shots=(4 16)                                   # 2 shot settings

# General hyperparameter grid - 24 configurations
# Format: "desc,optimizer,lr,weight_decay,max_epoch,lr_scheduler,batch_size"
general_grid=(
  # SGD configurations
  "sgd_lr001_wd0_300ep,sgd,0.01,0.0,300,cosine,128"
  "sgd_lr005_wd0_300ep,sgd,0.05,0.0,300,cosine,128"
  "sgd_lr01_wd0_300ep,sgd,0.1,0.0,300,cosine,128"
  "sgd_lr02_wd0_300ep,sgd,0.2,0.0,300,cosine,128"
  "sgd_lr01_wd1e4_300ep,sgd,0.1,1e-4,300,cosine,128"
  "sgd_lr01_wd1e3_300ep,sgd,0.1,1e-3,300,cosine,128"
  "sgd_lr01_wd0_200ep,sgd,0.1,0.0,200,cosine,128"
  "sgd_lr01_wd0_400ep,sgd,0.1,0.0,400,cosine,128"
  "sgd_lr01_wd0_step,sgd,0.1,0.0,300,step,128"
  "sgd_lr01_bs64,sgd,0.1,0.0,300,cosine,64"
  "sgd_lr01_bs256,sgd,0.1,0.0,300,cosine,256"
  "sgd_lr005_wd1e4_300ep,sgd,0.05,1e-4,300,cosine,128"
  # AdamW configurations
  "adamw_lr001_wd0_300ep,adamw,0.001,0.0,300,cosine,128"
  "adamw_lr005_wd0_300ep,adamw,0.005,0.0,300,cosine,128"
  "adamw_lr01_wd0_300ep,adamw,0.01,0.0,300,cosine,128"
  "adamw_lr001_wd1e4_300ep,adamw,0.001,1e-4,300,cosine,128"
  "adamw_lr001_wd1e3_300ep,adamw,0.001,1e-3,300,cosine,128"
  "adamw_lr001_wd1e2_300ep,adamw,0.001,1e-2,300,cosine,128"
  "adamw_lr001_wd0_200ep,adamw,0.001,0.0,200,cosine,128"
  "adamw_lr001_wd0_400ep,adamw,0.001,0.0,400,cosine,128"
  "adamw_lr001_bs64,adamw,0.001,0.0,300,cosine,64"
  "adamw_lr001_bs256,adamw,0.001,0.0,300,cosine,256"
  "adamw_lr005_wd1e4_300ep,adamw,0.005,1e-4,300,cosine,128"
)

# ---------- 3.  Decode SLURM_ARRAY_TASK_ID -----------------------------------
let "d =  SLURM_ARRAY_TASK_ID / 46"          # 0-2 (datasets)
let "s = (SLURM_ARRAY_TASK_ID / 23) % 2"     # 0-1 (shots)
let "g =  SLURM_ARRAY_TASK_ID % 23"          # 0-23 (general configs)

dataset=${datasets[$d]}
shots_val=${shots[$s]}
IFS=',' read desc optimizer lr weight_decay max_epoch lr_scheduler batch_size <<< "${general_grid[$g]}"

echo "Dataset=${dataset}  Shots=${shots_val}  Config=${desc} (ID $SLURM_ARRAY_TASK_ID)"

# ---------- 4.  Generate YAML on the fly -------------------------------------
cfg=$(mktemp /tmp/general_cfg.XXXX.yaml)

# Set LR scheduler specific parameters
if [ "$lr_scheduler" = "step" ]; then
    lr_scheduler_params="
  LR_SCHEDULER: \"step\"
  STEPSIZE: [100, 200]
  GAMMA: 0.1"
else
    lr_scheduler_params="
  LR_SCHEDULER: \"cosine\""
fi

cat > "$cfg" <<EOF
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: ${batch_size}
  TEST:
    BATCH_SIZE: ${batch_size}
  NUM_WORKERS: 8

INPUT:
  SIZE: (224, 224)
  INTERPOLATION: "bicubic"
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  TRANSFORMS: ["random_resized_crop", "random_flip", "normalize"]

OPTIM:
  NAME: "${optimizer}"
  LR: ${lr}
  MAX_EPOCH: ${max_epoch}${lr_scheduler_params}
  WARMUP_EPOCH: 1
  WARMUP_TYPE: "constant"
  WARMUP_CONS_LR: 1e-5
  WEIGHT_DECAY: ${weight_decay}

TRAIN:
  PRINT_FREQ: 5

TRAINER:
  ADAPTER:
    USE_GP: ${use_gp}
    GP_KERNEL_TYPE: "${gp_kernel}"
    GP_LR: ${gp_lr}
    GP_BETA: ${gp_beta}
    GP_LENGTHSCALE: ${gp_lengthscale}
    GP_OUTPUTSCALE: ${gp_outputscale}
    GP_NOISE: ${gp_noise}
    GP_NUM_MC_SAMPLES: ${gp_mc_samples}
    GP_USE_DIAGONAL_COV: ${gp_diag}
EOF

# ---------- 5.  Launch --------------------------------------------------------
for seed in ${seeds[@]}; do
python train.py \
    --root "$root" \
    --seed $seed \
    --trainer ADAPTER \
    --dataset-config-file configs/datasets/${dataset}.yaml \
    --config-file "$cfg" \
    --output-dir output/general_grid/${dataset}/${shots_val}shot_${desc}/seed${seed} \
    --backbone $backbone \
    DATASET.NUM_SHOTS $shots_val \
    TRAINER.ADAPTER.INIT $init \
    TRAINER.ADAPTER.CONSTRAINT $constraint \
    TRAINER.ADAPTER.NUM_TEMPLATES $templates
done

rm "$cfg"   # clean up 