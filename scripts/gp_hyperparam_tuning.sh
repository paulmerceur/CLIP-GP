#!/bin/bash
#SBATCH --job-name=gp_optim
#SBATCH --account=rrg-josedolz
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-119              # 3 datasets × 2 shots × 20 configs
#SBATCH --output=logs/gp_optim/%x_%A_%a.out

# ---------- 1.  Static choices -----------------------------------------------
source .venv/bin/activate
export TORCH_HOME=$SCRATCH/torch-cache     # keeps models off $HOME

init="ZS"
constraint="none"
backbone="RN50"
templates=10
seed=3
root=/scratch/pmerceur/data

# ---------- 2.  Search space --------------------------------------------------
datasets=(caltech101 oxford_flowers food101)   # ⇐ extend here
shots=(4 16)                                   # number of examples per class

# Each entry:  "desc,use_gp,lr,beta,kernel,lsc,outsc,noise,mc,diag"
gp_grid=(
  "baseline,False,0.01,0.10,rbf,1.0,1.0,1e-4,3,True"
  "conservative_rbf,True,0.001,0.01,rbf,1.0,1.0,1e-4,3,True"
  "moderate_rbf,True,0.01,0.10,rbf,1.0,1.0,1e-4,3,True"
  "aggressive_rbf,True,0.05,0.50,rbf,1.0,1.0,1e-4,3,True"
  "linear_kernel,True,0.01,0.10,linear,1.0,1.0,1e-4,3,True"
  "short_lsc,True,0.01,0.10,rbf,0.5,1.0,1e-4,3,True"
  "long_lsc,True,0.01,0.10,rbf,2.0,1.0,1e-4,3,True"
  "high_outsc,True,0.01,0.10,rbf,1.0,2.0,1e-4,3,True"
  "low_noise,True,0.01,0.10,rbf,1.0,1.0,1e-6,3,True"
  "high_noise,True,0.01,0.10,rbf,1.0,1.0,1e-3,3,True"
  "more_mc,True,0.01,0.10,rbf,1.0,1.0,1e-4,10,True"
  "full_cov,True,0.01,0.10,rbf,1.0,1.0,1e-4,3,False"
  "low_beta,True,0.01,0.01,rbf,1.0,1.0,1e-4,3,True"
  "high_beta,True,0.01,1.00,rbf,1.0,1.0,1e-4,3,True"
  "opt_rbf,True,0.02,0.20,rbf,1.5,1.5,5e-5,5,True"
  "opt_linear,True,0.02,0.15,linear,1.0,2.0,1e-4,5,True"
  "conservative_full,True,0.005,0.05,rbf,1.0,1.0,1e-4,3,False"
  "high_variance,True,0.03,0.30,rbf,0.8,2.5,2e-4,7,True"
  "minimal_gp,True,0.001,0.005,rbf,3.0,0.5,1e-3,1,True"
  "ultra_gp,True,0.10,2.00,rbf,0.3,3.0,1e-6,15,False"
)

# ---------- 3.  Decode SLURM_ARRAY_TASK_ID -----------------------------------
let "d =  SLURM_ARRAY_TASK_ID / 40"          # 0-2
let "s = (SLURM_ARRAY_TASK_ID / 20) % 2"     # 0-1
let "g =  SLURM_ARRAY_TASK_ID % 20"           # 0-19

dataset=${datasets[$d]}
shots_val=${shots[$s]}
IFS=',' read desc use_gp gp_lr gp_beta kernel lsc osc noise mc diag <<< "${gp_grid[$g]}"

echo "Dataset=${dataset}  Shots=${shots_val}  Config=${desc} (ID $SLURM_ARRAY_TASK_ID)"

# ---------- 4.  Generate YAML on the fly -------------------------------------
cfg=$(mktemp /tmp/gp_cfg.XXXX.yaml)

cat > "$cfg" <<EOF
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 256
  TEST:
    BATCH_SIZE: 500
  NUM_WORKERS: 8

OPTIM:
  NAME: "sgd"
  LR: 0.1
  MAX_EPOCH: 150
  LR_SCHEDULER: "cosine"
  WARMUP_EPOCH: 1
  WARMUP_TYPE: "constant"
  WARMUP_CONS_LR: 1e-5
  WEIGHT_DECAY: 0.0

TRAINER:
  ADAPTER:
    USE_GP: ${use_gp}
    GP_LR: ${gp_lr}
    GP_BETA: ${gp_beta}
    GP_KERNEL_TYPE: "${kernel}"
    GP_LENGTHSCALE: ${lsc}
    GP_OUTPUTSCALE: ${osc}
    GP_NOISE_VAR: ${noise}
    GP_NUM_MC_SAMPLES: ${mc}
    GP_USE_DIAGONAL_COV: ${diag}
EOF

# ---------- 5.  Launch --------------------------------------------------------
python train.py \
  --root "$root" \
  --seed $seed \
  --trainer ADAPTER \
  --dataset-config-file configs/datasets/${dataset}.yaml \
  --config-file "$cfg" \
  --output-dir output/gp_grid/${dataset}/${shots_val}shot_${desc}/seed${seed} \
  --backbone $backbone \
  DATASET.NUM_SHOTS $shots_val \
  TRAINER.ADAPTER.INIT $init \
  TRAINER.ADAPTER.CONSTRAINT $constraint \
  TRAINER.ADAPTER.NUM_TEMPLATES $templates

rm "$cfg"   # clean up
