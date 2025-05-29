#!/bin/bash

#SBATCH --job-name=clap_baselines
#SBATCH --account=def-josedolz
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/test/%x_%A_%a.out
#SBATCH --array=1-50

source .venv/bin/activate

# RN50 Experiments for all datasets - CLAP - 1 shots
bash scripts/adapt.sh 0 caltech101 SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_pets SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 stanford_cars SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_flowers SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 food101 SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 fgvc_aircraft SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 sun397 SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 dtd SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 eurosat SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 ucf101 SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 1 test
# RN50 Experiments for all datasets - CLAP - 2 shots
bash scripts/adapt.sh 0 caltech101 SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_pets SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 stanford_cars SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_flowers SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 food101 SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 fgvc_aircraft SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 sun397 SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 dtd SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 eurosat SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 ucf101 SGD_lr1e-1_B256_ep300 2 ZS l2 RN50 1 test
# RN50 Experiments for all datasets - CLAP - 4 shots
bash scripts/adapt.sh 0 caltech101 SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_pets SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 stanford_cars SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_flowers SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 food101 SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 fgvc_aircraft SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 sun397 SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 dtd SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 eurosat SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 ucf101 SGD_lr1e-1_B256_ep300 4 ZS l2 RN50 1 test
# RN50 Experiments for all datasets - CLAP - 8 shots
bash scripts/adapt.sh 0 caltech101 SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_pets SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 stanford_cars SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_flowers SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 food101 SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 fgvc_aircraft SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 sun397 SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 dtd SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 eurosat SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 ucf101 SGD_lr1e-1_B256_ep300 8 ZS l2 RN50 1 test
# RN50 Experiments for all datasets - CLAP - 16 shots
bash scripts/adapt.sh 0 caltech101 SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_pets SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 stanford_cars SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 oxford_flowers SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 food101 SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 fgvc_aircraft SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 sun397 SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 dtd SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 eurosat SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test
bash scripts/adapt.sh 0 ucf101 SGD_lr1e-1_B256_ep300 16 ZS l2 RN50 1 test