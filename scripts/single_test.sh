#!/bin/bash
#SBATCH --job-name=clap_baselines
#SBATCH --account=def-josedolz
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/test_gp2/%x_%j.out
# ──────────────────────────
# 0. Parse arguments and environment set-up  
# ──────────────────────────

source .venv/bin/activate

dataset="caltech101"
optim="SGD_lr1e-1_B256_ep300_GP"
shots=2
init="ZS"
constraint="none"
backbone="RN50"
nb_templates="10"
experiment_name="test_gp2"

bash scripts/adapt.sh 3 "$dataset" "$optim" "$shots" "$init" "$constraint" "$backbone" "$nb_templates" "$experiment_name"
