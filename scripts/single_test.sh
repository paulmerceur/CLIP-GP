#!/bin/bash
#SBATCH --job-name=clap_baselines
#SBATCH --account=def-josedolz
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-0%1
#SBATCH --output=logs/single_test/%x_%j.out
# ──────────────────────────
# 0. Parse arguments and environment set-up  
# ──────────────────────────

source "${PWD}/.venv/bin/activate"

export TORCH_HOME="${PWD}/.cache"
export PYTHONHASHSEED=0

dataset="stanford_cars"
optim="SGD_lr1e-1_B256_ep300"
shot=1
init="ZS"
constraint="none"
backbone="RN50"
nb_templates=1

bash scripts/adapt.sh 0 "$dataset" "$optim" "$shot" "$init" "$constraint" "$backbone" "$nb_templates"
