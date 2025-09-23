#!/bin/bash

# Quick launcher for fine-tuning experiments.
# Examples:
#   ./scripts/run_finetune.sh                # default experiment YAML
#   ./scripts/run_finetune.sh configs/trainers/finetune_experiment.yaml "ft_sweep" --devices "0,1" --jobs-per-gpu 1

EXP_FILE=${1:-configs/trainers/finetune_experiment.yaml}
shift || true

EXP_NAME=""
if [ $# -ge 1 ]; then
  case "$1" in
    --*) ;;
    *) EXP_NAME="$1"; shift ;;
  esac
fi

if [ -n "$EXP_NAME" ]; then
  python -m utils.hparam_search --config-file "${EXP_FILE}" --experiment-name "${EXP_NAME}" "$@"
else
  python -m utils.hparam_search --config-file "${EXP_FILE}" "$@"
fi


