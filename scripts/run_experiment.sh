#!/bin/bash

# Thin wrapper around the Python experiment runner
# Usage:
#   ./scripts/run_experiment.sh <experiment_yaml> [<experiment_name>] [--devices "0,1" --jobs-per-gpu 1 --verbose]

if [ $# -lt 1 ]; then
  echo "Usage: $0 <experiment_yaml> [<experiment_name>] [--devices \"0,1\" --jobs-per-gpu 1 --verbose]"
  exit 1
fi

EXP_FILE=$1
shift

# Optional experiment name if next arg doesn't look like a flag
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


