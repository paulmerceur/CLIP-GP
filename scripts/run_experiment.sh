#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <experiment_name>"
  exit 1
fi
EXPERIMENT_NAME=$1

# make sure the log folder exists
mkdir -p logs/${EXPERIMENT_NAME}

sbatch \
  --job-name=clip_gp_${EXPERIMENT_NAME} \
  --output=logs/${EXPERIMENT_NAME}/%x_%A_%a.out \
  scripts/baseline.sh "${EXPERIMENT_NAME}"