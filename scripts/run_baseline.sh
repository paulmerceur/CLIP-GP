#!/bin/bash

DATA="/export/datasets/public"

# Run baseline experiments
# Usage: ./run_baseline.sh <experiment_name> <dataset> [GPU_ID]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> <dataset> [GPU_ID]"
    echo "Example: $0 baseline_test caltech101 0"
    exit 1
fi

EXPERIMENT_NAME=$1
DATASET=$2
GPU_ID=${3:-0}  # Default to 0 if not provided

# Experiment parameters
SEEDS=(1 2 3)
SHOTS=(1 4 8 16)
CONFIG="baseline"

echo "Running baseline experiments..."
echo "Experiment: $EXPERIMENT_NAME"
echo "Dataset: $DATASET"
echo "Shots: ${SHOTS[*]}"
echo ""

for SEED in "${SEEDS[@]}"; do
    for SHOT in "${SHOTS[@]}"; do
        DIR=output/${EXPERIMENT_NAME}/${DATASET}/${CONFIG}_${SHOT}shots/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
            continue
        fi

        echo "Running: seed=$SEED dataset=$DATASET shots=$SHOT"

        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
            --root $DATA \
            --seed $SEED \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${CONFIG}.yaml \
            --output-dir $DIR \
            DATASET.NUM_SHOTS $SHOT
    done
done

echo "Baseline experiments completed!"
echo "Results saved in: output/${EXPERIMENT_NAME}/"
