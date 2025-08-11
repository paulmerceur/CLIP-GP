#!/bin/bash

DATA="/export/datasets/public"

# Run GP experiments  
# Usage: ./run_gp.sh <experiment_name> <dataset> [L2_LAMBDA] [GPU_ID]

if [ $# -lt 1 ]; then
echo "Usage: $0 <experiment_name> <dataset> [L2_LAMBDA] [GPU_ID]"
echo "Example: $0 gp_test caltech101 0.1 0"
    exit 1
fi

EXPERIMENT_NAME=$1
DATASET=$2
L2_LAMBDA=${3:-0.1}  # Default to 0.1 if not provided
GPU_ID=${4:-0}  # Default to 0 if not provided

# Experiment parameters
SEEDS=(1 2 3)
SHOTS=(1 4 8 16)
CONFIG="gp"

echo "Running GP experiments..."
echo "Experiment: $EXPERIMENT_NAME"
echo "L2_LAMBDA: $L2_LAMBDA"
echo "Datasets: ${DATASET[*]}"
echo "Shots: ${SHOTS[*]}"
echo ""

for SEED in "${SEEDS[@]}"; do
    for SHOT in "${SHOTS[@]}"; do
        DIR=output/${EXPERIMENT_NAME}/${DATASET}/${CONFIG}_${SHOT}shots_l2${L2_LAMBDA}/seed${SEED}
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
            DATASET.NUM_SHOTS $SHOT \
            TRAINER.ADAPTER.L2_LAMBDA $L2_LAMBDA
    done
done

echo "GP experiments completed!"
echo "Results saved in: output/${EXPERIMENT_NAME}/"
