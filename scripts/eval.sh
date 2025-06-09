#!/bin/bash

# custom config
TRAINER=ADAPTER
DATA=/scratch/pmerceur/data

SEEDS=$1
DATASET_SOURCE=$2       # source dataset
DATASET_TARGET=$3       # target dataset
CFG=$4                  # config file
SHOTS=$5                # number of shots
INIT=$6                 # adapter init method
CONSTRAINT=$7           # constraint type
BACKBONE=$8             # CLIP backbone
NB_TEMPLATES=$9         # number of templates
SOURCE_EXP_NAME=${10}   # experiment name for loading the model
TARGET_EXP_NAME=${11}   # experiment name for saving the results

for ((seed=1; seed<=SEEDS; seed++)); do
    MODELDIR=output/${SOURCE_EXP_NAME}/${DATASET_SOURCE}/${CFG}_${INIT}Init_${CONSTRAINT}Constraint_${SHOTS}shots_${NB_TEMPLATES}templates/seed${seed}
    OUTDIR=output/${TARGET_EXP_NAME}/${DATASET_SOURCE}_to_${DATASET_TARGET}/${CFG}_${INIT}Init_${CONSTRAINT}Constraint_${SHOTS}shots_${NB_TEMPLATES}templates/seed${seed}
    
    if [ -d "$OUTDIR" ]; then
        echo "Oops! The results exist at ${OUTDIR} (so skip this job)"
    else
        # Check if the model directory exists before running
        if [ ! -d "$MODELDIR" ]; then
            echo "Warning: Model directory not found at ${MODELDIR}. Skipping this job."
            continue
        fi

        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${seed} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET_TARGET}.yaml \
        --config-file configs/trainers/${CFG}.yaml \
        --output-dir ${OUTDIR} \
        --model-dir ${MODELDIR} \
        --eval-only \
        --load-epoch 300 \
        --backbone ${BACKBONE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.ADAPTER.INIT ${INIT} \
        TRAINER.ADAPTER.CONSTRAINT ${CONSTRAINT} \
        TRAINER.ADAPTER.NUM_TEMPLATES ${NB_TEMPLATES}
    fi
done