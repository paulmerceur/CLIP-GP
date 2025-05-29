#!/bin/bash

# custom config
DATA=/scratch/pmerceur/data
TRAINER=ADAPTER

SEEDS=$1
DATASET=$2      # target dataset - i.e. {imagenet, caltech101, oxford_pets, stanford_cars, oxford_flowers, food101,
                #                        fgvc_aircraft, sun397, dtd, eurosat, ucf101}
CFG=$3          # config file - SGD_lr1e-1_B256_ep300
SHOTS=$4        # number of shots (1, 2, 4, 8, 16)
INIT=$5         # Method / Linear Probe init - i.e. {RANDOM, ZS, ClipA, TipA, TipA-f-, TR, TRenh}
CONSTRAINT=$6   # apply class-adaptive constraint in Linear Probing (CLAP) - i.e. {none, l2}
BACKBONE=$7     # CLIP backbone to sue - i.e. {RN50, RN101, ViT-B/32, ViT-B/16}
NB_TEMPLATES=$8 # number of templates
EXPERIMENT_NAME=${9:-"single_test"}  # experiment name for organizing outputs

for ((seed=1; seed<=SEEDS; seed++)); do
    DIR=output/${EXPERIMENT_NAME}/${DATASET}/${CFG}_${INIT}Init_${CONSTRAINT}Constraint_${SHOTS}shots_${NB_TEMPLATES}templates/seed${seed}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${seed} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${CFG}.yaml \
        --output-dir ${DIR} \
        --backbone ${BACKBONE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.ADAPTER.INIT ${INIT} \
        TRAINER.ADAPTER.CONSTRAINT ${CONSTRAINT} \
        TRAINER.ADAPTER.NUM_TEMPLATES ${NB_TEMPLATES}
    fi
done