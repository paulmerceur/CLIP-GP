#!/bin/bash

# custom config
#DATA=/scratch/pmerceur/data
DATA=/export/datasets/public

TRAINER=ADAPTER

SEEDS=$1
DATASET=$2      # target dataset - i.e. {imagenet, caltech101, oxford_pets, stanford_cars, oxford_flowers, food101,
                #                        fgvc_aircraft, sun397, dtd, eurosat, ucf101}
CFG=$3          # config file - SGD_lr1e-1_B256_ep300
SHOTS=$4        # number of shots (1, 2, 4, 8, 16)
BACKBONE=$5     # CLIP backbone to sue - i.e. {RN50, RN101, ViT-B/32, ViT-B/16}
GP_LR=$6        # GP learning rate for GP parameters
GP_BETA=$7      # GP beta (KL weight)
GP_W_REG_COEF=$8 # visual projection regularization
EXPERIMENT_NAME=${9:-"single_test"}  # experiment name for organizing outputs
GPU_ID=${10:-0}

for ((seed=1; seed<=SEEDS; seed++)); do
    DIR=output/${EXPERIMENT_NAME}/${DATASET}/${CFG}_${SHOTS}shots_LR${GP_LR}_B${GP_BETA}_WR${GP_W_REG_COEF}/seed${seed}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        --root ${DATA} \
        --seed ${seed} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${CFG}.yaml \
        --output-dir ${DIR} \
        --backbone ${BACKBONE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.ADAPTER.GP_LR ${GP_LR} \
        TRAINER.ADAPTER.GP_BETA ${GP_BETA} \
        TRAINER.ADAPTER.GP_W_REG_COEF ${GP_W_REG_COEF}
    fi
done