#!/usr/bin/env bash

set -euo pipefail

# Grid runner for OxfordPets with multiple GP config variants, 3 seeds, and shots {1,4,8,16}.
# Each variant is backed by its own config file name so your comparison script can group by config.
#
# Usage:
#   scripts/grid_oxford_pets.sh [GPU_ID] [DATA_ROOT]
#
# Defaults:
#   GPU_ID: 0
#   DATA_ROOT: /export/datasets/public

GPU_ID=${1:-0}
DATA_ROOT="/export/datasets/public"

DATASET=oxford_pets
TRAINER_CFG_DIR="configs/trainers"

# Base config to duplicate
BASE_CFG="${TRAINER_CFG_DIR}/gp.yaml"

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "Base config not found: ${BASE_CFG}" >&2
  exit 1
fi

# Helper: create a config variant by copying gp.yaml and overriding a few keys via sed
make_cfg() {
  local out_cfg="$1"; shift
  cp -f "${BASE_CFG}" "${out_cfg}"
  # Each remaining argument is KEY=VALUE to set in YAML (single line keys only)
  for kv in "$@"; do
    local key="${kv%%=*}"
    local val="${kv#*=}"
    # Replace the first occurrence of the key (anchored start, allow indentation)
    sed -i "0,/^[[:space:]]*${key}:.*/s//\ \ \ \ ${key}: ${val}/" "${out_cfg}"
    # If key didn't exist, insert it under TRAINER.ADAPTER block after a known anchor
    if ! grep -q "^[[:space:]]*${key}:" "${out_cfg}"; then
      if grep -q "^[[:space:]]*GP_FREEZE_ADAPTER_EPOCHS:" "${out_cfg}"; then
        sed -i "/^[[:space:]]*GP_FREEZE_ADAPTER_EPOCHS:.*/a\\    ${key}: ${val}" "${out_cfg}"
      elif grep -q "^[[:space:]]*GP_NORMALIZE_MEAN_INIT:" "${out_cfg}"; then
        sed -i "/^[[:space:]]*GP_NORMALIZE_MEAN_INIT:.*/a\\    ${key}: ${val}" "${out_cfg}"
      else
        # Fallback: append at end with indentation (still under ADAPTER in most cases)
        echo "    ${key}: ${val}" >> "${out_cfg}"
      fi
    fi
  done
}

echo "Preparing config variants in ${TRAINER_CFG_DIR}"

# Sweep only key knobs; keep logit_scale trainable and no adapter freeze
BETAS=(0.0001)
GP_LRS=(0.1)
L2S=(20.0)

CONFIGS=()

for beta in "${BETAS[@]}"; do
  for gp_lr in "${GP_LRS[@]}"; do
    for l2 in "${L2S[@]}"; do
      cfg_name="b${beta}_glr${gp_lr}_l2${l2}"
      out_path="${TRAINER_CFG_DIR}/${cfg_name}.yaml"
      make_cfg "${out_path}" \
        "GP_WEIGHT_TRANSFORM=softmax" \
        "GP_BETA=${beta}" \
        "GP_LR=${gp_lr}" \
        "L2_LAMBDA=${l2}" \
        "FREEZE_LOGIT_SCALE=False" \
        "GP_DETERMINISTIC_TRAIN=False"
      CONFIGS+=("${cfg_name}")
    done
  done
done

SEEDS=(1 2 3)
SHOTS=(1 4 8 16)

echo "Running configs: ${#CONFIGS[@]} variants"
echo "Seeds: ${SEEDS[*]} | Shots: ${SHOTS[*]} | GPU: ${GPU_ID} | Data: ${DATA_ROOT}"

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for shot in "${SHOTS[@]}"; do
      OUT_DIR="output/sweep2/${DATASET}/gp_${shot}shots_${cfg}/seed${seed}"
      if [[ -d "${OUT_DIR}" ]]; then
        echo "Skip existing: ${OUT_DIR}"
        continue
      fi
      echo "Run: cfg=${cfg} seed=${seed} shots=${shot}"
      CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        --root "${DATA_ROOT}" \
        --seed "${seed}" \
        --trainer ADAPTER \
        --dataset-config-file "configs/datasets/${DATASET}.yaml" \
        --config-file "${TRAINER_CFG_DIR}/${cfg}.yaml" \
        --output-dir "${OUT_DIR}" \
        DATASET.NUM_SHOTS "${shot}"
    done
  done
done

echo "All jobs submitted. Results under output/<cfg>/<dataset>/..."


# Remove all configs
for cfg in "${CONFIGS[@]}"; do
  rm -f "${TRAINER_CFG_DIR}/${cfg}.yaml"
done