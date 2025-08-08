# CLIP‑GP

> Gaussian‑Process Weighted Template Adaptation for CLIP  
> Paul Merceur ⋅ ÉTS Montréal

CLIP‑GP explores few‑shot adaptation of CLIP using multiple text templates per class. It provides two comparable adaptation methods:

- Baseline: average of template prototypes with a visual projection (L2‑regularized)
- GP Method: Gaussian‑Process weighting over templates with an RBF or linear kernel and KL regularization, also with a visual projection

Both methods keep the CLIP encoders frozen and train only small heads for fair, stable adaptation.

---

## Installation

1) Create and activate a Python environment (Python ≥3.8), then install dependencies:

```bash
pip install -r requirements.txt
```

2) Prepare datasets as described in `configs/datasets/*.yaml` (and `DATASETS.md` if present). Set your dataset root via `--root` or edit the scripts’ `DATA`/`DATA_ROOT` variables.

---

## Quick Start


### Baseline
```bash
./scripts/run_baseline.sh <experiment_name> <dataset_key> [L2_LAMBDA] [GPU_ID]
# example
./scripts/run_baseline.sh test_v1 caltech101 100.0 0
```

### GP Method
```bash
./scripts/run_gp.sh <experiment_name> <dataset_key> [L2_LAMBDA] [GPU_ID]
# example
./scripts/run_gp.sh test_v1 caltech101 100.0 0
```

### Battery of tests (multiple datasets)
```bash
./scripts/run_big_tests.sh <experiment_name> [GPU_ID]
```

---

## Manual Usage

Use the native CLI (no external framework). You can mix YAML configs and flags.

```bash
# Baseline
python train.py \
  --root /path/to/datasets \
  --dataset Caltech101 \
  --shots 4 \
  --backbone RN50 \
  --trainer ADAPTER \
  --config-file configs/trainers/baseline.yaml \
  --output-dir output/demo/baseline

# GP method
python train.py \
  --root /path/to/datasets \
  --dataset Caltech101 \
  --shots 4 \
  --backbone RN50 \
  --trainer ADAPTER \
  --config-file configs/trainers/gp.yaml \
  --use-gp --num-templates 7 --gp-lr 0.1 --gp-beta 0.001 \
  --output-dir output/demo/gp
```

Key CLI flags (subset): `--dataset`, `--shots`, `--backbone`, `--use-gp`, `--num-templates`, `--gp-lr`, `--gp-beta`, `--output-dir`. You can also override YAML fields via `OPTS` style, e.g., `TRAINER.ADAPTER.L2_LAMBDA 0.1`.

---

## What you get

- Frozen CLIP encoders with template‑based text prototypes
- Optional GP weighting over templates with KL regularization
- Visual projection with L2 regularization
- Robust evaluation: top‑1 accuracy, macro‑F1, and ECE

---

## Credits

This repository is based on the CLAP project (CVPR’24) “A Closer Look at the Few‑Shot Adaptation of Large Vision‑Language Models.” See the original code and paper materials here: [CLAP on GitHub](https://github.com/jusiro/CLAP).

This cleaned‑up version removes the heavyweight Dassl dependency and provides a minimal, PyTorch‑native training and data pipeline.
