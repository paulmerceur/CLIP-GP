# CLIP‑GP

> Code for the 2025 Master’s research project “Calibrating Vision–Language Models in Few–Shot Settings” (Software Engineering and IT, ÉTS Montréal).
> Author: Paul Merceur

CLIP‑GP explores few‑shot adaptation of CLIP using multiple text templates per class. It provides a small set of lightweight adapters that keep CLIP encoders frozen and train only small heads:

- Adapter (default): visual projection with optional learned template weighting
- GP Template Weighter: Gaussian‑Process weights over templates (RBF/Matern/Linear kernels), used alone or to initialize other adapters
- CLIP‑Adapter: 2‑layer MLP on image features (with optional GP‑derived prototypes)
- TaskRes: learn a residual on top of frozen text features (optionally initialized from GP prototypes)
- CoOp: learnable prompt tokens (context optimization)
- Tip‑Adapter / Tip‑Adapter‑F: cache‑based adapter with optional trainable linear head

All trainers report accuracy and calibration metrics (ECE, AECE), and write a compact `metrics.json` for downstream aggregation and plotting.

---

## Installation

1) Create and activate a Python environment (Python ≥3.8), then install dependencies:

```bash
pip install -r requirements.txt
```

2) Prepare datasets as described in `DATASETS.md`. Set your dataset root via `--root` or edit the scripts’ `DATA`/`DATA_ROOT` variables.

---

## Quick Start (recommended): run_experiment.sh

The primary entrypoint is the wrapper, which schedules single runs or grids and writes results in a consistent layout:

```bash
./scripts/run_experiment.sh <experiment_yaml> [<experiment_name>] [--devices "0,1" --jobs-per-gpu 1 --verbose]
# example
./scripts/run_experiment.sh configs/trainers/gp_small.yaml my_experiment --devices "0,1" --jobs-per-gpu 1
```

Notes:
- This script calls the Python runner (`python -m utils.hparam_search`) with your YAML.
- Outputs are written under `output/<experiment>/<dataset>/<config_signature>/seed<seed>/`.
- Combine `--devices` and `--jobs-per-gpu` to distribute runs across GPUs.

See the “Experiment sweeps (grids) and scheduling” section for a minimal YAML example.

---

## Single run (advanced, one‑off)

Use the native CLI. You can mix YAML config files and command‑line overrides.

```bash
# Baseline Adapter (visual projection + template prototypes)
python train.py \
  --root /path/to/datasets \
  --dataset Caltech101 \
  --shots 4 \
  --backbone RN50 \
  --config-file configs/trainers/baseline.yaml \
  --output-dir output/demo/baseline

# Adapter with GP template weighting
python train.py \
  --root /path/to/datasets \
  --dataset Caltech101 \
  --shots 4 \
  --backbone RN50 \
  --config-file configs/trainers/gp.yaml \
  --use-gp --num-templates 8 --gp-lr 1e-3 --gp-beta 1e-3 \
  --output-dir output/demo/gp

# CLIP‑Adapter
python train.py \
  --root /path/to/datasets \
  --dataset Caltech101 \
  --shots 4 \
  --backbone RN50 \
  --config-file configs/trainers/clip_adapter.yaml \
  --output-dir output/demo/clip_adapter

# TaskRes (optionally preceded by GP pre‑training of prototypes)
python train.py \
  --root /path/to/datasets \
  --dataset Caltech101 \
  --shots 4 \
  --backbone RN50 \
  --config-file configs/trainers/taskres.yaml \
  --output-dir output/demo/taskres
```

Notes:
- Prefer YAMLs in `configs/trainers/` to set the trainer and defaults. You can still override any field via CLI.
- Important adapter options (subset): `--num-templates`, `--use-gp`, `--gp-lr`, `--gp-beta`, `--gp-num-mc-samples-train`, `--gp-num-mc-samples-eval`, `--l2-lambda`, `--freeze-visual-proj`.

---

## Experiment sweeps (grids) and scheduling

Use the experiment runner to expand grids and schedule runs evenly across GPUs. Wrapper:

```bash
# Wrapper script (recommended)
./scripts/run_experiment.sh configs/trainers/gp_small.yaml my_experiment --devices "0,1" --jobs-per-gpu 1

# Or call the runner directly
python -m utils.hparam_search \
  --config-file configs/trainers/gp_small.yaml \
  --devices "0,1" \
  --jobs-per-gpu 1 \
  --experiment-name my_experiment
```

Minimal YAML structure (example):

```yaml
name: gp_small
datasets: [caltech101, oxford_pets]
seeds: [1, 2, 3]
shots: [1, 2, 4, 8]
output_root: output
template: "{experiment}/{dataset}/{sig}/seed{seed}"
grid:
  TRAINER.ADAPTER.USE_GP: [True]
  TRAINER.ADAPTER.NUM_TEMPLATES: [8, 32]
  OPTIM.LR: [0.001]
  MODEL.BACKBONE.NAME: ["RN50"]
```

The runner writes each trial under:

```
output/<experiment>/<dataset>/<config_signature>/seed<seed>/
```

Each run creates `metrics.json`, `log.txt`, and stores the resolved `config.json`.

---

## Aggregation and plotting

After runs finish, aggregate and plot:

```bash
python scripts/aggregate_results.py <experiment_name> \
  [--grouped] [--show-zero-shot]
```

It reads:

```
output/<experiment>/<dataset>/<config_signature>/seed*/metrics.json
```

and prints per‑dataset summaries, averages across datasets, plus saves:

- Plots: `output/<experiment>/_plots/perf_per_shots/` and `.../_plots/acc_vs_ece/`
- Tables (per dataset and averaged): `output/<experiment>/_tables/`

Flags:
- `--grouped`: group multiple configs into single lines using `GROUP_SUBSTRINGS` in `scripts/aggregate_results.py`
- `--show-zero-shot`: plot zero‑shot performance as stars at shot=0

`metrics.json` schema (minimal):

```json
{
  "dataset": "Caltech101",
  "shots": 1,
  "seed": 1,
  "method": "baseline|gp|clip-adapter|coop|cocoop|tipa|tipaf",
  "backbone": "RN50",
  "zero_shot": {"top1_acc": 0.0, "ece": 0.0, "aece": 0.0, "calibration": {...}},
  "metrics": {"top1_acc": 0.0, "ece": 0.0, "aece": 0.0},
  "config": {...},
  "output_dir": "output/...",
  "train_time_s": 0.0
}
```

---

## Trainers and key options (quick reference)

- Adapter (default): visual projection with L2 regularization; template prototypes from multiple prompts
  - Template weighting (non‑GP): `--train-template-weights` or `--use-linear-template-weighting`
  - Weight init: `--template-init-method {uniform,val_weighted,top3,minmax}`
  - Shared weights across classes: `--shared-template-weights`
  - Regularization: `--l2-lambda`, `--freeze-visual-proj`

- GP Template Weighter: Gaussian‑Process over per‑template logits
  - Enable: `--use-gp`
  - Kernel: `--gp-kernel-type {rbf,linear}` (Matern available via config)
  - Hyper‑params: `--gp-lr`, `--gp-beta`, `--gp-num-mc-samples-train`, `--gp-num-mc-samples-eval`, `--gp-pca-dim`

- CLIP‑Adapter: 2‑layer MLP on image features blending adapted/original features
  - Key flags: `ADAPTER.CLIP_ADAPTER_REDUCTION`, `ADAPTER.CLIP_ADAPTER_RATIO`, `ADAPTER.CLIP_ADAPTER_LR`, `ADAPTER.CLIP_ADAPTER_EPOCHS`

- TaskRes: learn residuals on top of base text features
  - Scale: `ADAPTER.TASKRES_RESIDUAL_SCALE`
  - Optional GP to initialize base prototypes before residual learning

- CoOp: prompt learning with learnable context tokens
  - Flags: `ADAPTER.N_CTX`, `ADAPTER.CTX_INIT`, `ADAPTER.CSC`

- Tip‑Adapter / Tip‑Adapter‑F: cache‑based (with optional trainable linear head)
  - Flags: `ADAPTER.TIP_ADAPTER_TRAINABLE`, `ADAPTER.TIP_ADAPTER_LR`, `ADAPTER.TIP_ADAPTER_EPOCHS`

Most of these are also exposed via YAML in `configs/trainers/` and can be overridden on the CLI using the `OPTS` style, e.g.:

```bash
... TRAINER.ADAPTER.L2_LAMBDA 0.1 TRAINER.ADAPTER.NUM_TEMPLATES 8
```

---

## What you get

- Frozen CLIP encoders with template‑based text prototypes
- Optional GP weighting over templates with KL regularization
- Visual projection with L2 or alternative adapters (CLIP‑Adapter, TaskRes, CoOp, Tip‑Adapter)
- Robust evaluation: top‑1 accuracy, macro‑F1 (if sklearn available), ECE, AECE
- Clean experiment runner for grids and reproducible outputs

---

## Credits

This repository is based on the CLAP project (CVPR’24) “A Closer Look at the Few‑Shot Adaptation of Large Vision‑Language Models.” See the original code and paper materials here: [CLAP on GitHub](https://github.com/jusiro/CLAP).

This version removes heavyweight dependencies and provides a minimal, PyTorch‑native training and data pipeline with simple configs and scripts.
