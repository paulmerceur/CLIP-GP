# CLIP‑GP

> **Gaussian‑Process Weighted Template Adaptation for CLIP**  

> Paul Merceur ⋅ ÉTS Montréal

CLIP‑GP implements a Gaussian Process approach for few-shot adaptation of CLIP models.
This cleaned-up version provides two methods for comparison:

- **Baseline**: Multi-template adaptation with visual projection and L2 regularization
- **GP Method**: Gaussian Process weighted template adaptation

Both methods use visual projection for fair comparison.

---

## Quick Start

### Run Baseline Method
```bash
./scripts/run_baseline.sh baseline_exp 0.1
```

### Run GP Method  
```bash
./scripts/run_gp.sh gp_exp 0.1
```

### Run Both for Comparison
```bash
./scripts/run_comparison.sh comparison_exp 0.1
```

The third parameter (0.1) is the L2 regularization coefficient for visual projection.

---

## Installation

1. **Dassl + PyTorch** – follow the [Dassl.pytorch installation guide](https://github.com/KaiyangZhou/Dassl.pytorch#installation).
2. Activate the `dassl` conda env and run:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets listed in `DATASETS.md`.

---

## Configuration

### Baseline (`configs/trainers/baseline.yaml`)
- Uses 7 templates per class
- Visual projection with L2 regularization  
- Standard CLIP adaptation

### GP Method (`configs/trainers/gp.yaml`)
- Uses 7 templates per class
- Gaussian Process weighting of templates
- RBF kernel with auto-initialized length scale
- Visual projection with L2 regularization
- Separate learning rate for GP parameters (0.1)
- KL weight (β = 0.001)

---

## Manual Usage

For custom experiments, you can run individual configurations:

```bash
python train.py \
    --root /path/to/datasets \
    --seed 1 \
    --trainer ADAPTER \
    --dataset-config-file configs/datasets/caltech101.yaml \
    --config-file configs/trainers/baseline.yaml \
    --output-dir output/test/caltech101/baseline/seed1 \
    MODEL.BACKBONE.NAME RN50 \
    DATASET.NUM_SHOTS 4 \
    TRAINER.ADAPTER.L2_LAMBDA 0.1
```

Replace `baseline.yaml` with `gp.yaml` to run the GP method.

---

## Key Parameters

- `L2_LAMBDA`: Visual projection regularization (tunable via scripts)
- `GP_LR`: Learning rate for GP parameters (default: 0.1)
- `GP_BETA`: KL divergence weight (default: 0.001)
- `GP_NUM_MC_SAMPLES`: Monte Carlo samples for evaluation (default: 10)
- `NUM_TEMPLATES`: Number of templates per class (default: 7)

---

## Analysis

After running experiments, analyze results with:

```bash
python analyze_experiment.py <experiment_name>
```

This generates:
- `<experiment_name>_summary.csv` - aggregated metrics
- `plots/<dataset>_accuracy.png` - per-dataset visualizations

---

## Project Structure

```
├── trainers/
│   ├── adapters.py              # Unified baseline + GP implementation
│   └── gp_template_weigher.py   # GP core logic
├── configs/trainers/
│   ├── baseline.yaml            # Baseline configuration
│   └── gp.yaml                  # GP configuration  
├── scripts/
│   ├── run_baseline.sh          # Run baseline experiments
│   ├── run_gp.sh                # Run GP experiments
│   └── run_comparison.sh        # Run both methods
└── analyze_experiment.py        # Results analysis
```
