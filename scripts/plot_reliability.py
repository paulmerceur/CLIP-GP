#!/usr/bin/env python3
"""Plot a reliability diagram (confidence vs accuracy) for a saved GP-CLIP model.

Example usage
-------------
$ python scripts/plot_reliability.py \
    --run-dir output/gp_test6/fgvc_aircraft/GP_rbf_16shots_LR0.1_B0.01_WR0.0001/seed1 \
    --dataset-cfg configs/datasets/fgvc_aircraft.yaml \
    --trainer-cfg configs/trainers/GP_rbf.yaml \
    --root /scratch/pmerceur/data \
    --bins 15

The script rebuilds the trainer, loads the checkpoint found in
``<run-dir>/adapter/model-best.pth.tar`` (or the last-epoch file if the
best model is missing) and runs a single forward pass over the *test* set
with the Monte-Carlo aggregation logic used during validation.

It then plots a reliability diagram and prints the Expected Calibration
Error (ECE).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Local imports (training framework)
# -----------------------------------------------------------------------------

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))  # project root

from dassl.config import get_cfg_default  # type: ignore
from dassl.engine import build_trainer  # type: ignore
from dassl.utils import set_random_seed  # type: ignore

import trainers.adapters  # noqa: F401 (ensures the trainer is registered)

# -----------------------------------------------------------------------------
# Fix for PyTorch >=2.6 safe deserialisation: allow LR schedulers found in our
# checkpoints.
# -----------------------------------------------------------------------------
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR  # noqa: E402
from torch.optim import SGD, Adam  # noqa: E402

import torch.serialization  # noqa: E402

torch.serialization.add_safe_globals([CosineAnnealingLR, StepLR, SGD, Adam])

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _find_checkpoint(run_dir: pathlib.Path) -> Tuple[pathlib.Path, Optional[int]]:
    """Return the path of the checkpoint to load inside *run_dir*.

    Priority order:
    1. model-best.pth.tar
    2. Highest-epoch model.pth.tar-XXX
    """
    adapter_dir = run_dir / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"'{adapter_dir}' not found (run directory wrong?)")

    # Prefer explicit "model-best.pth.tar" if present
    best = adapter_dir / "model-best.pth.tar"
    if best.is_file():
        return best, None

    # Otherwise pick the model with the highest epoch number
    pattern = re.compile(r"model\.pth\.tar-(\d+)$")
    epoch_ckpts = []
    for p in adapter_dir.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                epoch_ckpts.append((int(m.group(1)), p))
    if not epoch_ckpts:
        raise FileNotFoundError("No checkpoint file found in 'adapter' directory")
    epoch_ckpts.sort(key=lambda t: t[0], reverse=True)
    return epoch_ckpts[0][1], epoch_ckpts[0][0]


def _build_cfg(dataset_cfg: pathlib.Path, trainer_cfg: pathlib.Path, root: pathlib.Path) -> "CfgNode":  # noqa: D401
    cfg = get_cfg_default()

    # Extend with custom keys defined in train.py
    from train import extend_cfg  # local import to avoid circular deps

    extend_cfg(cfg)

    # Inherit settings
    cfg.merge_from_file(str(dataset_cfg))
    cfg.merge_from_file(str(trainer_cfg))

    # Override a few options for evaluation
    cfg.DATASET.ROOT = str(root)
    cfg.OUTPUT_DIR = "_tmp_out"  # no artefacts needed
    cfg.SEED = 42

    cfg.TEST.NO_TEST = True  # we will run our own evaluation code

    # Disable training
    cfg.TRAIN.CHECKPOINT_FREQ = 0

    # Force trainer name because the base YAML may leave it blank
    cfg.TRAINER.NAME = "ADAPTER"
    cfg.MODEL.BACKBONE.NAME = "RN50"

    cfg.freeze()
    return cfg


def _get_logits_and_labels(trainer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the model on the *test* split and return (logits, labels)."""
    trainer.set_model_mode("eval")

    labels_all, logits_all = [], []
    for batch in trainer.test_loader:
        inp, lab = trainer.parse_batch_test(batch)
        with torch.no_grad():
            logits = trainer.model(inp)
        labels_all.append(lab.cpu())
        logits_all.append(logits.cpu())
    labels = torch.cat(labels_all)
    logits = torch.cat(logits_all)
    return logits, labels


def _reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Compute per-bin accuracy/confidence and ECE (scalar, percentage)."""
    assert probs.ndim == 2
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, num=n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_cnt = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i < n_bins - 1 else (conf > lo) & (conf <= hi)
        # include conf == 1.0 in last bin
        if i == n_bins - 1:
            m = (conf > lo) & (conf <= hi + 1e-12)
        if m.sum() > 0:
            bin_acc[i] = correct[m].mean()
            bin_conf[i] = conf[m].mean()
            bin_cnt[i] = m.sum()

    ece = np.sum(np.abs(bin_acc - bin_conf) * bin_cnt) / len(conf) * 100
    return bin_edges, bin_acc, bin_conf, ece


def plot_reliability(bin_edges, bin_acc, bin_conf, out_path: pathlib.Path):
    plt.figure(figsize=(6, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.bar(bin_centers, bin_acc, width=bin_edges[1] - bin_edges[0], alpha=0.6, edgecolor='black', label="Accuracy")
    plt.plot(bin_centers, bin_conf, marker='o', linestyle='-', color='red', label="Confidence")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    print(f"Saved reliability diagram to {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    ckpt_path, epoch = _find_checkpoint(run_dir)
    print(f"Loading checkpoint: {ckpt_path}, epoch: {epoch if epoch is not None else 'best'}")

    cfg = _build_cfg(pathlib.Path(args.dataset_cfg), pathlib.Path(args.trainer_cfg), pathlib.Path(args.root))

    set_random_seed(cfg.SEED)

    trainer = build_trainer(cfg)

    # Load weights
    trainer.load_model(str(run_dir), cfg, epoch=epoch)

    logits, labels = _get_logits_and_labels(trainer)

    probs = F.softmax(logits, dim=-1).numpy()
    labels_np = labels.numpy()

    bin_edges, bin_acc, bin_conf, ece = _reliability_bins(probs, labels_np, n_bins=args.bins)
    print(f"ECE: {ece:.2f}% (n_bins={args.bins})")

    out_path = pathlib.Path(args.output).expanduser().resolve()
    plot_reliability(bin_edges, bin_acc, bin_conf, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot reliability diagram for a saved model")
    parser.add_argument("--run-dir", required=True, help="Path to the *seed* directory of a run")
    parser.add_argument("--dataset-cfg", default="configs/datasets/fgvc_aircraft.yaml", help="Dataset YAML config file")
    parser.add_argument("--trainer-cfg", default="configs/trainers/GP_rbf.yaml", help="Trainer/method YAML config file")
    parser.add_argument("--root", default="/export/datasets/public/", help="Dataset root directory")
    parser.add_argument("--bins", type=int, default=15, help="Number of confidence bins")
    parser.add_argument("--output", default="reliability_fgvc.png", help="Output PNG file for the plot")

    main(parser.parse_args()) 