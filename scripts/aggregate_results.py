#!/usr/bin/env python3
"""
Aggregate metrics.json files under output/<experiment>/ and print quick summaries.
Also generate plots (Accuracy, ECE, AECE vs shots) per dataset and update a global CSV.

Usage:
  python scripts/aggregate_json.py <experiment_name> [--update-csv] [--csv runs.csv]

This expects a layout like:
  output/<experiment>/<dataset>/<config>/seed*/metrics.json

Where metrics.json minimally contains:
  {
    "dataset": ..., "shots": ..., "seed": ..., "method": "baseline"|"gp",
    "backbone": ..., "metrics": {"top1_acc": ..., "ece": ..., "aece": ...},
    "config": { ... full config ... }
  }
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import statistics
import math
import matplotlib.pyplot as plt


def load_runs(exp_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for dataset_dir in sorted(d for d in exp_dir.iterdir() if d.is_dir()):
        for config_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
            for seed_dir in sorted(d for d in config_dir.glob("seed*")):
                metrics_path = seed_dir / "metrics.json"
                if not metrics_path.is_file():
                    continue
                try:
                    payload = json.loads(metrics_path.read_text())
                    payload["_dataset_dir"] = dataset_dir.name
                    payload["_config_label"] = config_dir.name
                    payload["_seed_dir"] = seed_dir.name
                    runs.append(payload)
                except Exception:
                    pass
    return runs


def group_by_dataset_shots_config(runs: List[Dict[str, Any]]):
    grouped: Dict[str, Dict[int, Dict[str, List[Dict[str, Any]]]]] = {}
    for r in runs:
        ds = r.get("dataset") or r.get("_dataset_dir")
        shots = int(r.get("shots", 0))
        cfg = r.get("_config_label", "config")
        grouped.setdefault(ds, {}).setdefault(shots, {}).setdefault(cfg, []).append(r)
    return grouped


def print_summary(grouped):
    for ds, shots_map in grouped.items():
        print(f"\n=== Dataset: {ds} ===")
        # Determine the max config label length for column width
        max_cfg_len = max((len(cfg) for cfg_map in shots_map.values() for cfg in cfg_map), default=6)
        print(f"{'Config':<{max_cfg_len}} {'Shots':>5} {'Seeds':>5} | {'Acc µ':>7} {'Acc σ':>7} | {'ECE µ':>7} {'ECE σ':>7} | {'AECE µ':>7} {'AECE σ':>7}")
        print("-" * (max_cfg_len + 66))
        rows = []
        for shots, cfg_map in sorted(shots_map.items()):
            for cfg, rs in sorted(cfg_map.items()):
                accs = [float(r["metrics"].get("top1_acc", float('nan'))) for r in rs]
                eces = [float(r["metrics"].get("ece", float('nan'))) for r in rs]
                aeces = [float(r["metrics"].get("aece", float('nan'))) for r in rs]
                n = len(rs)
                acc_mean = statistics.fmean(accs) if accs else float('nan')
                acc_std = statistics.pstdev(accs) if n > 1 else 0.0
                ece_mean = statistics.fmean(eces) if eces else float('nan')
                ece_std = statistics.pstdev(eces) if n > 1 else 0.0
                aece_mean = statistics.fmean(aeces) if aeces else float('nan')
                aece_std = statistics.pstdev(aeces) if n > 1 else 0.0
                rows.append((cfg, shots, n, acc_mean, acc_std, ece_mean, ece_std, aece_mean, aece_std))
        rows.sort(key=lambda x: (x[1], x[0]))
        for cfg, shots, n, acc_m, acc_s, ece_m, ece_s, aece_m, aece_s in rows:
            print(f"{cfg:<{max_cfg_len}} {shots:>5d} {n:>5d} | {acc_m:7.2f} {acc_s:7.2f} | {ece_m:7.3f} {ece_s:7.3f} | {aece_m:7.3f} {aece_s:7.3f}")


def make_plots(grouped, exp_name: str):
    plots_dir = Path("output") / exp_name / "_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ds, shots_map in grouped.items():
        # Build per-config shot->mean maps
        per_cfg: Dict[str, Dict[int, Dict[str, float]]] = {}
        all_shots: List[int] = sorted(shots_map.keys())
        for shots, cfg_map in shots_map.items():
            for cfg, rs in cfg_map.items():
                accs = [float(r["metrics"].get("top1_acc", float('nan'))) for r in rs]
                eces = [float(r["metrics"].get("ece", float('nan'))) for r in rs]
                aeces = [float(r["metrics"].get("aece", float('nan'))) for r in rs]
                # Normalize config label across shots to draw a single line
                fam = cfg.replace(f"_{shots}shots", "_shots")
                per_cfg.setdefault(fam, {})[shots] = {
                    "acc": statistics.fmean(accs) if accs else float('nan'),
                    "ece": statistics.fmean(eces) if eces else float('nan'),
                    "aece": statistics.fmean(aeces) if aeces else float('nan'),
                }

        # Dynamic figure size and legend layout based on number of configs
        num_cfg = len(per_cfg)
        # Aim for ~2-3 legend rows. Cap columns reasonably.
        ncol = min(max(4, int(math.ceil(num_cfg / 3))) if num_cfg else 4, 16)
        nrows = int(math.ceil(num_cfg / max(1, ncol))) if num_cfg else 1
        # Grow figure width with number of configs; cap to avoid excessive sizes
        fig_w = max(16, min(48, 16 + 0.4 * max(0, num_cfg - 8)))
        # Height scales with legend rows
        fig_h = max(5, min(20, 5 + 0.8 * max(0, nrows - 1)))

        fig, (ax_acc, ax_ece, ax_aece) = plt.subplots(1, 3, figsize=(fig_w, fig_h), sharex=True)
        for cfg, shot_map in per_cfg.items():
            xs = [s for s in all_shots if s in shot_map]
            if not xs:
                continue
            ys_acc = [shot_map[s]["acc"] for s in xs]
            ys_ece = [shot_map[s]["ece"] for s in xs]
            ys_aece = [shot_map[s]["aece"] for s in xs]
            ax_acc.plot(xs, ys_acc, marker="o", linestyle="-", label=cfg)
            ax_ece.plot(xs, ys_ece, marker="o", linestyle="-", label=cfg)
            ax_aece.plot(xs, ys_aece, marker="o", linestyle="-", label=cfg)

        for ax, title, ylabel in [
            (ax_acc, "Accuracy", "Accuracy (%)"),
            (ax_ece, "ECE", "ECE"),
            (ax_aece, "AECE", "AECE"),
        ]:
            ax.set_xlabel("# Shots")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(all_shots)

        handles, labels = ax_acc.get_legend_handles_labels()
        if labels:
            # Reserve dynamic top space for legend; place legend in that space
            legend_fontsize = 8 if num_cfg > 12 else 9
            top_space = min(0.40, 0.14 + 0.06 * max(0, nrows - 1))
            y_legend = 1.0 - top_space / 2.0
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, y_legend),
                ncol=min(ncol, max(1, len(labels))),
                frameon=False,
                fontsize=legend_fontsize,
                columnspacing=0.8,
                handlelength=1.8,
                handletextpad=0.6,
                borderaxespad=0.2,
            )
            fig.suptitle(ds, y=1.0)
            fig.tight_layout(rect=(0, 0, 1, 1 - top_space))
        else:
            fig.suptitle(ds, y=0.99)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(plots_dir / f"{ds}_metrics.png")
        plt.close(fig)


def update_global_csv(grouped, exp_name: str, csv_path: Path) -> None:
    """Append aggregated rows (mean over seeds) to a global CSV.

    One row per dataset × shots × config label.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name", "dataset", "shots", "config_label", "n_seeds",
        "acc_mean", "acc_std", "ece_mean", "ece_std", "aece_mean", "aece_std",
    ]
    exists = csv_path.is_file()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for ds, shots_map in grouped.items():
            for shots, cfg_map in shots_map.items():
                for cfg, rs in cfg_map.items():
                    accs = [float(r["metrics"].get("top1_acc", float('nan'))) for r in rs]
                    eces = [float(r["metrics"].get("ece", float('nan'))) for r in rs]
                    aeces = [float(r["metrics"].get("aece", float('nan'))) for r in rs]
                    n = len(rs)
                    row = {
                        "experiment_name": exp_name,
                        "dataset": ds,
                        "shots": shots,
                        "config_label": cfg,
                        "n_seeds": n,
                        "acc_mean": statistics.fmean(accs) if accs else float('nan'),
                        "acc_std": statistics.pstdev(accs) if n > 1 else 0.0,
                        "ece_mean": statistics.fmean(eces) if eces else float('nan'),
                        "ece_std": statistics.pstdev(eces) if n > 1 else 0.0,
                        "aece_mean": statistics.fmean(aeces) if aeces else float('nan'),
                        "aece_std": statistics.pstdev(aeces) if n > 1 else 0.0,
                    }
                    writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics.json runs under output/<experiment>/")
    ap.add_argument("experiment", help="Experiment subfolder under output/")
    ap.add_argument("--csv", default="runs.csv", help="Global CSV filename under output/")
    ap.add_argument("--update-csv", action="store_true", help="Append aggregated rows to global CSV (off by default)")
    args = ap.parse_args()

    exp_dir = Path("output") / args.experiment
    runs = load_runs(exp_dir)
    if not runs:
        print("No metrics.json found. Did the runs finish?")
        return
    grouped = group_by_dataset_shots_config(runs)
    print_summary(grouped)
    make_plots(grouped, args.experiment)
    if args.update_csv:
        update_global_csv(grouped, args.experiment, Path("output") / args.csv)


if __name__ == "__main__":
    main()
