#!/usr/bin/env python3
"""aggregate_results.py
Aggregate experimental results logged under ./output/EXPERIMENT_NAME/ …

The script walks the directory tree produced by *scripts/adapt.sh* runs and
summarises test metrics (accuracy, ECE, AECE) across random seeds. It prints a
readable table to stdout and saves plots per dataset (accuracy, ECE, AECE).

Usage
-----
python scripts/aggregate_results.py EXP_NAME

Example
-------
python scripts/aggregate_results.py gp_test5
"""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# ───────────────────────────
# Regex helpers
# ───────────────────────────
_RE_ACC = re.compile(r"\*\s+accuracy:\s+([\d.]+)%", re.IGNORECASE)
_RE_ECE = re.compile(r"\*\s+ECE:\s+([\d.]+)%", re.IGNORECASE)
_RE_AECE = re.compile(r"\*\s+AECE:\s+([\d.]+)%", re.IGNORECASE)
_RE_SHOTS_IN_DIR = re.compile(r"_(\d+)shots?", re.IGNORECASE)

# Prefer explicit base LR matches and avoid GP_LR collisions
_RE_LOG_LR_UPPER = re.compile(r"(?<!GP_)LR:\s+([\d.eE+-]+)")  # matches 'LR:' but not 'GP_LR:'
_RE_LOG_LR_LOWER = re.compile(r"(?<!gp_)lr:\s+([\d.eE+-]+)")  # matches 'lr:' but not 'gp_lr:'
_RE_LOG_GP_LR = re.compile(r"GP_LR:\s+([\d.eE+-]+)", re.IGNORECASE)
_RE_LOG_GP_BETA = re.compile(r"GP_BETA:\s+([\d.eE+-]+)", re.IGNORECASE)
_RE_LOG_L2_LAMBDA = re.compile(r"L2_LAMBDA:\s+([\d.eE+-]+)", re.IGNORECASE)

# Fallback: read lr from directory name suffix like '_lr0.001'
_RE_DIR_LR = re.compile(r"_lr([\d.eE+-]+)", re.IGNORECASE)

# ───────────────────────────
# Single-file parser
# ───────────────────────────

def parse_log(log_path: pathlib.Path) -> Tuple[float | None, float | None, float | None, str | None, str | None, str | None, str | None]:
    """Return (accuracy, ece, aece, lr, gp_lr, gp_beta, l2_lambda) extracted from *log_path*.

    Robust to different capitalisations and avoids confusing LR with GP_LR.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"! Could not read {log_path}: {e}")
        return None, None, None, None, None, None, None

    acc_match = _RE_ACC.search(text)
    ece_match = _RE_ECE.search(text)
    aece_match = _RE_AECE.search(text)
    # Try base LR in multiple forms, avoiding GP_LR
    lr_match = _RE_LOG_LR_UPPER.search(text) or _RE_LOG_LR_LOWER.search(text)
    gp_lr_match = _RE_LOG_GP_LR.search(text)
    gp_beta_match = _RE_LOG_GP_BETA.search(text)
    l2_lambda_match = _RE_LOG_L2_LAMBDA.search(text)

    if acc_match is None:
        print(f"! Could not read {log_path}")
        return None, None, None, None, None, None, None

    acc = float(acc_match.group(1)) if acc_match else None
    ece = float(ece_match.group(1)) if ece_match else None
    aece = float(aece_match.group(1)) if aece_match else None
    lr = lr_match.group(1) if lr_match else None
    gp_lr = gp_lr_match.group(1) if gp_lr_match else None
    gp_beta = gp_beta_match.group(1) if gp_beta_match else None
    l2_lambda = l2_lambda_match.group(1) if l2_lambda_match else None

    return acc, ece, aece, lr, gp_lr, gp_beta, l2_lambda


# ───────────────────────────
# Experiment walker
# ───────────────────────────

def walk_experiment(exp_name: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return nested dict keyed by dataset, each value is list of config records."""
    base = pathlib.Path("output") / exp_name
    if not base.is_dir():
        raise FileNotFoundError(base)

    results: Dict[str, List[Dict[str, Any]]] = {}

    # directory layout: output/<exp>/<dataset>/<config>/seed*/log.txt
    for dataset_dir in sorted(base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        dataset_records: List[Dict[str, Any]] = []

        for config_dir in sorted(dataset_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            cfg_name_full = config_dir.name
            # Extract #shots from directory name (fallback to 0)
            m = _RE_SHOTS_IN_DIR.search(cfg_name_full)
            num_shots = int(m.group(1)) if m else 0
            # Config label without trailing "_4shots"
            config_label = cfg_name_full[: m.start()] if m else cfg_name_full

            # Treat each config_dir as a variant holder (our runs encode lr in the name)
            variant_dirs = [config_dir]

            for variant_dir in variant_dirs:
                name_dir = variant_dir.name
                # Prefer LR from log; fallback to directory suffix '_lr<val>'
                dir_lr_match = _RE_DIR_LR.search(name_dir)

                variant_label = name_dir

                acc_values: List[float] = []
                ece_values: List[float] = []
                aece_values: List[float] = []
                lr_val: str | None = dir_lr_match.group(1) if dir_lr_match else None
                gp_lr_val: str | None = None
                gp_beta_val: str | None = None
                l2_lambda_val: str | None = None

                for log_file in variant_dir.glob("seed*/log.txt"):
                    acc, ece, aece, lr, gp_lr, gp_beta, l2_lambda = parse_log(log_file)
                    if acc is not None:
                        acc_values.append(acc)
                    if ece is not None:
                        ece_values.append(ece)
                    if aece is not None:
                        aece_values.append(aece)
                    # Prefer log-derived LR over dir-derived; take first non-None
                    if lr is not None:
                        lr_val = lr
                    if gp_lr_val is None and gp_lr is not None:
                        gp_lr_val = gp_lr
                    if gp_beta_val is None and gp_beta is not None:
                        gp_beta_val = gp_beta
                    if l2_lambda_val is None and l2_lambda is not None:
                        l2_lambda_val = l2_lambda
                if not acc_values and not ece_values:
                    continue  # skip empty variant

                record = {
                    "config": variant_label,
                    "lr": lr_val,
                    "gp_lr": gp_lr_val,
                    "gp_beta": gp_beta_val,
                    "l2_lambda": l2_lambda_val,
                    "shots": num_shots,
                    "n_seeds": max(len(acc_values), len(ece_values), len(aece_values)),
                    "acc_mean": statistics.mean(acc_values) if acc_values else float("nan"),
                    "acc_std": statistics.stdev(acc_values) if len(acc_values) > 1 else 0.0,
                    "ece_mean": statistics.mean(ece_values) if ece_values else float("nan"),
                    "ece_std": statistics.stdev(ece_values) if len(ece_values) > 1 else 0.0,
                    "aece_mean": statistics.mean(aece_values) if aece_values else float("nan"),
                    "aece_std": statistics.stdev(aece_values) if len(aece_values) > 1 else 0.0,
                }
                dataset_records.append(record)

        # sort by shots then config name for nicer display
        dataset_records.sort(key=lambda r: (r["shots"], r["config"]))
        results[dataset_name] = dataset_records

    return results


# ───────────────────────────
# Pretty printer
# ───────────────────────────

def print_results(results: Dict[str, List[Dict[str, Any]]]):
    for dataset, records in results.items():
        if not records:
            continue
        print("\n=== Dataset:", dataset, "===")
        header = (
            f"{'Config':<35} {'LR':>8} {'GP_LR':>8} {'GP_BETA':>8} {'L2_LAMBDA':>10} {'Shots':>5} {'Seeds':>5} | "
            f"{'Acc µ':>7} {'Acc σ':>7} | {'ECE µ':>7} {'ECE σ':>7} | {'AECE µ':>7} {'AECE σ':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in records:
            print(
                f"{r['config']:<35} {r['lr'] or '-':>8} {r['gp_lr'] or '-':>8} {r['gp_beta'] or '-':>8} "
                f"{r['l2_lambda'] or '-':>10} "
                f"{r['shots']:>5d} {r['n_seeds']:>5d} | "
                f"{r['acc_mean']:7.2f} {r['acc_std']:7.2f} | "
                f"{r['ece_mean']:7.2f} {r['ece_std']:7.2f} | "
                f"{r['aece_mean']:7.2f} {r['aece_std']:7.2f}"
            )


# ───────────────────────────
# Plotting helpers
# ───────────────────────────


def make_plots(results: Dict[str, List[Dict[str, Any]]], exp_name: str):
    """Save accuracy and ECE plots to plots/<exp_name>/<dataset>_*.png.

    For each *dataset*, two plots are created (accuracy & ECE). Each line
    corresponds to a configuration ("run").  The baseline configuration is
    labelled "baseline"; the others are labelled "GP_B{beta}_<config>".
    """

    plots_dir = pathlib.Path("plots") / exp_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    for dataset, records in results.items():
        if not records:
            continue

        # Group records by configuration label
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            if "baseline" in r["config"]:
                label = "baseline"
            else:
                label = f"GP_L2{r['l2_lambda']}_B{r['gp_beta']}"
            grouped.setdefault(label, []).append(r)

        # Combined figure with Accuracy, ECE and AECE side-by-side
        fig, (ax_acc, ax_ece, ax_aece) = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

        # Collect unique shot counts for common x-axis ticks
        unique_shots = sorted({rec["shots"] for rec in records if rec["shots"] > 0})

        for label, recs in grouped.items():
            recs_sorted = sorted(recs, key=lambda x: x["shots"])
            xs = [rec["shots"] for rec in recs_sorted]

            # Accuracy
            ys_acc = [rec["acc_mean"] for rec in recs_sorted]
            ax_acc.plot(xs, ys_acc, marker="o", label=label)

            # ECE
            ys_ece = [rec["ece_mean"] for rec in recs_sorted]
            ax_ece.plot(xs, ys_ece, marker="o", label=label)

            # AECE
            ys_aece = [rec["aece_mean"] for rec in recs_sorted]
            ax_aece.plot(xs, ys_aece, marker="o", label=label)

        if unique_shots:
            ax_acc.set_xticks(unique_shots)
            ax_ece.set_xticks(unique_shots)
            ax_aece.set_xticks(unique_shots)

        ax_acc.set_xlabel("# Shots")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Accuracy")

        ax_ece.set_xlabel("# Shots")
        ax_ece.set_ylabel("ECE (%)")
        ax_ece.set_title("ECE")

        ax_aece.set_xlabel("# Shots")
        ax_aece.set_ylabel("AECE (%)")
        ax_aece.set_title("AECE")

        # Shared legend – place slightly below the title
        handles, labels = ax_acc.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),  # y < 1 leaves room for title
            ncol=len(labels),
            frameon=False,
        )

        fig.suptitle(dataset, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(plots_dir / f"{dataset}_metrics.png")
        plt.close(fig)


# ───────────────────────────
# CLI
# ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Aggregate experiment logs and print accuracy/ECE.")
    ap.add_argument("experiment", help="Name of the sub-folder under ./output")
    args = ap.parse_args()

    results = walk_experiment(args.experiment)
    if not results:
        print("No results found – check experiment name and path.")
        return
    print_results(results)
    make_plots(results, args.experiment)


if __name__ == "__main__":
    main() 