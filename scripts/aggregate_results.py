#!/usr/bin/env python3
"""aggregate_results.py
Aggregate experimental results logged under ./output/EXPERIMENT_NAME/ …

The script walks the directory tree produced by *scripts/adapt.sh* runs and
summarises test metrics (accuracy & ECE) across random seeds.  It prints a
readable table to stdout – no CSV or plots, just quick inspection.

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
from typing import Dict, List, Tuple

# ───────────────────────────
# Regex helpers
# ───────────────────────────
_RE_ACC = re.compile(r"\*\s+accuracy:\s+([\d.]+)%")
_RE_ECE = re.compile(r"\*\s+ECE:\s+([\d.]+)%")
_RE_SHOTS_IN_DIR = re.compile(r"_(\d+)shots?", re.IGNORECASE)
# Pattern for sub-directory naming the GP learning-rate & β (created by scripts/adapt.sh)
_RE_LR_BETA_DIR = re.compile(r"LR([\d.eE+-]+)_B([\d.eE+-]+)")

# Parse hyper-parameters from log files (TRAINER.ADAPTER.* entries)
_RE_LOG_LR = re.compile(r"GP_LR:\s+([\d.eE+-]+)")
_RE_LOG_BETA = re.compile(r"GP_BETA:\s+([\d.eE+-]+)")
_RE_LOG_LAMBDA = re.compile(r"RESID_LAMBDA:\s+([\d.eE+-]+)")


# ───────────────────────────
# Single-file parser
# ───────────────────────────

def parse_log(log_path: pathlib.Path) -> Tuple[float | None, float | None, str | None, str | None, str | None]:
    """Return (accuracy, ece, lr, beta) extracted from *log_path*."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"! Could not read {log_path}: {e}")
        return None, None, None, None, None, None

    acc_match = _RE_ACC.search(text)
    ece_match = _RE_ECE.search(text)
    lr_match = _RE_LOG_LR.search(text)
    beta_match = _RE_LOG_BETA.search(text)
    lambda_resid_match = _RE_LOG_LAMBDA.search(text)

    acc = float(acc_match.group(1)) if acc_match else None
    ece = float(ece_match.group(1)) if ece_match else None
    lr = lr_match.group(1) if lr_match else None
    beta = beta_match.group(1) if beta_match else None
    lambda_resid = lambda_resid_match.group(1) if lambda_resid_match else None

    return acc, ece, lr, beta, lambda_resid


# ───────────────────────────
# Experiment walker
# ───────────────────────────

def walk_experiment(exp_name: str) -> Dict[str, List[Dict[str, float]]]:
    """Return nested dict keyed by dataset, each value is list of config records."""
    base = pathlib.Path("output") / exp_name
    if not base.is_dir():
        raise FileNotFoundError(base)

    results: Dict[str, List[Dict[str, float]]] = {}

    # directory layout: output/<exp>/<dataset>/<config>/seed*/log.txt
    for dataset_dir in sorted(base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        dataset_records: List[Dict[str, float]] = []

        for config_dir in sorted(dataset_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            cfg_name_full = config_dir.name  # e.g. GP_rbf_length1e-2_4shots
            # Extract #shots from directory name (fallback to 0)
            m = _RE_SHOTS_IN_DIR.search(cfg_name_full)
            num_shots = int(m.group(1)) if m else 0
            # Config label without trailing "_4shots"
            config_label = cfg_name_full[: m.start()] if m else cfg_name_full

            # 1️⃣ Find variant sub-directories (LR / Beta combos)
            variant_dirs = [d for d in config_dir.iterdir()
                            if d.is_dir() and _RE_LR_BETA_DIR.match(d.name)]

            # If no such dirs, treat *config_dir* itself as variant holder
            if not variant_dirs:
                variant_dirs = [config_dir]

            for variant_dir in variant_dirs:
                name_dir = variant_dir.name
                m_lr = _RE_LR_BETA_DIR.search(name_dir)
                lr_val = m_lr.group(1) if m_lr else None
                beta_val = m_lr.group(2) if m_lr else None

                variant_label = config_label  # keep base name only

                acc_values: List[float] = []
                ece_values: List[float] = []
                lr_val: str | None = None
                beta_val: str | None = None
                lambda_resid_val: str | None = None

                for log_file in variant_dir.glob("seed*/log.txt"):
                    acc, ece, lr, beta, lambda_resid = parse_log(log_file)
                    if acc is not None:
                        acc_values.append(acc)
                    if ece is not None:
                        ece_values.append(ece)
                    # Take first non-None hyper-param value (should be same across seeds)
                    if lr_val is None and lr is not None:
                        lr_val = lr
                    if beta_val is None and beta is not None:
                        beta_val = beta
                    if lambda_resid_val is None and lambda_resid is not None:
                        lambda_resid_val = lambda_resid
                if not acc_values and not ece_values:
                    continue  # skip empty variant

                record = {
                    "config": variant_label,
                    "lr": lr_val,
                    "beta": beta_val,
                    "lambda_resid": lambda_resid_val,
                    "shots": num_shots,
                    "n_seeds": max(len(acc_values), len(ece_values)),
                    "acc_mean": statistics.mean(acc_values) if acc_values else float("nan"),
                    "acc_std": statistics.stdev(acc_values) if len(acc_values) > 1 else 0.0,
                    "ece_mean": statistics.mean(ece_values) if ece_values else float("nan"),
                    "ece_std": statistics.stdev(ece_values) if len(ece_values) > 1 else 0.0,
                }
                dataset_records.append(record)

        # sort by shots then config name for nicer display
        dataset_records.sort(key=lambda r: (r["shots"], r["config"]))
        results[dataset_name] = dataset_records

    return results


# ───────────────────────────
# Pretty printer
# ───────────────────────────

def print_results(results: Dict[str, List[Dict[str, float]]]):
    for dataset, records in results.items():
        if not records:
            continue
        print("\n=== Dataset:", dataset, "===")
        header = (
            f"{'Config':<25} {'LR':>8} {'Beta':>8} {'Lambda':>8} {'Shots':>5} {'Seeds':>5} | "
            f"{'Acc µ':>7} {'Acc σ':>7} | {'ECE µ':>7} {'ECE σ':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in records:
            print(
                f"{r['config']:<25} {r['lr'] or '-':>8} {r['beta'] or '-':>8} "
                f"{r['lambda_resid'] or '-':>8} "
                f"{r['shots']:>5d} {r['n_seeds']:>5d} | "
                f"{r['acc_mean']:7.2f} {r['acc_std']:7.2f} | "
                f"{r['ece_mean']:7.2f} {r['ece_std']:7.2f}"
            )


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


if __name__ == "__main__":
    main() 