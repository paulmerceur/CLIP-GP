#!/usr/bin/env python3
"""
analyze_experiment.py – aggregate CLAP/GP few‑shot logs
------------------------------------------------------

The script walks the `output/<exp>/<dataset>/<config>/seedN/` tree, parses every
`log.txt`, aggregates metrics across seeds and number‑of‑shots, writes a summary
CSV and draws accuracy‑vs‑shot curves.

Usage
-----
python analyze_experiment.py EXP_NAME
python analyze_experiment.py EXP_NAME \
       --important_params BACKBONE,GP_KERNEL_TYPE \
       --metrics accuracy,macro_f1

Outputs
-------
* <EXP_NAME>_summary.csv       – wide CSV (mean & std for each metric)
* plots/<dataset>_accuracy.png – per‑dataset line plot
"""

import argparse
import pathlib
import re
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────
# 1. Regex helpers
# ───────────────────────────
_RE = {
    "num_shots":     re.compile(r"\bNUM_SHOTS:\s+(\d+)"),
    "use_gp":        re.compile(r"\bUSE_GP:\s+(True|False)"),
    "num_templates": re.compile(r"\bNUM_TEMPLATES:\s+(\d+)"),
    "backbone":      re.compile(r"\bbackbone:\s+(\S+)"),
    "max_epoch":     re.compile(r"\bMAX_EPOCH:\s+(\d+)"),
    "lr":            re.compile(r"\bLR:\s+([\d.]+)"),
    "weight_decay":  re.compile(r"\bWEIGHT_DECAY:\s+([\d.]+)"),
    "batch_size":    re.compile(r"\bBATCH_SIZE:\s+(\d+)"),
    "accuracy":      re.compile(r"\*\s+accuracy:\s+([\d.]+)%"),
    "macro_f1":      re.compile(r"\*\s+macro_f1:\s+([\d.]+)%"),
    # Optional GP params
    "gp_lr":         re.compile(r"\bGP_LR:\s+([\d.]+)"),
    "gp_beta":       re.compile(r"\bGP_BETA:\s+([\d.]+)"),
    "gp_kernel":     re.compile(r"\bGP_KERNEL_TYPE:\s+(\S+)"),
    "gp_length":     re.compile(r"\bGP_LENGTHSCALE:\s+([\d.]+)"),
    "gp_output":     re.compile(r"\bGP_OUTPUTSCALE:\s+([\d.]+)"),
    "gp_noise":      re.compile(r"\bGP_NOISE:\s+([\d.]+)"),
    "gp_samples":    re.compile(r"\bGP_NUM_MC_SAMPLES:\s+(\d+)"),
    "gp_diag":       re.compile(r"\bGP_USE_DIAGONAL_COV:\s+(True|False)"),
}


def _first_match(regex: re.Pattern, text: str, cast=float):
    """Return the first capture group converted by *cast* (None → str)."""
    m = regex.search(text)
    if not m:
        return None
    return cast(m.group(1)) if cast else m.group(1)


# ───────────────────────────
# 2. Single‑log parser
# ───────────────────────────

def parse_log(path: pathlib.Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8", errors="ignore")

    info = {
        "num_shots":     int(_first_match(_RE["num_shots"], txt, int) or 0),
        "use_gp":        _first_match(_RE["use_gp"], txt, str) == "True",
        "num_templates": int(_first_match(_RE["num_templates"], txt, int) or 0),
        "backbone":      _first_match(_RE["backbone"], txt, str),
        "max_epoch":     int(_first_match(_RE["max_epoch"], txt, int) or 0),
        "lr":            float(_first_match(_RE["lr"], txt, float) or float("nan")),
        "weight_decay":  float(_first_match(_RE["weight_decay"], txt, float) or float("nan")),
        "batch_size":    int(_first_match(_RE["batch_size"], txt, int) or 0),
        "accuracy":      float(_first_match(_RE["accuracy"], txt) or float("nan")),
        "macro_f1":      float(_first_match(_RE["macro_f1"], txt) or float("nan")),
        # GP params
        "gp_lr":         _first_match(_RE["gp_lr"], txt, float),
        "gp_beta":       _first_match(_RE["gp_beta"], txt, float),
        "gp_kernel":     _first_match(_RE["gp_kernel"], txt, str),
        "gp_length":     _first_match(_RE["gp_length"], txt, float),
        "gp_output":     _first_match(_RE["gp_output"], txt, float),
        "gp_noise":      _first_match(_RE["gp_noise"], txt, float),
        "gp_samples":    _first_match(_RE["gp_samples"], txt, int),
        "gp_diag":       _first_match(_RE["gp_diag"], txt, str) == "True",
    }

    if info["num_shots"] == 0:
        return None
    
    return info


# ───────────────────────────
# 3. Label builder
# ───────────────────────────

def build_label(info: Dict[str, Any], important: List[str]) -> str:
    """Compose a readable config label."""
    prefix = "GP" if info["use_gp"] else "BASELINE"
    label = f"{prefix}_{info['num_templates']}TEMPLATES"

    for param in important:
        key = param.lower()
        if key in info and info[key] not in (None, "", float("nan")):
            label += f"_{param.upper()}={info[key]}"
    return label


# ───────────────────────────
# 4. Experiment walker
# ───────────────────────────

def collect_logs(exp_name: str, important: List[str]) -> pd.DataFrame:
    base = pathlib.Path("output") / exp_name
    if not base.is_dir():
        raise FileNotFoundError(base)
    
    df = pd.DataFrame()
    for dataset_dir in base.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for config_dir in dataset_dir.iterdir():
            if not config_dir.is_dir():
                continue
            for log_file in config_dir.glob("seed*/log.txt"):
                info = parse_log(log_file)
                if info is not None:
                    df = pd.concat([df, pd.DataFrame([info])], ignore_index=True)
    
    general_params = ["lr", "weight_decay", "max_epoch", "batch_size", "num_templates"]
    gp_params = ["gp_lr", "gp_beta", "gp_kernel", "gp_length", "gp_output", "gp_noise", "gp_samples", "gp_diag"]
    hyperparams = important + [param for param in gp_params if param not in important]
    hyperparams += [param for param in general_params if param not in important]
    hyperparams = [param for param in hyperparams if df[param].nunique() > 1]
    hyperparams = []

    rows: List[Dict[str, Any]] = []
    for dataset_dir in base.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for config_dir in dataset_dir.iterdir():
            if not config_dir.is_dir():
                continue
            for log_file in config_dir.glob("seed*/log.txt"):
                info = parse_log(log_file)
                if info is not None:
                    info.update(
                        dataset=dataset,
                        shot=info["num_shots"],
                        config_label=build_label(info, hyperparams),
                        seed=log_file.parent.name,
                    )
                    rows.append(info)
                else:
                    print(f"No info found for {log_file}")

    if not rows:
        raise RuntimeError("No log files found – check path or experiment name.")
    return pd.DataFrame(rows)


# ───────────────────────────
# 5. Aggregation & CSV
# ───────────────────────────

def summarise(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    agg = {m: ["mean", "std"] for m in metrics}
    summary = (
        df.groupby(["dataset", "config_label", "shot"], dropna=False)
          .agg(agg)
          .reset_index()
    )
    # Flatten MultiIndex → dataset, config_label, shot, accuracy_mean, accuracy_std,…
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary.sort_values(["dataset", "config_label", "shot"], inplace=True)
    return summary


# ───────────────────────────
# 6. Plotting (accuracy only for now)
# ───────────────────────────

def plot_accuracy(summary: pd.DataFrame, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in summary.dataset.unique():
        sub = summary[summary["dataset"] == dataset]
        # Print top 5 configs by accuracy, per dataset and number of shots
        # for shot in sub["shot"].unique():
        #     print(f"Top 5 configs for {dataset} with {shot} shots:")
        #     pd.set_option('display.max_colwidth', None)
        #     print(sub[sub["shot"] == shot].sort_values("accuracy_mean", ascending=False).head(5)[["config_label", "accuracy_mean"]])
        #     print("\n")

        plt.figure()
        
        for label, grp in sub.groupby("config_label"):
            plt.plot(grp["shot"], grp["accuracy_mean"], marker="o", label=label)
            # Optional confidence band
            plt.fill_between(grp["shot"],
                             grp["accuracy_mean"] - grp["accuracy_std"],
                             grp["accuracy_mean"] + grp["accuracy_std"],
                             alpha=0.15)
        plt.title(dataset)
        plt.xlabel("# shots")
        plt.ylabel("Test accuracy (%)")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset}_accuracy.png", dpi=144)
        plt.close()


# ───────────────────────────
# 7. CLI
# ───────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", help="sub‑directory of ./output to analyse")
    p.add_argument("--important_params", default="", help="comma‑separated list")
    p.add_argument("--metrics", default="accuracy", help="comma‑separated list")
    args = p.parse_args()

    important = [s.strip() for s in args.important_params.split(',') if s.strip()]
    metrics   = [s.strip() for s in args.metrics.split(',') if s.strip()]

    results_dir = pathlib.Path("results") / args.experiment
    results_dir.mkdir(parents=True, exist_ok=True)

    df = collect_logs(args.experiment, important)
    summary = summarise(df, metrics)

    csv_path = results_dir / f"{args.experiment}_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"✓ CSV written to {csv_path}")

    if "accuracy" in metrics:
        plot_accuracy(summary, results_dir)
        print(f"✓ Plots saved to {results_dir}")


if __name__ == "__main__":
    main()
