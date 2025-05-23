#!/usr/bin/env python3
"""
Parse CLAP baseline logs and aggregate metrics.

Usage
-----
# activate the same venv you used for training
source .venv/bin/activate

# run from the project root
python parse_baseline_logs.py              # default paths
python parse_baseline_logs.py \
    --out_csv results/baseline_metrics.csv # custom CSV name
"""
import argparse, csv, glob, os, re
from collections import defaultdict
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np

# ───────────────────────────────
# helpers
# ───────────────────────────────
LOG_GLOB = "output/**/**/seed*/log.txt"      # recursive glob (**) ≈ any depth

re_acc_best   = re.compile(r"acc_test\s+([0-9]+\.[0-9]+)")
re_acc_zs     = re.compile(r"Zero-Shot accuracy on test:\s+([0-9]+\.[0-9]+)")
re_ece        = re.compile(r"(?:ECE|ece)[^\d]*([0-9]+\.[0-9]+)")
re_nll        = re.compile(r"(?:NLL|nll)[^\d]*([0-9]+\.[0-9]+)")
re_dataset    = re.compile(r"output/(?P<dataset>[^/]+)/")
re_config     = re.compile(r"output/(?P<dataset>[^/]+)/(?P<config>[^/]+)/")
re_seed       = re.compile(r"/seed(?P<seed>\d+)/")
re_shots      = re.compile(r"(\d+)shots")

def parse_single_log(path: str) -> dict:
    """Return a dict with metrics parsed from *one* log file."""
    best_acc = 0.0
    zs_acc   = None
    ece_val  = None
    nll_val  = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if m := re_acc_best.search(line):
                best_acc = max(best_acc, float(m.group(1)))
            elif m := re_acc_zs.search(line):
                zs_acc = float(m.group(1))
            elif m := re_ece.search(line):
                ece_val = float(m.group(1))
            elif m := re_nll.search(line):
                nll_val = float(m.group(1))

    # Edge case: zero-shot run has no acc_test lines → fall back
    if best_acc == 0.0 and zs_acc is not None:
        best_acc = zs_acc

    return dict(acc = best_acc,
                zs_acc = zs_acc,
                ece = ece_val,
                nll = nll_val)

def config_to_method(config_str: str, shots: int) -> str:
    """Map config folder name to a human-readable method label."""
    if shots == 0:
        return "ZS-0"
    if "l2Constraint" in config_str:
        return "CLAP"
    return "ZS-LP"

# ───────────────────────────────
# plotting
# ───────────────────────────────
def create_plots(rows_by_key: dict, metrics_dir: str):
    """Create line plots comparing ZS-LP vs CLAP for each dataset."""
    # Group data by dataset
    datasets = defaultdict(lambda: defaultdict(list))
    
    for (dataset, shots, method), metrics_list in rows_by_key.items():
        if method in ["ZS-LP", "CLAP"]:  # Only plot these two methods
            accs = [d["acc"] for d in metrics_list if d["acc"] is not None]
            if accs:
                acc_mean = mean(accs)
                acc_std = stdev(accs) if len(accs) > 1 else 0
                datasets[dataset][method].append((shots, acc_mean, acc_std))
    
    # Create a plot for each dataset
    for dataset, methods_data in datasets.items():
        plt.figure(figsize=(10, 6))
        
        for method, data_points in methods_data.items():
            if not data_points:
                continue
                
            # Sort by shots
            data_points.sort(key=lambda x: x[0])
            shots = [x[0] for x in data_points]
            accs = [x[1] for x in data_points]
            stds = [x[2] for x in data_points]
            
            # Plot line with error bars
            plt.errorbar(shots, accs, yerr=stds, 
                        label=method, marker='o', linewidth=2, markersize=6,
                        capsize=5, capthick=1)
        
        plt.xlabel('Number of Shots', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Performance Comparison: {dataset}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(metrics_dir, f'{dataset}_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot: {plot_path}")

# ───────────────────────────────
# main
# ───────────────────────────────
def main(output_root: str, out_csv: str):
    # Create metrics directory
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Update CSV path to be in metrics directory
    csv_path = os.path.join(metrics_dir, os.path.basename(out_csv))
    
    rows_by_key = defaultdict(list)  # key = (dataset, shots, method)
    for log_path in glob.glob(os.path.join(output_root, LOG_GLOB), recursive=True):
        # extract dataset/config/seed from the *path*
        dataset = re_dataset.search(log_path).group("dataset")
        config  = re_config.search(log_path).group("config")
        seed    = int(re_seed.search(log_path).group("seed"))

        # shots & method derived from config string
        shots_match = re_shots.search(config)
        if shots_match:
            shots = int(shots_match.group(1))
        else:
            # Handle configs without shots pattern (e.g., zero-shot configs)
            shots = 0
            print(f"Warning: No shots pattern found in config '{config}', assuming 0-shot")
        method = config_to_method(config, shots)

        metrics = parse_single_log(log_path)
        rows_by_key[(dataset, shots, method)].append(metrics)

    # average across seeds and write CSV
    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        header = ["dataset", "shots", "method",
                  "acc_mean", "acc_std",
                  "ece_mean", "nll_mean", "zs_acc_mean"]
        writer.writerow(header)

        for (dataset, shots, method), lst in sorted(rows_by_key.items()):
            # Some runs (e.g. >0-shot) don't have zs_acc; handle None gracefully
            accs    = [d["acc"]     for d in lst if d["acc"]     is not None]
            eces    = [d["ece"]     for d in lst if d["ece"]     is not None]
            nlls    = [d["nll"]     for d in lst if d["nll"]     is not None]
            zs_accs = [d["zs_acc"]  for d in lst if d["zs_acc"]  is not None]

            row = [
                dataset,
                shots,
                method,
                f"{mean(accs):.3f}",
                f"{stdev(accs):.3f}"  if len(accs) > 1 else "",
                f"{mean(eces):.4f}"   if eces else "",
                f"{mean(nlls):.3f}"   if nlls else "",
                f"{mean(zs_accs):.3f}" if zs_accs else ""
            ]
            writer.writerow(row)

    print(f"✓ Parsed {len(rows_by_key)} experiment groups → {csv_path}")

    # Create plots
    create_plots(rows_by_key, metrics_dir)

# ───────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=".",
                        help="root folder containing experiment outputs")
    parser.add_argument("--out_csv", default="baseline_metrics.csv",
                        help="name of the summary CSV to create")
    args = parser.parse_args()
    main(args.output_root, args.out_csv)
