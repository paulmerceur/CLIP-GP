#!/usr/bin/env python3
"""
Parse CLAP baseline logs and aggregate metrics.

Usage
-----
# activate the same venv you used for training
source .venv/bin/activate

# run from the project root
python parse_baseline_logs.py my_experiment              # specify experiment name
python parse_baseline_logs.py my_experiment \
    --out_csv my_results.csv                             # custom CSV name
"""
import argparse, csv, glob, os, re
from collections import defaultdict
from statistics import mean, stdev
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ───────────────────────────────
# helpers
# ───────────────────────────────
LOG_GLOB = "output/{experiment}/**/**/seed*/log.txt"      # recursive glob (**) ≈ any depth

re_acc_best   = re.compile(r"acc_test\s+([0-9]+\.[0-9]+)")
re_acc_zs     = re.compile(r"Zero-Shot accuracy on test:\s+([0-9]+\.[0-9]+)")
re_ece        = re.compile(r"(?:ECE|ece)[^\d]*([0-9]+\.[0-9]+)")
re_nll        = re.compile(r"(?:NLL|nll)[^\d]*([0-9]+\.[0-9]+)")
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

def detect_experiment_type(experiment_name: str, output_root: str) -> str:
    """Detect if this is a cross-dataset transfer experiment or regular baseline."""
    log_pattern = LOG_GLOB.format(experiment=experiment_name)
    sample_paths = list(glob.glob(os.path.join(output_root, log_pattern), recursive=True))[:5]
    
    for path in sample_paths:
        if "transfer" in path:
            return "cross_dataset"
    return "baseline"

# ───────────────────────────────
# baseline plotting
# ───────────────────────────────
def create_baseline_plots(rows_by_key: dict, metrics_dir: str):
    """Create line plots comparing ZS-LP vs CLAP for each dataset (baseline experiments)."""
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
# cross-dataset matrix plotting
# ───────────────────────────────
def create_transfer_matrices(rows_by_key: dict, metrics_dir: str):
    """Create transfer matrix heatmaps for cross-dataset experiments."""
    # Group data by method and shots
    transfer_data = defaultdict(lambda: defaultdict(dict))  # method -> shots -> (source, target) -> acc
    
    for (source, target, shots, method), metrics_list in rows_by_key.items():
        accs = [d["acc"] for d in metrics_list if d["acc"] is not None]
        if accs:
            acc_mean = mean(accs)
            transfer_data[method][shots][(source, target)] = acc_mean
    
    # Get all unique datasets
    all_datasets = set()
    for method_data in transfer_data.values():
        for shots_data in method_data.values():
            for source, target in shots_data.keys():
                all_datasets.add(source)
                all_datasets.add(target)
    all_datasets = sorted(list(all_datasets))
    
    # Create matrices for each method and shot combination
    for method in ["ZS-LP", "CLAP"]:
        if method not in transfer_data:
            continue
            
        for shots in sorted(transfer_data[method].keys()):
            # Create transfer matrix
            matrix = np.full((len(all_datasets), len(all_datasets)), np.nan)
            
            for (source, target), acc in transfer_data[method][shots].items():
                source_idx = all_datasets.index(source)
                target_idx = all_datasets.index(target)
                matrix[source_idx, target_idx] = acc
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            mask = np.isnan(matrix)
            
            # Create DataFrame for better labeling
            df = pd.DataFrame(matrix, index=all_datasets, columns=all_datasets)
            
            sns.heatmap(df, annot=True, fmt='.3f', cmap='viridis', 
                       mask=mask, cbar_kws={'label': 'Accuracy'},
                       square=True, linewidths=0.5)
            
            plt.title(f'Cross-Dataset Transfer Matrix: {method} ({shots} shots)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Target Dataset', fontsize=12)
            plt.ylabel('Source Dataset', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(metrics_dir, f'transfer_matrix_{method}_{shots}shots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved transfer matrix: {plot_path}")
    
    # Create aggregate transfer heatmap (average across all shots for each method)
    for method in ["ZS-LP", "CLAP"]:
        if method not in transfer_data:
            continue
            
        # Average across all shots
        aggregate_matrix = np.full((len(all_datasets), len(all_datasets)), np.nan)
        count_matrix = np.zeros((len(all_datasets), len(all_datasets)))
        
        for shots_data in transfer_data[method].values():
            for (source, target), acc in shots_data.items():
                source_idx = all_datasets.index(source)
                target_idx = all_datasets.index(target)
                if np.isnan(aggregate_matrix[source_idx, target_idx]):
                    aggregate_matrix[source_idx, target_idx] = 0
                aggregate_matrix[source_idx, target_idx] += acc
                count_matrix[source_idx, target_idx] += 1
        
        # Compute averages
        with np.errstate(invalid='ignore'):
            aggregate_matrix = np.where(count_matrix > 0, 
                                      aggregate_matrix / count_matrix, 
                                      np.nan)
        
        # Create aggregate heatmap
        plt.figure(figsize=(12, 10))
        mask = np.isnan(aggregate_matrix)
        
        df = pd.DataFrame(aggregate_matrix, index=all_datasets, columns=all_datasets)
        
        sns.heatmap(df, annot=True, fmt='.3f', cmap='viridis',
                   mask=mask, cbar_kws={'label': 'Average Accuracy'},
                   square=True, linewidths=0.5)
        
        plt.title(f'Average Cross-Dataset Transfer: {method} (All Shots)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Target Dataset', fontsize=12)
        plt.ylabel('Source Dataset', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(metrics_dir, f'transfer_matrix_{method}_aggregate.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved aggregate transfer matrix: {plot_path}")

# ───────────────────────────────
# main parsing functions
# ───────────────────────────────
def parse_baseline_experiment(experiment_name: str, output_root: str, metrics_dir: str) -> dict:
    """Parse regular baseline experiment logs."""
    log_pattern = LOG_GLOB.format(experiment=experiment_name)
    rows_by_key = defaultdict(list)  # key = (dataset, shots, method)
    
    for log_path in glob.glob(os.path.join(output_root, log_pattern), recursive=True):
        # extract dataset/config/seed from the *path*
        dataset_match = re.search(rf"output/{re.escape(experiment_name)}/([^/]+)/", log_path)
        config_match = re.search(rf"output/{re.escape(experiment_name)}/([^/]+)/([^/]+)/", log_path)
        seed_match = re_seed.search(log_path)
        
        if not (dataset_match and config_match and seed_match):
            print(f"Warning: Could not parse baseline path {log_path}")
            continue
            
        dataset = dataset_match.group(1)
        config = config_match.group(2)
        seed = int(seed_match.group("seed"))

        # shots & method derived from config string
        shots_match = re_shots.search(config)
        if shots_match:
            shots = int(shots_match.group(1))
        else:
            shots = 0
            
        method = config_to_method(config, shots)
        metrics = parse_single_log(log_path)
        rows_by_key[(dataset, shots, method)].append(metrics)
    
    return rows_by_key

def parse_cross_dataset_experiment(experiment_name: str, output_root: str, metrics_dir: str) -> dict:
    """Parse cross-dataset transfer experiment logs."""
    log_pattern = LOG_GLOB.format(experiment=experiment_name)
    rows_by_key = defaultdict(list)  # key = (source, target, shots, method)
    
    for log_path in glob.glob(os.path.join(output_root, log_pattern), recursive=True):
        # Extract transfer info from path
        transfer_match = re.search(rf"output/{re.escape(experiment_name)}/transfer_([^_]+)_to_([^/]+)/", log_path)
        config_match = re.search(rf"transfer_[^/]+/([^/]+)/", log_path)
        seed_match = re_seed.search(log_path)
        
        if not (transfer_match and config_match and seed_match):
            print(f"Warning: Could not parse transfer path {log_path}")
            continue
            
        source = transfer_match.group(1)
        target = transfer_match.group(2)
        config = config_match.group(1)
        seed = int(seed_match.group("seed"))

        # shots & method derived from config string
        shots_match = re_shots.search(config)
        if shots_match:
            shots = int(shots_match.group(1))
        else:
            shots = 0
            
        method = config_to_method(config, shots)
        metrics = parse_single_log(log_path)
        rows_by_key[(source, target, shots, method)].append(metrics)
    
    return rows_by_key

def write_baseline_csv(rows_by_key: dict, csv_path: str):
    """Write baseline experiment results to CSV."""
    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        header = ["dataset", "shots", "method",
                  "acc_mean", "acc_std",
                  "ece_mean", "nll_mean", "zs_acc_mean"]
        writer.writerow(header)

        for (dataset, shots, method), lst in sorted(rows_by_key.items()):
            accs    = [d["acc"]     for d in lst if d["acc"]     is not None]
            eces    = [d["ece"]     for d in lst if d["ece"]     is not None]
            nlls    = [d["nll"]     for d in lst if d["nll"]     is not None]
            zs_accs = [d["zs_acc"]  for d in lst if d["zs_acc"]  is not None]

            row = [
                dataset, shots, method,
                f"{mean(accs):.3f}",
                f"{stdev(accs):.3f}"  if len(accs) > 1 else "",
                f"{mean(eces):.4f}"   if eces else "",
                f"{mean(nlls):.3f}"   if nlls else "",
                f"{mean(zs_accs):.3f}" if zs_accs else ""
            ]
            writer.writerow(row)

def write_transfer_csv(rows_by_key: dict, csv_path: str):
    """Write cross-dataset transfer results to CSV."""
    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        header = ["source_dataset", "target_dataset", "shots", "method",
                  "acc_mean", "acc_std",
                  "ece_mean", "nll_mean", "zs_acc_mean"]
        writer.writerow(header)

        for (source, target, shots, method), lst in sorted(rows_by_key.items()):
            accs    = [d["acc"]     for d in lst if d["acc"]     is not None]
            eces    = [d["ece"]     for d in lst if d["ece"]     is not None]
            nlls    = [d["nll"]     for d in lst if d["nll"]     is not None]
            zs_accs = [d["zs_acc"]  for d in lst if d["zs_acc"]  is not None]

            row = [
                source, target, shots, method,
                f"{mean(accs):.3f}",
                f"{stdev(accs):.3f}"  if len(accs) > 1 else "",
                f"{mean(eces):.4f}"   if eces else "",
                f"{mean(nlls):.3f}"   if nlls else "",
                f"{mean(zs_accs):.3f}" if zs_accs else ""
            ]
            writer.writerow(row)

# ───────────────────────────────
# main
# ───────────────────────────────
def main(experiment_name: str, output_root: str, out_csv: str):
    # Create metrics directory for this experiment
    metrics_dir = f"metrics/{experiment_name}"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Update CSV path to be in experiment's metrics directory
    csv_path = os.path.join(metrics_dir, os.path.basename(out_csv))
    
    # Detect experiment type
    exp_type = detect_experiment_type(experiment_name, output_root)
    print(f"Detected experiment type: {exp_type}")
    
    if exp_type == "cross_dataset":
        # Parse cross-dataset transfer results
        rows_by_key = parse_cross_dataset_experiment(experiment_name, output_root, metrics_dir)
        write_transfer_csv(rows_by_key, csv_path)
        create_transfer_matrices(rows_by_key, metrics_dir)
        print(f"✓ Parsed {len(rows_by_key)} transfer experiment groups → {csv_path}")
        
    else:
        # Parse regular baseline results
        rows_by_key = parse_baseline_experiment(experiment_name, output_root, metrics_dir)
        write_baseline_csv(rows_by_key, csv_path)
        create_baseline_plots(rows_by_key, metrics_dir)
        print(f"✓ Parsed {len(rows_by_key)} baseline experiment groups → {csv_path}")

# ───────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name",
                        help="name of the experiment")
    parser.add_argument("--output_root", default=".",
                        help="root folder containing experiment outputs")
    parser.add_argument("--out_csv", default="baseline_metrics.csv",
                        help="name of the summary CSV to create")
    args = parser.parse_args()
    main(args.experiment_name, args.output_root, args.out_csv)
