#!/usr/bin/env python3
"""
Parse CLAP cross-dataset transfer logs and aggregate metrics.

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
    best_acc = None
    zs_acc   = None
    ece_val  = None
    nll_val  = None
    completed = False

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
            # Check if experiment completed
            if "Finished" in content or "completed" in content or "acc_test" in content:
                completed = True
            
            for line in content.split('\n'):
                if m := re_acc_best.search(line):
                    if best_acc is None:
                        best_acc = float(m.group(1))
                    else:
                        best_acc = max(best_acc, float(m.group(1)))
                elif m := re_acc_zs.search(line):
                    zs_acc = float(m.group(1))
                elif m := re_ece.search(line):
                    ece_val = float(m.group(1))
                elif m := re_nll.search(line):
                    nll_val = float(m.group(1))
    except Exception as e:
        print(f"Warning: Error reading {path}: {e}")
        return dict(acc=None, zs_acc=None, ece=None, nll=None, completed=False)

    # Edge case: zero-shot run has no acc_test lines → fall back
    if best_acc is None and zs_acc is not None:
        best_acc = zs_acc

    return dict(acc = best_acc,
                zs_acc = zs_acc,
                ece = ece_val,
                nll = nll_val,
                completed = completed)

def config_to_method(config_str: str, shots: int) -> str:
    """Map config folder name to a human-readable method label."""
    if shots == 0:
        return "ZS-0"
    if "l2Constraint" in config_str:
        return "CLAP"
    return "ZS-LP"

def is_transfer_path(path: str, experiment_name: str) -> bool:
    """Check if a path corresponds to a cross-dataset transfer experiment."""
    # Look for transfer_X_to_Y pattern in the path
    transfer_pattern = rf"output/{re.escape(experiment_name)}/transfer_[^_]+_to_[^/]+/"
    return bool(re.search(transfer_pattern, path))

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
    
    if not all_datasets:
        print("No transfer data found for plotting")
        return
        
    all_datasets = sorted(list(all_datasets))
    
    # Create matrices for each method and shot combination
    for method in ["ZS-LP", "CLAP"]:
        if method not in transfer_data:
            continue
            
        for shots in sorted(transfer_data[method].keys()):
            # Create transfer matrix with relative improvements
            matrix = np.full((len(all_datasets), len(all_datasets)), np.nan)
            
            # First, get ZS-LP baseline performance for each target dataset
            zslp_baseline = {}
            if "ZS-LP" in transfer_data and shots in transfer_data["ZS-LP"]:
                for (source, target), acc in transfer_data["ZS-LP"][shots].items():
                    if target not in zslp_baseline:
                        zslp_baseline[target] = []
                    zslp_baseline[target].append(acc)
                
                # Average ZS-LP performance across all source datasets for each target
                for target in zslp_baseline:
                    zslp_baseline[target] = mean(zslp_baseline[target])
            
            # Compute relative improvements w.r.t. ZS-LP baseline
            for (source, target), acc in transfer_data[method][shots].items():
                source_idx = all_datasets.index(source)
                target_idx = all_datasets.index(target)
                
                if target in zslp_baseline:
                    # Relative improvement = (method_acc - baseline_acc)
                    relative_improvement = acc - zslp_baseline[target]
                    matrix[source_idx, target_idx] = relative_improvement
                else:
                    # If no baseline available, use raw accuracy
                    matrix[source_idx, target_idx] = acc
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            mask = np.isnan(matrix)
            
            # Create DataFrame for better labeling
            df = pd.DataFrame(matrix, index=all_datasets, columns=all_datasets)
            
            # Use a diverging colormap centered at 0 for relative improvements
            vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix))) if not np.all(np.isnan(matrix)) else 1
            
            sns.heatmap(df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                       mask=mask, cbar_kws={'label': 'Relative Improvement (%)'},
                       square=True, linewidths=0.5, vmin=-vmax, vmax=vmax)
            
            plt.title(f'Cross-Dataset Transfer Matrix: {method} ({shots} shots)\nRelative to ZS-LP Baseline', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Target Dataset', fontsize=12)
            plt.ylabel('Source Dataset', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(metrics_dir, f'transfer_matrix_{method}_{shots}shots_relative.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved relative improvement transfer matrix: {plot_path}")
    
    # Also create raw accuracy matrices for reference
    for method in ["ZS-LP", "CLAP"]:
        if method not in transfer_data:
            continue
            
        for shots in sorted(transfer_data[method].keys()):
            # Create raw accuracy matrix
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
            
            plt.title(f'Cross-Dataset Transfer Matrix: {method} ({shots} shots)\nRaw Accuracy', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Target Dataset', fontsize=12)
            plt.ylabel('Source Dataset', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(metrics_dir, f'transfer_matrix_{method}_{shots}shots_raw.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved raw accuracy transfer matrix: {plot_path}")
    
    # Create aggregate relative improvement heatmap (average across all shots for each method)
    for method in ["ZS-LP", "CLAP"]:
        if method not in transfer_data:
            continue
            
        # Compute relative improvements across all shots
        aggregate_matrix = np.full((len(all_datasets), len(all_datasets)), np.nan)
        count_matrix = np.zeros((len(all_datasets), len(all_datasets)))
        
        for shots in transfer_data[method].keys():
            # Get ZS-LP baseline for this shot setting
            zslp_baseline = {}
            if "ZS-LP" in transfer_data and shots in transfer_data["ZS-LP"]:
                for (source, target), acc in transfer_data["ZS-LP"][shots].items():
                    if target not in zslp_baseline:
                        zslp_baseline[target] = []
                    zslp_baseline[target].append(acc)
                
                for target in zslp_baseline:
                    zslp_baseline[target] = mean(zslp_baseline[target])
            
            # Compute relative improvements for this shot setting
            for (source, target), acc in transfer_data[method][shots].items():
                source_idx = all_datasets.index(source)
                target_idx = all_datasets.index(target)
                
                if target in zslp_baseline:
                    relative_improvement = acc - zslp_baseline[target]
                    if np.isnan(aggregate_matrix[source_idx, target_idx]):
                        aggregate_matrix[source_idx, target_idx] = 0
                    aggregate_matrix[source_idx, target_idx] += relative_improvement
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
        
        # Use a diverging colormap centered at 0
        vmax = max(abs(np.nanmin(aggregate_matrix)), abs(np.nanmax(aggregate_matrix))) if not np.all(np.isnan(aggregate_matrix)) else 1
        
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   mask=mask, cbar_kws={'label': 'Average Relative Improvement (%)'},
                   square=True, linewidths=0.5, vmin=-vmax, vmax=vmax)
        
        plt.title(f'Average Cross-Dataset Transfer: {method} (All Shots)\nRelative to ZS-LP Baseline', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Target Dataset', fontsize=12)
        plt.ylabel('Source Dataset', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(metrics_dir, f'transfer_matrix_{method}_aggregate_relative.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved aggregate relative improvement transfer matrix: {plot_path}")

# ───────────────────────────────
# main parsing functions
# ───────────────────────────────
def parse_transfer_logs(experiment_name: str, output_root: str) -> dict:
    """Parse cross-dataset transfer experiment logs."""
    log_pattern = LOG_GLOB.format(experiment=experiment_name)
    rows_by_key = defaultdict(list)  # key = (source, target, shots, method)
    
    total_logs = 0
    completed_logs = 0
    
    for log_path in glob.glob(os.path.join(output_root, log_pattern), recursive=True):
        # Only process transfer experiments
        if not is_transfer_path(log_path, experiment_name):
            continue
            
        total_logs += 1
            
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
        
        if metrics["completed"]:
            completed_logs += 1
            
        rows_by_key[(source, target, shots, method)].append(metrics)
    
    print(f"Found {total_logs} transfer experiment logs, {completed_logs} completed")
    return rows_by_key

def write_transfer_csv(rows_by_key: dict, csv_path: str):
    """Write cross-dataset transfer results to CSV."""
    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        header = ["source_dataset", "target_dataset", "shots", "method",
                  "acc_mean", "acc_std", "num_seeds", "completed_seeds",
                  "ece_mean", "nll_mean", "zs_acc_mean"]
        writer.writerow(header)

        for (source, target, shots, method), lst in sorted(rows_by_key.items()):
            accs    = [d["acc"]     for d in lst if d["acc"]     is not None]
            eces    = [d["ece"]     for d in lst if d["ece"]     is not None]
            nlls    = [d["nll"]     for d in lst if d["nll"]     is not None]
            zs_accs = [d["zs_acc"]  for d in lst if d["zs_acc"]  is not None]
            completed = sum(1 for d in lst if d["completed"])

            row = [
                source, target, shots, method,
                f"{mean(accs):.3f}" if accs else "N/A",
                f"{stdev(accs):.3f}" if len(accs) > 1 else ("0.000" if len(accs) == 1 else "N/A"),
                len(lst),
                completed,
                f"{mean(eces):.4f}" if eces else "N/A",
                f"{mean(nlls):.3f}" if nlls else "N/A",
                f"{mean(zs_accs):.3f}" if zs_accs else "N/A"
            ]
            writer.writerow(row)

# ───────────────────────────────
# main
# ───────────────────────────────
def main(experiment_name: str, output_root: str, out_csv: str):
    # Create metrics directory for this experiment
    metrics_dir = f"metrics/{experiment_name}"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Parse transfer experiments only
    print(f"Parsing cross-dataset transfer experiments for: {experiment_name}")
    transfer_rows = parse_transfer_logs(experiment_name, output_root)
    
    if transfer_rows:
        # Generate CSV output
        csv_path = os.path.join(metrics_dir, out_csv)
        write_transfer_csv(transfer_rows, csv_path)
        print(f"✓ Parsed {len(transfer_rows)} transfer experiment groups → {csv_path}")
        
        # Generate transfer matrices
        create_transfer_matrices(transfer_rows, metrics_dir)
        
        # Print summary statistics
        total_experiments = len(transfer_rows)
        completed_experiments = sum(1 for metrics_list in transfer_rows.values() 
                                  if any(d["completed"] for d in metrics_list))
        with_results = sum(1 for metrics_list in transfer_rows.values() 
                          if any(d["acc"] is not None for d in metrics_list))
        
        print(f"\nSummary:")
        print(f"  Total experiment configurations: {total_experiments}")
        print(f"  Configurations with completed runs: {completed_experiments}")
        print(f"  Configurations with results: {with_results}")
        
    else:
        print("No transfer experiments found")

# ───────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name",
                        help="name of the experiment")
    parser.add_argument("--output_root", default=".",
                        help="root folder containing experiment outputs")
    parser.add_argument("--out_csv", default="transfer_results.csv",
                        help="name of the summary CSV to create")
    args = parser.parse_args()
    main(args.experiment_name, args.output_root, args.out_csv)
