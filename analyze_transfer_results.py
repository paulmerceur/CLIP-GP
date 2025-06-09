#!/usr/bin/env python3
"""
analyze_transfer_results.py – Analysis of cross-dataset transfer learning experiments.
-------------------------------------------------------------------------------------

This script analyzes the evaluation results from the `cross_dataset_eval.sh`
campaign. It compares the transfer performance of three methods:
1. GP-weighted prototypes (10 templates)
2. Baseline with simple averaging (10 templates)
3. Baseline with a single curated template (1 template)

Usage:
    python analyze_transfer_results.py cross_dataset_eval

Outputs:
    - Comparative analysis of methods for each transfer task.
    - Aggregate performance summaries.
    - Transfer performance matrices for each method.
"""

import argparse
import pathlib
import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# ───────────────────────────
# 1. Regex helpers (from analyze_gp_tuning.py)
# ───────────────────────────
_RE = {
    "num_shots":     re.compile(r"\bNUM_SHOTS:\s+(\d+)"),
    "use_gp":        re.compile(r"\bUSE_GP:\s+(True|False)"),
    "num_templates": re.compile(r"\bNUM_TEMPLATES:\s+(\d+)"),
    "accuracy":      re.compile(r"\*\s+accuracy:\s+([\d.]+)%"),
    "macro_f1":      re.compile(r"\*\s+macro_f1:\s+([\d.]+)%"),
    "gp_beta":       re.compile(r"\bGP_BETA:\s+([\d.]+)"),
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
    """Extracts key information from a single log file."""
    txt = path.read_text(encoding="utf-8", errors="ignore")

    # Extract transfer path, e.g., "caltech101_to_dtd"
    transfer_pair_str = "unknown"
    for part in path.parts:
        if "_to_" in part:
            transfer_pair_str = part
            break
    
    source_dataset, target_dataset = "unknown", "unknown"
    if transfer_pair_str != "unknown":
        source_dataset, target_dataset = transfer_pair_str.split("_to_")

    info = {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "shots":          int(_first_match(_RE["num_shots"], txt, int) or 0),
        "use_gp":         _first_match(_RE["use_gp"], txt, str) == "True",
        "num_templates":  int(_first_match(_RE["num_templates"], txt, int) or 0),
        "accuracy":       float(_first_match(_RE["accuracy"], txt) or float("nan")),
        "macro_f1":       float(_first_match(_RE["macro_f1"], txt) or float("nan")),
        "gp_beta":        _first_match(_RE["gp_beta"], txt, float)
    }

    if info["shots"] == 0:
        return None
    
    # Define the method based on GP usage and template count
    if info["use_gp"] and info["num_templates"] == 10:
        info["method"] = "GP (10t)"
    elif not info["use_gp"] and info["num_templates"] == 10:
        info["method"] = "Baseline (10t)"
    elif not info["use_gp"] and info["num_templates"] == 1:
        info["method"] = "Baseline (1t)"
    else:
        info["method"] = "Unknown"
    
    return info

# ───────────────────────────
# 3. Data collection
# ───────────────────────────

def collect_logs(exp_name: str) -> pd.DataFrame:
    """Collects and parses all log files from the experiment directory."""
    base = pathlib.Path("output") / exp_name
    if not base.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {base}")
    
    rows: List[Dict[str, Any]] = []
    
    # Path format: output/{exp_name}/{source}_to_{target}/{config}/seed{N}/log.txt
    for log_file in base.glob("*_to_*/**/seed*/log.txt"):
        info = parse_log(log_file)
        if info is not None:
            info["seed"] = log_file.parent.name
            rows.append(info)
        else:
            print(f"Warning: Could not parse {log_file}")

    if not rows:
        raise RuntimeError("No valid log files found")
    
    return pd.DataFrame(rows)

# ───────────────────────────
# 4. Analysis functions
# ───────────────────────────

def analyze_aggregate_performance(df: pd.DataFrame):
    """Provides an aggregate summary of each method's performance."""
    print("\n" + "=" * 80)
    print("AGGREGATE PERFORMANCE SUMMARY")
    print("=" * 80)

    # Group by method and calculate mean/std accuracy across all runs
    agg_stats = df.groupby('method')['accuracy'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    
    print("Overall transfer performance (averaged across all tasks and seeds):\n")
    for method, row in agg_stats.iterrows():
        print(f"  - {method:15s} | Acc: {row['mean']:5.2f} ± {row['std']:.2f}% | (N={int(row['count'])})")
    
    best_method = agg_stats.index[0]
    print(f"\n🏆 Best overall method: {best_method}")

def analyze_transfer_matrices(df: pd.DataFrame):
    """Displays transfer performance as a matrix for each method."""
    print("\n" + "=" * 80)
    print("TRANSFER PERFORMANCE MATRICES")
    print("=" * 80)

    # First, compute mean accuracy across seeds
    mean_df = df.groupby(['method', 'shots', 'source_dataset', 'target_dataset'])['accuracy'].mean().reset_index()
    
    # Get sorted lists of datasets for consistent matrix dimensions
    datasets = sorted(mean_df['source_dataset'].unique())

    for method in sorted(mean_df['method'].unique()):
        for shots in sorted(mean_df['shots'].unique()):
            subset = mean_df[(mean_df['method'] == method) & (mean_df['shots'] == shots)]
            
            if subset.empty:
                continue

            # Create the transfer matrix using pivot_table
            transfer_matrix = subset.pivot_table(
                index='source_dataset', 
                columns='target_dataset', 
                values='accuracy'
            )
            
            # Reorder for consistency
            transfer_matrix = transfer_matrix.reindex(index=datasets, columns=datasets)

            # Calculate the mean accuracy for transfers from each source
            transfer_matrix['AVG_TRANSFER'] = transfer_matrix.mean(axis=1)

            print(f"\n--- Method: {method} ({shots}-shot) ---")
            # Print formatted table
            print(transfer_matrix.to_string(float_format="%.2f", na_rep="-"))

def analyze_comparative_performance(df: pd.DataFrame):
    """Compares the methods side-by-side for each transfer task."""
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARATIVE ANALYSIS")
    print("=" * 80)

    # Compute mean accuracy across seeds
    mean_df = df.groupby(['source_dataset', 'target_dataset', 'shots', 'method'])['accuracy'].mean().unstack()
    
    # Define the desired order of methods for display
    method_order = ['Baseline (1t)', 'Baseline (10t)', 'GP (10t)']
    # Ensure all methods are present, filling missing ones with NaN
    for method in method_order:
        if method not in mean_df.columns:
            mean_df[method] = np.nan
    mean_df = mean_df[method_order]

    # Find the best method for each row
    mean_df['winner'] = mean_df.idxmax(axis=1)

    # Print the results, grouped by shot
    for shots in sorted(df['shots'].unique()):
        print(f"\n--- Comparative Performance ({shots}-shot) ---\n")
        
        shot_df = mean_df[mean_df.index.get_level_values('shots') == shots]
        
        if shot_df.empty:
            continue
            
        # Drop the 'shots' level from the index for cleaner printing
        shot_df.index = shot_df.index.droplevel('shots')
        
        print(shot_df.to_string(float_format="%.2f", na_rep="-"))
        print("-" * 80)

# ───────────────────────────
# 5. Main function
# ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze cross-dataset transfer learning results.")
    parser.add_argument("experiment", help="Experiment name (subdirectory of ./output)")
    args = parser.parse_args()
    
    try:
        print(f"Analyzing experiment: {args.experiment}")
        df = collect_logs(args.experiment)
        print(f"Found {len(df)} experiment results across {df['seed'].nunique()} seeds.")
        
        # Run all analyses
        analyze_aggregate_performance(df)
        analyze_transfer_matrices(df)
        analyze_comparative_performance(df)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 