#!/usr/bin/env python3
"""
analyze_gp_tuning.py – numerical analysis of GP hyperparameter tuning results
-----------------------------------------------------------------------------

This script analyzes the results from gp_hyperparam_tuning.sh, focusing on 
finding the best configurations for each dataset/shots combination and 
providing overall conclusions.

Usage:
    python analyze_gp_tuning.py gp_grid

Outputs:
    - Best configs per (dataset, shots) combination
    - Overall ranking of GP configurations
    - Statistical summary and conclusions
"""

import argparse
import pathlib
import re
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

# ───────────────────────────
# 1. Regex helpers (copied from analyze_experiment.py)
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
    # GP params
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
# 2. Single‑log parser (copied from analyze_experiment.py)
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
# 3. Config identification
# ───────────────────────────

def get_config_name(log_path: pathlib.Path) -> str:
    """Extract config name from path like .../4shot_conservative_rbf/seed1/log.txt"""
    parts = log_path.parts
    for part in parts:
        if "shot_" in part:
            return part.split("shot_", 1)[1]
    return "unknown"

def build_config_signature(info: Dict[str, Any]) -> str:
    """Build a unique signature for this configuration"""
    if not info["use_gp"]:
        return "baseline"
    
    gp_params = []
    for key in ["lr", "weight_decay", "max_epoch", "batch_size", "num_templates", "gp_lr", "gp_beta", "gp_kernel", "gp_length", "gp_output", "gp_noise", "gp_samples"]:
        if info[key] is not None:
            gp_params.append(f"{key}={info[key]}")
    gp_params.append(f"diag={info['gp_diag']}")
    
    return "GP_" + "_".join(gp_params)

# ───────────────────────────
# 4. Data collection
# ───────────────────────────

def collect_logs(exp_name: str) -> pd.DataFrame:
    base = pathlib.Path("output") / exp_name
    if not base.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {base}")
    
    rows: List[Dict[str, Any]] = []
    for dataset_dir in base.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        
        for config_dir in dataset_dir.iterdir():
            if not config_dir.is_dir():
                continue
            
            config_name = get_config_name(config_dir / "seed1" / "log.txt")
            
            for log_file in config_dir.glob("seed*/log.txt"):
                info = parse_log(log_file)
                if info is not None:
                    info.update(
                        dataset=dataset,
                        config_name=config_name,
                        config_signature=build_config_signature(info),
                        seed=log_file.parent.name,
                        shots=info["num_shots"]
                    )
                    rows.append(info)
                else:
                    print(f"Warning: No valid info found for {log_file}")

    if not rows:
        raise RuntimeError("No valid log files found")
    
    return pd.DataFrame(rows)

# ───────────────────────────
# 5. Analysis functions
# ───────────────────────────

def analyze_best_configs_per_experiment(df: pd.DataFrame) -> None:
    """Print best configs for each (dataset, shots) combination"""
    print("=" * 80)
    print("BEST CONFIGURATIONS PER EXPERIMENT")
    print("=" * 80)
    
    # Group by dataset, shots, config and compute mean accuracy across seeds
    grouped = df.groupby(['dataset', 'shots', 'config_name']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'macro_f1': ['mean', 'std'],
        'lr': 'first',
        'weight_decay': 'first',
        'max_epoch': 'first',
        'batch_size': 'first',
        'num_templates': 'first',
        'use_gp': 'first',
        'gp_lr': 'first',
        'gp_beta': 'first',
        'gp_kernel': 'first',
        'gp_length': 'first',
        'gp_output': 'first',
        'gp_noise': 'first',
        'gp_samples': 'first',
        'gp_diag': 'first'
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
    
    for dataset in sorted(df['dataset'].unique()):
        for shots in sorted(df['shots'].unique()):
            subset = grouped[(grouped['dataset'] == dataset) & (grouped['shots'] == shots)]
            if subset.empty:
                continue
                
            # Sort by mean accuracy
            subset = subset.sort_values('accuracy_mean', ascending=False)
            
            print(f"\n{dataset.upper()} - {shots} shots:")
            print("-" * 50)
            
            # Show top 5 configs
            for i, (_, row) in enumerate(subset.head(5).iterrows()):
                rank = i + 1
                acc_mean = row['accuracy_mean']
                acc_std = row['accuracy_std']
                config = row['config_name']
                is_gp = row['use_gp_first']
                
                print(f"{rank:2d}. {config:20s} | Acc: {acc_mean:5.2f}±{acc_std:4.2f}% | GP: {is_gp}")
                
                # Show GP hyperparams for GP configs
                if is_gp and not pd.isna(row['gp_lr_first']):
                    gp_info = []
                    if not pd.isna(row['lr_first']):
                        gp_info.append(f"lr={row['lr_first']}")
                    if not pd.isna(row['weight_decay_first']):
                        gp_info.append(f"wd={row['weight_decay_first']}")
                    if not pd.isna(row['max_epoch_first']):
                        gp_info.append(f"epochs={row['max_epoch_first']}")
                    if not pd.isna(row['gp_lr_first']):
                        gp_info.append(f"lr={row['gp_lr_first']}")
                    if not pd.isna(row['gp_beta_first']):
                        gp_info.append(f"β={row['gp_beta_first']}")
                    if not pd.isna(row['gp_kernel_first']):
                        gp_info.append(f"kernel={row['gp_kernel_first']}")
                    if not pd.isna(row['gp_length_first']):
                        gp_info.append(f"lengthscale={row['gp_length_first']}")
                    if gp_info:
                        print(f"    GP params: {', '.join(gp_info)}")

def analyze_overall_ranking(df: pd.DataFrame) -> None:
    """Analyze overall performance across all experiments"""
    print("\n" + "=" * 80)
    print("OVERALL CONFIGURATION RANKING")
    print("=" * 80)
    
    # Compute mean accuracy for each config across all experiments
    overall_stats = df.groupby('config_name').agg({
        'accuracy': ['mean', 'std', 'count'],
        'macro_f1': ['mean', 'std'],
        'lr': 'first',
        'weight_decay': 'first',
        'max_epoch': 'first',
        'batch_size': 'first',
        'num_templates': 'first',
        'use_gp': 'first',
        'gp_lr': 'first',
        'gp_beta': 'first', 
        'gp_kernel': 'first',
        'gp_length': 'first',
        'gp_output': 'first',
        'gp_noise': 'first',
        'gp_samples': 'first',
        'gp_diag': 'first'
    }).reset_index()
    
    # Flatten column names
    overall_stats.columns = ['_'.join(col).strip('_') for col in overall_stats.columns]
    overall_stats = overall_stats.sort_values('accuracy_mean', ascending=False)
    
    print("\nTop 10 configurations (averaged across all datasets and shots):")
    print("-" * 80)
    
    for i, (_, row) in enumerate(overall_stats.head(10).iterrows()):
        rank = i + 1
        config = row['config_name']
        acc_mean = row['accuracy_mean']
        acc_std = row['accuracy_std']
        count = int(row['accuracy_count'])
        is_gp = row['use_gp_first']
        
        print(f"{rank:2d}. {config:20s} | Acc: {acc_mean:5.2f}±{acc_std:4.2f}% | N={count:2d} | GP: {is_gp}")

def analyze_gp_vs_baseline(df: pd.DataFrame) -> None:
    """Compare GP methods vs baseline"""
    print("\n" + "=" * 80)
    print("GP vs BASELINE COMPARISON")
    print("=" * 80)
    
    gp_results = df[df['use_gp'] == True]['accuracy']
    baseline_results = df[df['use_gp'] == False]['accuracy']
    
    print(f"\nBaseline (no GP):")
    print(f"  Mean accuracy: {baseline_results.mean():.2f}±{baseline_results.std():.2f}%")
    print(f"  N experiments: {len(baseline_results)}")
    
    print(f"\nGP methods:")
    print(f"  Mean accuracy: {gp_results.mean():.2f}±{gp_results.std():.2f}%")
    print(f"  N experiments: {len(gp_results)}")
    
    # Statistical test (simple)
    if len(gp_results) > 0 and len(baseline_results) > 0:
        improvement = gp_results.mean() - baseline_results.mean()
        print(f"\nMean improvement: {improvement:+.2f} percentage points")

def analyze_hyperparameter_effects(df: pd.DataFrame) -> None:
    """Analyze the effect of different hyperparameters"""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER EFFECT ANALYSIS")
    print("=" * 80)
    
    gp_df = df[df['use_gp'] == True].copy()
    
    if len(gp_df) == 0:
        print("No GP experiments found")
        return
    
    # Analyze each hyperparameter
    hyperparams = ['lr', 'weight_decay', 'max_epoch', 'batch_size', 'num_templates']
    hyperparams += ['gp_lr', 'gp_beta', 'gp_kernel', 'gp_length', 'gp_output', 'gp_noise', 'gp_samples', 'gp_diag']
    # Only keep hyperparams that change between configs
    hyperparams = [param for param in hyperparams if gp_df[param].nunique() > 1]
    
    for param in hyperparams:
        if param not in gp_df.columns or gp_df[param].isna().all():
            continue
            
        print(f"\n{param.upper()}:")
        print("-" * 30)
        
        if param == 'gp_kernel' or param == 'gp_diag':
            # Categorical analysis
            stats = gp_df.groupby(param)['accuracy'].agg(['mean', 'std', 'count'])
            for value, row in stats.iterrows():
                print(f"  {value}: {row['mean']:5.2f}±{row['std']:4.2f}% (N={int(row['count'])})")
        else:
            # Numerical analysis - show best and worst values
            corr = gp_df[[param, 'accuracy']].corr().iloc[0, 1]
            print(f"  Correlation with accuracy: {corr:.3f}")
            
            # Show top 3 values
            top_configs = gp_df.nlargest(3, 'accuracy')[[param, 'accuracy', 'config_name']]
            print(f"  Top 3 configs:")
            for _, row in top_configs.iterrows():
                print(f"    {row[param]:8} -> {row['accuracy']:5.2f}% ({row['config_name']})")

def print_conclusions(df: pd.DataFrame) -> None:
    """Print overall conclusions and recommendations"""
    print("\n" + "=" * 80)
    print("CONCLUSIONS & RECOMMENDATIONS")
    print("=" * 80)
    
    # Overall best config
    best_config = df.groupby('config_name')['accuracy'].mean().idxmax()
    best_acc = df.groupby('config_name')['accuracy'].mean().max()
    
    print(f"\n🏆 BEST OVERALL CONFIG: {best_config}")
    print(f"   Average accuracy: {best_acc:.2f}%")
    
    # Best GP config
    gp_df = df[df['use_gp'] == True]
    if len(gp_df) > 0:
        best_gp_config = gp_df.groupby('config_name')['accuracy'].mean().idxmax()
        best_gp_acc = gp_df.groupby('config_name')['accuracy'].mean().max()
        print(f"\n🔧 BEST GP CONFIG: {best_gp_config}")
        print(f"   Average accuracy: {best_gp_acc:.2f}%")
    
    # Check if GP helps overall
    baseline_mean = df[df['use_gp'] == False]['accuracy'].mean()
    gp_mean = df[df['use_gp'] == True]['accuracy'].mean()
    
    if gp_mean > baseline_mean:
        print(f"\n✅ GP methods show improvement (+{gp_mean - baseline_mean:.2f}% on average)")
    else:
        print(f"\n❌ GP methods show decline ({gp_mean - baseline_mean:.2f}% on average)")
    
    # Dataset-specific recommendations
    print(f"\n📊 DATASET-SPECIFIC INSIGHTS:")
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        best_for_dataset = dataset_df.groupby('config_name')['accuracy'].mean().idxmax()
        best_acc_dataset = dataset_df.groupby('config_name')['accuracy'].mean().max()
        print(f"   {dataset:15s}: {best_for_dataset:20s} ({best_acc_dataset:.2f}%)")

# ───────────────────────────
# 6. Main function
# ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze GP hyperparameter tuning results")
    parser.add_argument("experiment", help="Experiment name (subdirectory of ./output)")
    args = parser.parse_args()
    
    try:
        print(f"Analyzing experiment: {args.experiment}")
        df = collect_logs(args.experiment)
        print(f"Found {len(df)} experiment results")
        
        # Run all analyses
        analyze_best_configs_per_experiment(df)
        analyze_overall_ranking(df)
        analyze_gp_vs_baseline(df)
        analyze_hyperparameter_effects(df)
        print_conclusions(df)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 