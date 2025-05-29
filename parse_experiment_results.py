#!/usr/bin/env python3
"""
Script to parse experiment results and generate CSV summaries and plots.

This script:
1. Parses all log.txt files in the experiment directory structure
2. Extracts configuration parameters and metrics 
3. Generates CSV files with averaged results across seeds
4. Creates line plots comparing different methods across shot numbers

Usage: python parse_experiment_results.py <experiment_name>
Example: python parse_experiment_results.py default_baseline
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def parse_config_from_filename(config_dir: str) -> Dict[str, str]:
    """
    Parse configuration parameters from directory name.
    
    Expected format: {optimizer}_lr{lr}_B{batch_size}_ep{epochs}_{init_type}_{constraint}_{shots}shots_{nb_templates}templates
    Example: SGD_lr1e-1_B256_ep300_ZSInit_l2Constraint_1shots_10templates
    """
    config = {}
    
    # Extract shots
    shots_match = re.search(r'(\d+)shots', config_dir)
    if shots_match:
        config['num_shots'] = int(shots_match.group(1))
    
    # Extract optimizer and learning rate
    optim_lr_match = re.search(r'([A-Z]+)_lr([^_]+)', config_dir)
    if optim_lr_match:
        config['optimizer'] = optim_lr_match.group(1)
        config['learning_rate'] = optim_lr_match.group(2)
    
    # Extract batch size
    batch_match = re.search(r'_B(\d+)_', config_dir)
    if batch_match:
        config['batch_size'] = int(batch_match.group(1))
    
    # Extract epochs
    epoch_match = re.search(r'_ep(\d+)_', config_dir)
    if epoch_match:
        config['epochs'] = int(epoch_match.group(1))
    
    # Extract initialization type
    init_match = re.search(r'_([A-Z]+)Init_', config_dir)
    if init_match:
        config['init_type'] = init_match.group(1)
    
    # Extract constraint type
    constraint_match = re.search(r'_([^_]+)Constraint_', config_dir)
    if constraint_match:
        config['constraint'] = constraint_match.group(1)
    
    # Extract number of templates
    template_match = re.search(r'_(\d+)templates', config_dir)
    if template_match:
        config['nb_templates'] = int(template_match.group(1))
    
    # Extract GP
    gp_match = re.search(r'_GP', config_dir)
    if gp_match:
        config['gp'] = True
    else:
        config['gp'] = False
    
    return config

def parse_log_file(log_path: str) -> Optional[Dict[str, float]]:
    """
    Parse metrics from a log.txt file.
    
    Returns:
        Dictionary with parsed metrics or None if parsing fails
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        metrics = {}
        
        # Extract zero-shot accuracy
        zs_match = re.search(r'Zero-Shot accuracy on test: ([\d.]+)', content)
        if zs_match:
            metrics['zero_shot_accuracy'] = float(zs_match.group(1))
        
        # Extract final test accuracy
        test_acc_match = re.search(r'\* accuracy: ([\d.]+)%', content)
        if test_acc_match:
            metrics['test_accuracy'] = float(test_acc_match.group(1))

        # Extract final test macro-F1
        f1_match = re.search(r'\* macro_f1: ([\d.]+)%', content)
        if f1_match:
            metrics['test_macro_f1'] = float(f1_match.group(1))
        
        return metrics
        
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None

def collect_experiment_results(experiment_name: str) -> List[Dict]:
    """
    Collect all results from an experiment directory.
    
    Returns:
        List of dictionaries, each containing config and metrics for one run
    """
    results = []
    experiment_dir = Path("output") / experiment_name
    
    if not experiment_dir.exists():
        print(f"Experiment directory {experiment_dir} does not exist!")
        return results
    
    # Iterate through datasets
    for dataset_dir in experiment_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        print(f"Processing dataset: {dataset_name}")
        
        # Iterate through configs
        for config_dir in dataset_dir.iterdir():
            if not config_dir.is_dir():
                continue
                
            config_name = config_dir.name
            config_params = parse_config_from_filename(config_name)
            
            # Iterate through seeds
            for seed_dir in config_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith('seed'):
                    continue
                    
                seed_num = int(seed_dir.name.replace('seed', ''))
                log_path = seed_dir / 'log.txt'
                
                if log_path.exists():
                    metrics = parse_log_file(str(log_path))
                    if metrics:
                        result = {
                            'experiment': experiment_name,
                            'dataset': dataset_name,
                            'config_name': config_name,
                            'seed': seed_num,
                            **config_params,
                            **metrics
                        }
                        results.append(result)
                        #print(f"  ✓ Parsed {config_name}/seed{seed_num}")
                    else:
                        print(f"  ✗ Failed to parse {config_name}/seed{seed_num}")
                else:
                    print(f"  ✗ Missing log file: {log_path}")
    
    return results

def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """
    Aggregate results by averaging across seeds for each config.
    
    Returns:
        DataFrame with averaged metrics
    """
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Group by all config parameters and dataset, then average metrics
    group_cols = ['experiment', 'dataset', 'num_shots', 'optimizer', 'learning_rate', 
                'batch_size', 'epochs', 'init_type', 'constraint', 'nb_templates', 'gp']
    
    # Only include columns that exist in the dataframe
    group_cols = [col for col in group_cols if col in df.columns]
    
    # Metrics to average
    metric_cols = ['zero_shot_accuracy', 'test_accuracy', 'test_f1']
    metric_cols = [col for col in metric_cols if col in df.columns]
    
    # Aggregate
    agg_df = df.groupby(group_cols)[metric_cols].agg(['mean', 'std', 'count']).round(4)
    
    # Flatten column names
    agg_df.columns = [f"{col[0]}_{col[1]}" for col in agg_df.columns]
    agg_df = agg_df.reset_index()
    
    return agg_df

def create_method_name(row: pd.Series) -> str:
    """
    Create a method name for plotting.
    
    Format: {init_type}_{constraint}_{nb_templates}templates_{gp}GP
    """
    parts = []
    
    if pd.notna(row.get('init_type')):
        parts.append(str(row['init_type']))
    
    if pd.notna(row.get('constraint')):
        parts.append(str(row['constraint']))
    
    if pd.notna(row.get('nb_templates')):
        parts.append(f"{row['nb_templates']}templates")
    
    if pd.notna(row.get('gp')) and row['gp'] == True:
        parts.append("GP")
    
    return "_".join(parts)

def create_plots(df: pd.DataFrame, experiment_name: str, output_dir: Path):
    """
    Create line plots comparing methods across different shot numbers.
    For now, only plot test accuracy.
    """
    if df.empty:
        print("No data to plot!")
        return
    
    # Create method names
    df['method'] = df.apply(create_method_name, axis=1)
    
    # Get datasets
    datasets = df['dataset'].unique()
    
    # Create plots for each dataset
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset].copy()
        
        if dataset_df.empty:
            continue
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(dataset.upper(), fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('test_accuracy_mean', 'Test Accuracy (%)', axes)
        ]
        
        for metric, title, ax in metrics_to_plot:
            if metric not in dataset_df.columns:
                ax.set_title(f"{title} (No Data)")
                continue
                
            # Plot each method
            methods = dataset_df['method'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
            
            for method, color in zip(methods, colors):
                method_df = dataset_df[dataset_df['method'] == method].copy()
                method_df = method_df.sort_values('num_shots')
                
                if not method_df.empty:
                    ax.plot(method_df['num_shots'], method_df[metric], 
                           marker='o', label=method, color=color, linewidth=2, markersize=6)
                    
                    # Add error bars if std is available
                    std_col = metric.replace('_mean', '_std')
                    if std_col in method_df.columns:
                        ax.fill_between(method_df['num_shots'], 
                                      method_df[metric] - method_df[std_col],
                                      method_df[metric] + method_df[std_col],
                                      alpha=0.2, color=color)
            
            ax.set_xlabel('Number of Shots')
            ax.set_ylabel(title)
            if len(metrics_to_plot) > 1:
                ax.set_title(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to show all shot numbers
            if 'num_shots' in dataset_df.columns:
                shot_numbers = sorted(dataset_df['num_shots'].unique())
                ax.set_xticks(shot_numbers)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'{dataset}_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Parse experiment results and generate summaries')
    parser.add_argument('experiment_name', help='Name of the experiment directory')
    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    
    # Create output directory
    output_dir = Path('metrics') / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing experiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    
    # Collect results
    results = collect_experiment_results(experiment_name)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nFound {len(results)} individual results")
    
    # Aggregate results
    agg_df = aggregate_results(results)
    
    if agg_df.empty:
        print("No aggregated results!")
        return
    
    print(f"Aggregated to {len(agg_df)} unique configurations")
    
    # Save raw results CSV
    # raw_df = pd.DataFrame(results)
    # raw_csv_path = output_dir / f'raw_results.csv'
    # raw_df.to_csv(raw_csv_path, index=False)
    # print(f"Saved raw results: {raw_csv_path}")
    
    # Save aggregated results CSV
    agg_csv_path = output_dir / f'aggregated_results.csv'
    agg_df.to_csv(agg_csv_path, index=False)
    print(f"Saved aggregated results: {agg_csv_path}")
    
    # Create plots
    print("\nCreating plots...")
    create_plots(agg_df, experiment_name, output_dir)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    datasets = agg_df['dataset'].unique()
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        dataset_df = agg_df[agg_df['dataset'] == dataset]
        
        if 'test_accuracy_mean' in dataset_df.columns:
            best_acc = dataset_df['test_accuracy_mean'].max()
            best_row = dataset_df[dataset_df['test_accuracy_mean'] == best_acc].iloc[0]
            print(f"  Best final test accuracy: {best_acc:.2f}%")
            print(f"  Best method: {create_method_name(best_row)}")
            print(f"  Shots: {best_row.get('num_shots', 'N/A')}")


if __name__ == "__main__":
    main() 