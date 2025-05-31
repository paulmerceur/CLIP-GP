#!/usr/bin/env python3
"""
Analyze results from GP architecture hyperparameter tuning experiments.
Usage: python scripts/analyze_comprehensive_results.py
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

def extract_accuracy_from_log(log_path):
    """Extract final test accuracy from log file."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Look for final test accuracy in evaluation section
        eval_pattern = r'=> result\s*\n.*?accuracy:\s*([0-9]+\.?[0-9]*)%'
        eval_match = re.search(eval_pattern, content, re.MULTILINE | re.DOTALL)
        if eval_match:
            return float(eval_match.group(1))
        
        # Fallback: look for test accuracy patterns
        accuracy_patterns = [
            r'acc_test[:\s]+([0-9]+\.?[0-9]*)',
            r'Test accuracy[:\s]+([0-9]+\.?[0-9]*)',
            r'accuracy[:\s]+([0-9]+\.?[0-9]*)%',
        ]
        
        for pattern in accuracy_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return float(matches[-1])  # Return last (final) accuracy
        
        return None
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None

def extract_kl_divergence(log_path):
    """Extract final KL divergence from log file."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Look for KL divergence in training logs
        kl_pattern = r'kl_divergence\s+([0-9]+\.?[0-9]*)'
        matches = re.findall(kl_pattern, content)
        if matches:
            return float(matches[-1])  # Return final KL divergence
        
        return None
    except Exception as e:
        return None

def extract_training_time(log_path):
    """Extract training time from log file."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Look for elapsed time
        time_pattern = r'Elapsed:\s*([0-9:]+)'
        match = re.search(time_pattern, content)
        if match:
            time_str = match.group(1)
            # Convert to minutes
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
            elif len(parts) == 2:  # MM:SS
                return int(parts[0]) + int(parts[1]) / 60
        
        return None
    except Exception as e:
        return None

def get_config_details(config_name):
    """Return hyperparameter details for each configuration."""
    config_map = {
        'Baseline_No_GP': {'use_gp': False, 'kernel': 'none', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 0},
        'Conservative_RBF_LowLR': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.001, 'beta': 0.01, 'mc_samples': 3},
        'Moderate_RBF': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 3},
        'Aggressive_RBF': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.05, 'beta': 0.5, 'mc_samples': 3},
        'Linear_Kernel': {'use_gp': True, 'kernel': 'linear', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 3},
        'Short_Lengthscale': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 3},
        'Long_Lengthscale': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 3},
        'High_Output_Scale': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 3},
        'Low_Noise': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 3},
        'High_Noise': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 3},
        'More_MC_Samples': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 10},
        'Full_Covariance': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 3},
        'Low_KL_Weight': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 0.01, 'mc_samples': 3},
        'High_KL_Weight': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.01, 'beta': 1.0, 'mc_samples': 3},
        'Optimized_RBF': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.02, 'beta': 0.2, 'mc_samples': 5},
        'Optimized_Linear': {'use_gp': True, 'kernel': 'linear', 'lr': 0.02, 'beta': 0.15, 'mc_samples': 5},
        'Conservative_Full_Cov': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.005, 'beta': 0.05, 'mc_samples': 3},
        'High_Variance': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.03, 'beta': 0.3, 'mc_samples': 7},
        'Minimal_GP': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.001, 'beta': 0.005, 'mc_samples': 1},
        'Ultra_GP': {'use_gp': True, 'kernel': 'rbf', 'lr': 0.1, 'beta': 2.0, 'mc_samples': 15},
    }
    return config_map.get(config_name, {'use_gp': True, 'kernel': 'unknown', 'lr': 0.01, 'beta': 0.1, 'mc_samples': 3})

def main():
    print("=== GP Architecture Hyperparameter Tuning Results Analysis ===")
    print()
    
    # Define experiment structure - updated for new architecture
    datasets = ['caltech101', 'oxford_flowers', 'food101']
    shots = [4, 16]
    configs = [
        'Baseline_No_GP',
        'Conservative_RBF_LowLR', 
        'Moderate_RBF',
        'Aggressive_RBF',
        'Linear_Kernel',
        'Short_Lengthscale',
        'Long_Lengthscale',
        'High_Output_Scale',
        'Low_Noise',
        'High_Noise',
        'More_MC_Samples',
        'Full_Covariance',
        'Low_KL_Weight',
        'High_KL_Weight',
        'Optimized_RBF',
        'Optimized_Linear',
        'Conservative_Full_Cov',
        'High_Variance',
        'Minimal_GP',
        'Ultra_GP'
    ]
    
    # Find log directory
    log_dir = Path('logs/gp_architecture_tune')
    if not log_dir.exists():
        print(f"Log directory {log_dir} not found!")
        return
    
    # Find the latest job logs
    log_files = list(log_dir.glob('gp_architecture_tune_*.out'))
    if not log_files:
        print("No log files found!")
        return
    
    # Extract job ID from log files
    job_ids = set()
    for log_file in log_files:
        match = re.search(r'gp_architecture_tune_(\d+)_\d+\.out', log_file.name)
        if match:
            job_ids.add(match.group(1))
    
    if not job_ids:
        print("Could not extract job IDs from log files!")
        return
    
    # Use the latest job ID
    latest_job = max(job_ids)
    print(f"Analyzing results from job {latest_job}")
    print()
    
    # Collect results
    results = []
    missing_count = 0
    
    for dataset_idx, dataset in enumerate(datasets, 1):
        for shot_idx, shot in enumerate(shots, 1):
            for config_idx, config in enumerate(configs, 1):
                # Calculate array task ID - updated for new structure
                # 3 datasets × 2 shots × 20 configs = 120 total
                array_id = (dataset_idx - 1) * 40 + (shot_idx - 1) * 20 + config_idx
                
                # Find log file
                log_file = log_dir / f'gp_architecture_tune_{latest_job}_{array_id}.out'
                
                if log_file.exists():
                    accuracy = extract_accuracy_from_log(log_file)
                    kl_div = extract_kl_divergence(log_file)
                    train_time = extract_training_time(log_file)
                    
                    if accuracy is not None:
                        config_details = get_config_details(config)
                        result = {
                            'dataset': dataset,
                            'shots': shot,
                            'config': config,
                            'accuracy': accuracy,
                            'kl_divergence': kl_div,
                            'train_time_min': train_time,
                            'is_gp': config != 'Baseline_No_GP',
                            'array_id': array_id,
                            **config_details
                        }
                        results.append(result)
                    else:
                        missing_count += 1
                        print(f"Warning: Could not extract accuracy from {log_file}")
                else:
                    missing_count += 1
                    print(f"Warning: Missing log file {log_file}")
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} results, {missing_count} missing/failed")
    print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Summary by dataset and shots
    print("=== Results Summary by Dataset ===")
    print()
    
    for dataset in datasets:
        print(f"Dataset: {dataset.upper()}")
        print("-" * 60)
        
        dataset_df = df[df['dataset'] == dataset]
        
        if dataset_df.empty:
            print("No results found for this dataset")
            print()
            continue
        
        # Create pivot table
        pivot = dataset_df.pivot(index='config', columns='shots', values='accuracy')
        print(pivot.round(2))
        print()
        
        # Find best config per shot setting
        for shot in shots:
            shot_data = dataset_df[dataset_df['shots'] == shot]
            if not shot_data.empty:
                baseline = shot_data[shot_data['config'] == 'Baseline_No_GP']['accuracy']
                gp_results = shot_data[shot_data['config'] != 'Baseline_No_GP']
                
                if not baseline.empty and not gp_results.empty:
                    baseline_acc = baseline.iloc[0]
                    best_gp = gp_results.loc[gp_results['accuracy'].idxmax()]
                    improvement = best_gp['accuracy'] - baseline_acc
                    
                    print(f"  {shot}-shot: Baseline {baseline_acc:.2f}% | "
                          f"Best GP ({best_gp['config']}) {best_gp['accuracy']:.2f}% | "
                          f"Improvement: {improvement:+.2f}%")
        print()
    
    # GP Architecture Analysis
    print("=== GP Architecture Analysis ===")
    print()
    
    gp_df = df[df['is_gp'] == True].copy()
    baseline_df = df[df['is_gp'] == False].copy()
    
    if not gp_df.empty:
        # Kernel comparison
        print("Performance by Kernel Type:")
        kernel_perf = gp_df.groupby('kernel')['accuracy'].agg(['mean', 'std', 'count']).round(2)
        print(kernel_perf)
        print()
        
        # Learning rate analysis
        print("Performance by Learning Rate:")
        lr_perf = gp_df.groupby('lr')['accuracy'].agg(['mean', 'std', 'count']).round(2)
        print(lr_perf)
        print()
        
        # KL weight analysis
        print("Performance by KL Weight (β):")
        beta_perf = gp_df.groupby('beta')['accuracy'].agg(['mean', 'std', 'count']).round(2)
        print(beta_perf)
        print()
        
        # MC samples analysis
        print("Performance by MC Samples:")
        mc_perf = gp_df.groupby('mc_samples')['accuracy'].agg(['mean', 'std', 'count']).round(2)
        print(mc_perf)
        print()
        
        # Top 5 configurations
        print("Top 5 GP Configurations:")
        top_configs = gp_df.nlargest(5, 'accuracy')[['config', 'dataset', 'shots', 'accuracy', 'kernel', 'lr', 'beta']]
        print(top_configs.to_string(index=False))
        print()
    
    # Overall statistics
    print("=== Overall Statistics ===")
    
    if not baseline_df.empty and not gp_df.empty:
        print(f"Baseline average: {baseline_df['accuracy'].mean():.2f}% (±{baseline_df['accuracy'].std():.2f}%)")
        print(f"GP average: {gp_df['accuracy'].mean():.2f}% (±{gp_df['accuracy'].std():.2f}%)")
        print(f"Best GP config overall: {gp_df.loc[gp_df['accuracy'].idxmax(), 'config']}")
        print(f"Best GP accuracy: {gp_df['accuracy'].max():.2f}%")
        
        # Count improvements
        improvements = 0
        total_comparisons = 0
        improvement_details = []
        
        for dataset in datasets:
            for shot in shots:
                baseline_acc = baseline_df[(baseline_df['dataset'] == dataset) & 
                                         (baseline_df['shots'] == shot)]['accuracy']
                gp_accs = gp_df[(gp_df['dataset'] == dataset) & 
                               (gp_df['shots'] == shot)]['accuracy']
                
                if not baseline_acc.empty and not gp_accs.empty:
                    total_comparisons += 1
                    max_gp_acc = gp_accs.max()
                    improvement = max_gp_acc - baseline_acc.iloc[0]
                    improvement_details.append({
                        'dataset': dataset,
                        'shots': shot,
                        'baseline': baseline_acc.iloc[0],
                        'best_gp': max_gp_acc,
                        'improvement': improvement
                    })
                    if max_gp_acc > baseline_acc.iloc[0]:
                        improvements += 1
        
        print(f"GP beats baseline in {improvements}/{total_comparisons} cases ({100*improvements/total_comparisons:.1f}%)")
        
        # Average improvement
        avg_improvement = np.mean([d['improvement'] for d in improvement_details])
        print(f"Average improvement: {avg_improvement:+.2f}%")
        print()
        
        # Training time analysis
        if df['train_time_min'].notna().any():
            print("=== Training Time Analysis ===")
            baseline_time = baseline_df['train_time_min'].mean()
            gp_time = gp_df['train_time_min'].mean()
            
            if not pd.isna(baseline_time) and not pd.isna(gp_time):
                print(f"Baseline avg time: {baseline_time:.1f} min")
                print(f"GP avg time: {gp_time:.1f} min")
                print(f"GP time overhead: {(gp_time/baseline_time - 1)*100:.1f}%")
            print()
    
    # Save detailed results
    output_file = 'gp_architecture_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")
    
    # Save summary statistics
    summary_stats = []
    for dataset in datasets:
        for shot in shots:
            dataset_shot_df = df[(df['dataset'] == dataset) & (df['shots'] == shot)]
            baseline_row = dataset_shot_df[dataset_shot_df['is_gp'] == False]
            gp_rows = dataset_shot_df[dataset_shot_df['is_gp'] == True]
            
            if not baseline_row.empty and not gp_rows.empty:
                baseline_acc = baseline_row['accuracy'].iloc[0]
                best_gp = gp_rows.nlargest(1, 'accuracy')
                
                summary_stats.append({
                    'dataset': dataset,
                    'shots': shot,
                    'baseline_accuracy': baseline_acc,
                    'best_gp_config': best_gp.iloc[0]['config'],
                    'best_gp_accuracy': best_gp.iloc[0]['accuracy'],
                    'improvement': best_gp.iloc[0]['accuracy'] - baseline_acc
                })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('gp_architecture_summary.csv', index=False)
    print(f"Summary statistics saved to gp_architecture_summary.csv")

if __name__ == "__main__":
    main() 