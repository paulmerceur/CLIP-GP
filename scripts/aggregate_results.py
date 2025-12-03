#!/usr/bin/env python3
"""
Aggregate metrics.json files under output/<experiment>/ and print quick summaries.
Also generate plots (Accuracy, ECE, AECE vs shots) per dataset in _plots/perf_per_shots/,
Accuracy vs ECE plots per dataset in _plots/acc_vs_ece/, plus average plots across datasets,
and write experiment-specific tables under _tables/.

Usage:
  python scripts/aggregate_results.py <experiment_name>

This expects a layout like:
  output/<experiment>/<dataset>/<config>/seed*/metrics.json

Where metrics.json minimally contains:
  {
    "dataset": ..., "shots": ..., "seed": ..., "method": "baseline"|"gp",
    "backbone": ..., "metrics": {"top1_acc": ..., "ece": ..., "aece": ...},
    "config": { ... full config ... }
  }
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import statistics
import math
import matplotlib.pyplot as plt

# Edit this dict to control how configs are grouped when using --grouped.
# Keys: substrings to match against the normalized config name
#       (i.e., the config directory name with the "_<shots>shots" suffix removed).
# Values: labels to show on plots for the matched group.
# The order matters: a config will be assigned to the first matching substring.
GROUP_SUBSTRINGS: Dict[str, str] = {
    "_1template": "1 Template",
    "_8templates": "8 Templates",
    "_88templates": "88 Templates",
    "_custom_templates": "Custom Templates",
}

def _collect_zero_shot_values(runs: List[Dict[str, Any]], key: str) -> List[float]:
    """
    Collect a list of finite float values for the given zero-shot metric key from a list of runs.
    Safely skips runs missing zero-shot data or with non-numeric/NaN values.
    """
    values: List[float] = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        zs = r.get("zero_shot")
        if not isinstance(zs, dict):
            continue
        v = zs.get(key, None)
        try:
            v_f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(v_f):
            values.append(v_f)
    return values


def load_runs(exp_dir: Path, delete: bool = False) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for dataset_dir in sorted(d for d in exp_dir.iterdir() if d.is_dir()):
        for config_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
            for seed_dir in sorted(d for d in config_dir.glob("seed*")):
                metrics_path = seed_dir / "metrics.json"
                if not metrics_path.is_file():
                    if delete:
                        import shutil
                        shutil.rmtree(seed_dir)
                    else:
                        print(f"Skipping {seed_dir} because it doesn't exist")
                    continue
                try:
                    payload = json.loads(metrics_path.read_text())
                    payload["_dataset_dir"] = dataset_dir.name
                    payload["_config_label"] = config_dir.name
                    payload["_seed_dir"] = seed_dir.name
                    runs.append(payload)
                except Exception:
                    pass
    return runs


def group_by_dataset_shots_config(runs: List[Dict[str, Any]]):
    grouped: Dict[str, Dict[int, Dict[str, List[Dict[str, Any]]]]] = {}
    for r in runs:
        ds = r.get("dataset") or r.get("_dataset_dir")
        shots = int(r.get("shots", 0))
        cfg = r.get("_config_label", "config")
        grouped.setdefault(ds, {}).setdefault(shots, {}).setdefault(cfg, []).append(r)
    return grouped


def print_summary(grouped):
    for ds, shots_map in grouped.items():
        print(f"\n=== Dataset: {ds} ===")
        # Determine the max config label length for column width
        max_cfg_len = max((len(cfg) for cfg_map in shots_map.values() for cfg in cfg_map), default=6)
        print(f"{'Config':<{max_cfg_len}} {'Shots':>5} {'Seeds':>5} | {'Acc µ':>7} {'Acc σ':>7} | {'ECE µ':>7} {'ECE σ':>7} | {'AECE µ':>7} {'AECE σ':>7}")
        print("-" * (max_cfg_len + 66))
        rows = []
        # Collect standard per-shot rows
        for shots, cfg_map in sorted(shots_map.items()):
            for cfg, rs in sorted(cfg_map.items()):
                accs = [float(r["metrics"].get("accuracy", float('nan'))) for r in rs]
                if math.isnan(accs[0]):
                    accs = [float(r["metrics"].get("top1_acc", float('nan'))) for r in rs]
                eces = [float(r["metrics"].get("ece", float('nan'))) for r in rs]
                aeces = [float(r["metrics"].get("aece", float('nan'))) for r in rs]
                n = len(rs)
                try:
                    acc_mean = statistics.fmean(accs) if accs else float('nan')
                    acc_std = statistics.pstdev(accs) if n > 1 else 0.0
                    ece_mean = statistics.fmean(eces) if eces else float('nan')
                    ece_std = statistics.pstdev(eces) if n > 1 else 0.0
                    aece_mean = statistics.fmean(aeces) if aeces else float('nan')
                    aece_std = statistics.pstdev(aeces) if n > 1 else 0.0
                except Exception:
                    print(f"Error calculating statistics for {cfg} {shots}: {accs}")
                rows.append((cfg, shots, n, acc_mean, acc_std, ece_mean, ece_std, aece_mean, aece_std))
        # Add zero-shot rows: average zero_shot metrics over seeds from 1-shot runs
        if 1 in shots_map:
            for cfg_1shot, rs in sorted(shots_map[1].items()):
                zs_accs = _collect_zero_shot_values(rs, "top1_acc")
                zs_eces = _collect_zero_shot_values(rs, "ece")
                zs_aeces = _collect_zero_shot_values(rs, "aece")
                n = len(rs)
                try:
                    acc_mean = statistics.fmean(zs_accs) if zs_accs else float('nan')
                    acc_std = statistics.pstdev(zs_accs) if len(zs_accs) > 1 else 0.0
                    ece_mean = statistics.fmean(zs_eces) if zs_eces else float('nan')
                    ece_std = statistics.pstdev(zs_eces) if len(zs_eces) > 1 else 0.0
                    aece_mean = statistics.fmean(zs_aeces) if zs_aeces else float('nan')
                    aece_std = statistics.pstdev(zs_aeces) if len(zs_aeces) > 1 else 0.0
                except Exception:
                    print(f"Error calculating zero-shot statistics for {cfg_1shot}: {zs_accs}")
                # Replace the "_1shots" suffix with "_0shots" for display
                cfg_0shot = cfg_1shot.replace("_1shots", "_0shots")
                rows.append((cfg_0shot, 0, n, acc_mean, acc_std, ece_mean, ece_std, aece_mean, aece_std))
        rows.sort(key=lambda x: (x[1], x[0]))
        for cfg, shots, n, acc_m, acc_s, ece_m, ece_s, aece_m, aece_s in rows:
            print(f"{cfg:<{max_cfg_len}} {shots:>5d} {n:>5d} | {acc_m:7.2f} {acc_s:7.2f} | {ece_m:7.3f} {ece_s:7.3f} | {aece_m:7.3f} {aece_s:7.3f}")


def print_average_summary(grouped):
    """
    Print a table averaging metrics over datasets, per config and shots, including 0-shot.
    Averaging is done across datasets on top of per-dataset seed means.
    """
    # Collect: fam -> shots -> metric -> list[per-dataset mean]
    all_per_cfg_data: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    # Build per-dataset per-fam per-shot means
    for ds, shots_map in grouped.items():
        per_cfg: Dict[str, Dict[int, Dict[str, float]]] = {}
        for shots, cfg_map in shots_map.items():
            for cfg, rs in cfg_map.items():
                accs = [float(r["metrics"].get("accuracy", float('nan'))) for r in rs]
                if accs and math.isnan(accs[0]):
                    accs = [float(r["metrics"].get("top1_acc", float('nan'))) for r in rs]
                eces = [float(r["metrics"].get("ece", float('nan'))) for r in rs]
                aeces = [float(r["metrics"].get("aece", float('nan'))) for r in rs]
                fam = cfg.replace(f"_{shots}shots", "")
                per_cfg.setdefault(fam, {})[shots] = {
                    "acc": statistics.fmean(accs) if accs else float('nan'),
                    "ece": statistics.fmean(eces) if eces else float('nan'),
                    "aece": statistics.fmean(aeces) if aeces else float('nan'),
                }
        # Add zero-shot per fam from 1-shot runs
        if 1 in shots_map:
            for cfg_1shot, rs in shots_map[1].items():
                fam = cfg_1shot.replace("_1shots", "")
                zs_accs = _collect_zero_shot_values(rs, "top1_acc")
                zs_eces = _collect_zero_shot_values(rs, "ece")
                zs_aeces = _collect_zero_shot_values(rs, "aece")
                per_cfg.setdefault(fam, {})[0] = {
                    "acc": statistics.fmean(zs_accs) if zs_accs else float('nan'),
                    "ece": statistics.fmean(zs_eces) if zs_eces else float('nan'),
                    "aece": statistics.fmean(zs_aeces) if zs_aeces else float('nan'),
                }
        # Accumulate into global lists
        for fam, shot_map in per_cfg.items():
            for shots, metrics in shot_map.items():
                fam_map = all_per_cfg_data.setdefault(fam, {})
                shot_lists = fam_map.setdefault(shots, {"acc": [], "ece": [], "aece": []})
                for metric in ["acc", "ece", "aece"]:
                    if not math.isnan(metrics[metric]):
                        shot_lists[metric].append(metrics[metric])

    # Prepare rows
    rows = []
    cfg_labels = []
    for fam, shot_map in all_per_cfg_data.items():
        for shots, metric_lists in shot_map.items():
            values_acc = metric_lists["acc"]
            values_ece = metric_lists["ece"]
            values_aece = metric_lists["aece"]
            n_ds = max(len(values_acc), len(values_ece), len(values_aece))
            acc_mean = statistics.fmean(values_acc) if values_acc else float('nan')
            acc_std = statistics.pstdev(values_acc) if len(values_acc) > 1 else 0.0
            ece_mean = statistics.fmean(values_ece) if values_ece else float('nan')
            ece_std = statistics.pstdev(values_ece) if len(values_ece) > 1 else 0.0
            aece_mean = statistics.fmean(values_aece) if values_aece else float('nan')
            aece_std = statistics.pstdev(values_aece) if len(values_aece) > 1 else 0.0
            cfg_label = f"{fam}_{shots}shots"
            cfg_labels.append(cfg_label)
            rows.append((cfg_label, shots, n_ds, acc_mean, acc_std, ece_mean, ece_std, aece_mean, aece_std))

    if not rows:
        return

    # Formatting
    max_cfg_len = max((len(lbl) for lbl in cfg_labels), default=6)
    print(f"\n=== Average across datasets ({len(grouped)} datasets) ===")
    print(f"{'Config':<{max_cfg_len}} {'Shots':>5} {'Datasets':>9} | {'Acc µ':>7} {'Acc σ':>7} | {'ECE µ':>7} {'ECE σ':>7} | {'AECE µ':>7} {'AECE σ':>7}")
    print("-" * (max_cfg_len + 70))
    rows.sort(key=lambda x: (x[1], x[0]))
    for cfg, shots, n_ds, acc_m, acc_s, ece_m, ece_s, aece_m, aece_s in rows:
        print(f"{cfg:<{max_cfg_len}} {shots:>5d} {n_ds:>9d} | {acc_m:7.2f} {acc_s:7.2f} | {ece_m:7.3f} {ece_s:7.3f} | {aece_m:7.3f} {aece_s:7.3f}")


def _group_per_cfg(per_cfg: Dict[str, Dict[int, Dict[str, float]]]) -> tuple[Dict[str, Dict[int, Dict[str, float]]], Dict[str, Dict[int, Dict[str, float]]], Dict[str, set]]:
    """
    Group per-config metrics by substring. Returns:
      - per_group_for_plot: keys include counts e.g. "<label> (N)"
      - per_group_for_collect: keys are raw "<label>" (stable across datasets)
      - group_to_fams: mapping from "<label>" to the set of matched fam names
    """
    assigned = set()  # fams already grouped
    per_group_for_plot: Dict[str, Dict[int, Dict[str, float]]] = {}
    per_group_for_collect: Dict[str, Dict[int, Dict[str, float]]] = {}
    group_to_fams: Dict[str, set] = {}
    substrings = GROUP_SUBSTRINGS
    for sub, label in substrings.items():
        matched = [fam for fam in per_cfg.keys() if fam not in assigned and sub in fam]
        if not matched:
            continue
        # Aggregate across matched configs per shot
        shots_all = sorted({s for fam in matched for s in per_cfg[fam].keys()})
        shot_map: Dict[int, Dict[str, float]] = {}
        for s in shots_all:
            vals_acc = [per_cfg[fam][s]["acc"] for fam in matched if s in per_cfg[fam] and not math.isnan(per_cfg[fam][s]["acc"])]
            vals_ece = [per_cfg[fam][s]["ece"] for fam in matched if s in per_cfg[fam] and not math.isnan(per_cfg[fam][s]["ece"])]
            vals_aece = [per_cfg[fam][s]["aece"] for fam in matched if s in per_cfg[fam] and not math.isnan(per_cfg[fam][s]["aece"])]
            shot_map[s] = {
                "acc": statistics.fmean(vals_acc) if vals_acc else float('nan'),
                "ece": statistics.fmean(vals_ece) if vals_ece else float('nan'),
                "aece": statistics.fmean(vals_aece) if vals_aece else float('nan'),
            }
        label_with_count = f"{label} ({len(matched)})"
        per_group_for_plot[label_with_count] = shot_map
        per_group_for_collect[label] = shot_map
        group_to_fams.setdefault(label, set()).update(matched)
        assigned.update(matched)
    return per_group_for_plot, per_group_for_collect, group_to_fams


def make_plots(grouped, exp_name: str, use_grouping: bool = False, show_zero_shot: bool = False):
    plots_dir = Path("output") / exp_name / "_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    perf_per_shots_dir = plots_dir / "perf_per_shots"
    perf_per_shots_dir.mkdir(parents=True, exist_ok=True)

    acc_vs_ece_dir = plots_dir / "acc_vs_ece"
    acc_vs_ece_dir.mkdir(parents=True, exist_ok=True)

    # Collect data for averaging across datasets
    all_per_cfg_data = {}
    all_shots_set = set()
    global_group_to_fams: Dict[str, set] = {}

    for ds, shots_map in grouped.items():
        # Build per-config shot->mean maps
        per_cfg: Dict[str, Dict[int, Dict[str, float]]] = {}
        all_shots: List[int] = sorted(shots_map.keys())
        if show_zero_shot:
            all_shots_set.update([0, *all_shots])
        else:
            all_shots_set.update(all_shots)

        for shots, cfg_map in shots_map.items():
            for cfg, rs in cfg_map.items():
                accs = [float(r["metrics"].get("accuracy", float('nan'))) for r in rs]
                if math.isnan(accs[0]):
                    accs = [float(r["metrics"].get("top1_acc", float('nan'))) for r in rs]
                eces = [float(r["metrics"].get("ece", float('nan'))) for r in rs]
                aeces = [float(r["metrics"].get("aece", float('nan'))) for r in rs]
                # Normalize config label across shots to draw a single line
                fam = cfg.replace(f"_{shots}shots", "")
                per_cfg.setdefault(fam, {})[shots] = {
                    "acc": statistics.fmean(accs) if accs else float('nan'),
                    "ece": statistics.fmean(eces) if eces else float('nan'),
                    "aece": statistics.fmean(aeces) if aeces else float('nan'),
                }

        # Compute zero-shot metrics per fam from 1-shot runs (seed-averaged) for optional stars
        zero_shot_as_per_cfg: Dict[str, Dict[int, Dict[str, float]]] = {}
        if show_zero_shot and 1 in shots_map:
            for cfg_1shot, rs in shots_map[1].items():
                fam = cfg_1shot.replace("_1shots", "")
                zs_accs = _collect_zero_shot_values(rs, "top1_acc")
                zs_eces = _collect_zero_shot_values(rs, "ece")
                zs_aeces = _collect_zero_shot_values(rs, "aece")
                zero_shot_as_per_cfg.setdefault(fam, {})[0] = {
                    "acc": statistics.fmean(zs_accs) if zs_accs else float('nan'),
                    "ece": statistics.fmean(zs_eces) if zs_eces else float('nan'),
                    "aece": statistics.fmean(zs_aeces) if zs_aeces else float('nan'),
                }

        # Optionally regroup multiple configs into single lines by substring
        if use_grouping:
            per_group_for_plot, per_group_for_collect, group_to_fams = _group_per_cfg(per_cfg)
            per_cfg_for_plots = per_group_for_plot
            per_cfg_for_collect = per_group_for_collect
            for g, fams in group_to_fams.items():
                global_group_to_fams.setdefault(g, set()).update(fams)
            # Group zero-shot as well
            if show_zero_shot and zero_shot_as_per_cfg:
                zs_group_plot, zs_group_collect, _ = _group_per_cfg(zero_shot_as_per_cfg)
                # Extract shot=0 metric dicts
                zero_shot_for_plots = {label: shot_map.get(0, {"acc": float('nan'), "ece": float('nan'), "aece": float('nan')})
                                       for label, shot_map in zs_group_plot.items()}
                zero_shot_for_collect = {label: shot_map.get(0, {"acc": float('nan'), "ece": float('nan'), "aece": float('nan')})
                                         for label, shot_map in zs_group_collect.items()}
            else:
                zero_shot_for_plots = {}
                zero_shot_for_collect = {}
        else:
            per_cfg_for_plots = per_cfg
            per_cfg_for_collect = per_cfg
            zero_shot_for_plots = {fam: metrics_map[0] for fam, metrics_map in zero_shot_as_per_cfg.items()} if zero_shot_as_per_cfg else {}
            zero_shot_for_collect = zero_shot_for_plots

        # Performance per shots plots (original plots)
        make_perf_per_shots_plots(ds, per_cfg_for_plots, all_shots, perf_per_shots_dir, show_zero_shot=show_zero_shot, zero_shot_per_cfg=zero_shot_for_plots)

        # Accuracy vs ECE plots (new plots)
        make_acc_vs_ece_plots(ds, per_cfg_for_plots, all_shots, acc_vs_ece_dir, show_zero_shot=show_zero_shot, zero_shot_per_cfg=zero_shot_for_plots)

        # Collect data for averaging
        for cfg, shot_map in per_cfg_for_collect.items():
            if cfg not in all_per_cfg_data:
                all_per_cfg_data[cfg] = {}
            for shots, metrics in shot_map.items():
                if shots not in all_per_cfg_data[cfg]:
                    all_per_cfg_data[cfg][shots] = {"acc": [], "ece": [], "aece": []}
                for metric in ["acc", "ece", "aece"]:
                    if not math.isnan(metrics[metric]):
                        all_per_cfg_data[cfg][shots][metric].append(metrics[metric])
        # Collect zero-shot for averaging
        if show_zero_shot and zero_shot_for_collect:
            for cfg, metrics in zero_shot_for_collect.items():
                if 0 not in all_per_cfg_data.setdefault(cfg, {}):
                    all_per_cfg_data[cfg][0] = {"acc": [], "ece": [], "aece": []}
                for metric in ["acc", "ece", "aece"]:
                    if not math.isnan(metrics[metric]):
                        all_per_cfg_data[cfg][0][metric].append(metrics[metric])

    # Create average plots
    num_datasets = len(grouped)
    all_shots = sorted(all_shots_set)

    # Average the metrics across datasets
    avg_per_cfg = {}
    for cfg, shot_map in all_per_cfg_data.items():
        avg_per_cfg[cfg] = {}
        for shots, metrics_lists in shot_map.items():
            avg_per_cfg[cfg][shots] = {}
            for metric in ["acc", "ece", "aece"]:
                values = metrics_lists[metric]
                avg_per_cfg[cfg][shots][metric] = statistics.fmean(values) if values else float('nan')

    # If grouping, relabel average lines to include counts: "<group> (N configs)"
    if use_grouping and global_group_to_fams:
        avg_labeled: Dict[str, Dict[int, Dict[str, float]]] = {}
        for clean_group, shot_map in avg_per_cfg.items():
            n_cfg = len(global_group_to_fams.get(clean_group, set()))
            label = f"{clean_group} ({n_cfg} configs)"
            avg_labeled[label] = shot_map
        avg_for_plots = avg_labeled
    else:
        avg_for_plots = avg_per_cfg

    # Create average plots
    # Prepare zero-shot for average plots if requested
    if show_zero_shot:
        # Build zero-shot map per cfg label as used in avg_for_plots
        avg_zero_shot_map_clean = {cfg: shot_map.get(0, {"acc": float('nan'), "ece": float('nan'), "aece": float('nan')})
                                   for cfg, shot_map in avg_per_cfg.items()}
        if use_grouping and isinstance(avg_for_plots, dict):
            # avg_for_plots keys may include " (N configs)"; map back to clean key by splitting
            avg_zero_shot_for_plots = {}
            for label, shot_map in avg_for_plots.items():
                clean = label.split(" (")[0]
                avg_zero_shot_for_plots[label] = avg_zero_shot_map_clean.get(clean, {"acc": float('nan'), "ece": float('nan'), "aece": float('nan')})
        else:
            avg_zero_shot_for_plots = {cfg: avg_zero_shot_map_clean.get(cfg, {"acc": float('nan'), "ece": float('nan'), "aece": float('nan')})
                                       for cfg in avg_for_plots.keys()}
    else:
        avg_zero_shot_for_plots = {}

    make_perf_per_shots_plots(f"Average ({num_datasets} datasets)", avg_for_plots, all_shots, perf_per_shots_dir, show_zero_shot=show_zero_shot, zero_shot_per_cfg=avg_zero_shot_for_plots)
    make_acc_vs_ece_plots(f"Average ({num_datasets} datasets)", avg_for_plots, all_shots, acc_vs_ece_dir, show_zero_shot=show_zero_shot, zero_shot_per_cfg=avg_zero_shot_for_plots)


def make_perf_per_shots_plots(ds, per_cfg, all_shots, plots_dir, show_zero_shot: bool = False, zero_shot_per_cfg: Dict[str, Dict[str, float]] | None = None):
    num_cfg = len(per_cfg)
    fig_w = max(16, min(48, 16 + 0.4 * max(0, num_cfg - 8)))
    fig_h = 6  # normal height for plots

    # Create a gridspec with extra row for legend
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(fig_w, fig_h + 2))  # +2 for legend space
    gs = gridspec.GridSpec(2, 3, height_ratios=[fig_h, 2])
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_ece = fig.add_subplot(gs[0, 1])
    ax_aece = fig.add_subplot(gs[0, 2])

    color_acc = {}
    color_ece = {}
    color_aece = {}

    for cfg, shot_map in per_cfg.items():
        # Explicitly exclude 0 from line data to keep zero-shot stars disconnected
        xs = [s for s in all_shots if s in shot_map and s != 0]
        if not xs:
            continue
        ys_acc = [shot_map[s]["acc"] for s in xs]
        ys_ece = [shot_map[s]["ece"] for s in xs]
        ys_aece = [shot_map[s]["aece"] for s in xs]
        line_acc = ax_acc.plot(xs, ys_acc, marker="o", linestyle="-", label=cfg)[0]
        line_ece = ax_ece.plot(xs, ys_ece, marker="o", linestyle="-", label=cfg)[0]
        line_aece = ax_aece.plot(xs, ys_aece, marker="o", linestyle="-", label=cfg)[0]
        color_acc[cfg] = line_acc.get_color()
        color_ece[cfg] = line_ece.get_color()
        color_aece[cfg] = line_aece.get_color()

    for ax, title, ylabel in [
        (ax_acc, "Accuracy", "Accuracy (%)"),
        (ax_ece, "ECE", "ECE"),
        (ax_aece, "AECE", "AECE"),
    ]:
        ax.set_xlabel("# Shots")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ticks = sorted(set(all_shots) | ({0} if show_zero_shot else set()))
        ax.set_xticks(ticks)

    # Zero-shot stars (disconnected), same color as corresponding config
    if show_zero_shot and zero_shot_per_cfg:
        for cfg, zs in zero_shot_per_cfg.items():
            if "acc" in zs and not math.isnan(zs["acc"]) and cfg in color_acc:
                ax_acc.scatter(0, zs["acc"], marker="*", s=140, color=color_acc[cfg], zorder=10, edgecolors="none")
            if "ece" in zs and not math.isnan(zs["ece"]) and cfg in color_ece:
                ax_ece.scatter(0, zs["ece"], marker="*", s=140, color=color_ece[cfg], zorder=10, edgecolors="none")
            if "aece" in zs and not math.isnan(zs["aece"]) and cfg in color_aece:
                ax_aece.scatter(0, zs["aece"], marker="*", s=140, color=color_aece[cfg], zorder=10, edgecolors="none")

    handles, labels = ax_acc.get_legend_handles_labels()
    if labels:
        # Add legend to the bottom row spanning all columns
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis('off')
        legend_ax.legend(handles, labels, loc='center', ncol=2)
    fig.suptitle(ds)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # Use "average.png" for average plots, otherwise use dataset name
    filename = "average.png" if ds.startswith("Average") else f"{ds}_metrics.png"
    fig.savefig(plots_dir / filename)
    plt.close(fig)


def make_acc_vs_ece_plots(ds, per_cfg, all_shots, plots_dir, show_zero_shot: bool = False, zero_shot_per_cfg: Dict[str, Dict[str, float]] | None = None):
    fig, ax = plt.subplots(figsize=(10, 8))

    cfg_to_color = {}

    for cfg, shot_map in per_cfg.items():
        # Collect points for this config
        points = []
        for shots in all_shots:
            # Explicitly exclude 0 from the connected line
            if shots > 0 and shots in shot_map:
                acc = shot_map[shots]["acc"]
                ece = shot_map[shots]["ece"]
                if not (math.isnan(acc) or math.isnan(ece)):
                    points.append((ece, acc))

        if not points:
            continue

        # Sort points by ECE for cleaner line
        points.sort(key=lambda p: p[0])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        line = ax.plot(xs, ys, marker="o", linestyle="-", label=cfg)[0]
        cfg_to_color[cfg] = line.get_color()

        # Add star at average position
        avg_ece = statistics.fmean(xs) if xs else float('nan')
        avg_acc = statistics.fmean(ys) if ys else float('nan')
        if not (math.isnan(avg_ece) or math.isnan(avg_acc)):
            ax.scatter(avg_ece, avg_acc, marker="*", s=200, color=ax.get_lines()[-1].get_color(), zorder=10)

        # Optional zero-shot star (disconnected)
        if show_zero_shot and zero_shot_per_cfg and cfg in zero_shot_per_cfg:
            zs = zero_shot_per_cfg[cfg]
            zs_ece = zs.get("ece", float('nan'))
            zs_acc = zs.get("acc", float('nan'))
            if not (math.isnan(zs_ece) or math.isnan(zs_acc)):
                ax.scatter(zs_ece, zs_acc, marker="*", s=160, color=cfg_to_color.get(cfg, None), zorder=11)

    ax.set_xlabel("ECE")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{ds}: Accuracy vs ECE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Use "average.png" for average plots, otherwise use dataset name
    filename = "average.png" if ds.startswith("Average") else f"{ds}_acc_vs_ece.png"
    fig.savefig(plots_dir / filename)
    plt.close(fig)


def _compute_per_dataset_fam_shot_means(grouped) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """
    Build per-dataset -> family(config label sans _{shots}shots) -> shot -> metric means over seeds.
    Returns:
      { dataset: { fam: { shot: { "acc": float, "ece": float, "aece": float } } } }
    """
    per_ds: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    for ds, shots_map in grouped.items():
        fam_map: Dict[str, Dict[int, Dict[str, float]]] = {}
        for shots, cfg_map in shots_map.items():
            for cfg, rs in cfg_map.items():
                accs = [float(r["metrics"].get("accuracy", float('nan'))) for r in rs]
                if accs and math.isnan(accs[0]):
                    accs = [float(r["metrics"].get("top1_acc", float('nan'))) for r in rs]
                eces = [float(r["metrics"].get("ece", float('nan'))) for r in rs]
                aeces = [float(r["metrics"].get("aece", float('nan'))) for r in rs]
                fam = cfg.replace(f"_{shots}shots", "")
                fam_map.setdefault(fam, {})[shots] = {
                    "acc": statistics.fmean(accs) if accs else float('nan'),
                    "ece": statistics.fmean(eces) if eces else float('nan'),
                    "aece": statistics.fmean(aeces) if aeces else float('nan'),
                }
        per_ds[ds] = fam_map
    return per_ds


def _write_table_csv(table_path: Path, fam_to_shots: Dict[str, Dict[int, Dict[str, float]]], shots_set: List[int]) -> None:
    """
    Write a CSV with columns grouped by metric:
      method, acc_1, acc_2, ..., acc_32, ece_1, ece_2, ..., ece_32
    AECE is omitted.
    """
    table_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["method"] + [f"acc_{s}" for s in shots_set] + [f"ece_{s}" for s in shots_set]
    with table_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fam in sorted(fam_to_shots.keys()):
            row = [fam]
            # Accuracy block
            for s in shots_set:
                v = fam_to_shots.get(fam, {}).get(s, {}).get("acc", float('nan'))
                row.append(f"{v:.2f}" if not math.isnan(v) else "")
            # ECE block
            for s in shots_set:
                v = fam_to_shots.get(fam, {}).get(s, {}).get("ece", float('nan'))
                row.append(f"{v:.2f}" if not math.isnan(v) else "")
            writer.writerow(row)


def write_experiment_tables(grouped, exp_name: str, shots_set: List[int] | None = None) -> None:
    """
    Generate experiment-specific tables:
      - One CSV per dataset
      - One CSV averaging across datasets
    Saved to: output/<experiment>/_tables/
    Rows: grouped by metric (Accuracy, ECE, AECE) with one line per method (config family)
    Columns: fixed shot counts {1,2,4,8,16,32}
    """
    if shots_set is None:
        shots_set = [1, 2, 4, 8, 16, 32]
    out_dir = Path("output") / exp_name / "_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-dataset tables
    per_ds = _compute_per_dataset_fam_shot_means(grouped)
    for ds, fam_map in per_ds.items():
        table_path = out_dir / f"{ds}.csv"
        _write_table_csv(table_path, fam_map, shots_set)

    # Average across datasets: build fam -> shot -> lists then average
    agg: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    for fam_map in per_ds.values():
        for fam, shot_map in fam_map.items():
            for s, metrics in shot_map.items():
                fam_s = agg.setdefault(fam, {}).setdefault(s, {"acc": [], "ece": [], "aece": []})
                for metric in ["acc", "ece", "aece"]:
                    if not math.isnan(metrics.get(metric, float('nan'))):
                        fam_s[metric].append(metrics[metric])
    avg_fam: Dict[str, Dict[int, Dict[str, float]]] = {}
    for fam, shot_map in agg.items():
        avg_fam[fam] = {}
        for s, lists in shot_map.items():
            avg_fam[fam][s] = {
                "acc": statistics.fmean(lists["acc"]) if lists["acc"] else float('nan'),
                "ece": statistics.fmean(lists["ece"]) if lists["ece"] else float('nan'),
                "aece": statistics.fmean(lists["aece"]) if lists["aece"] else float('nan'),
            }
    _write_table_csv(out_dir / "Average.csv", avg_fam, shots_set)

def update_global_csv(grouped, exp_name: str, csv_path: Path) -> None:
    """Deprecated: global CSV is no longer used."""
    return None


def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics.json runs under output/<experiment>/")
    ap.add_argument("experiment", help="Experiment subfolder under output/")
    ap.add_argument("--delete", action="store_true", help="Delete uncompleted runs (metrics could not be retrieved)")
    ap.add_argument("--grouped", action="store_true", help="Group multiple configs into single lines using GROUP_SUBSTRINGS")
    ap.add_argument("--show-zero-shot", action="store_true", help="Show zero-shot performance as stars on plots")
    args = ap.parse_args()

    exp_dir = Path("output") / args.experiment
    runs = load_runs(exp_dir, args.delete)
    if not runs:
        print("No metrics.json found. Did the runs finish?")
        return
    grouped = group_by_dataset_shots_config(runs)
    print_summary(grouped)
    # Print an additional table averaging across datasets
    print_average_summary(grouped)
    make_plots(grouped, args.experiment, use_grouping=args.grouped, show_zero_shot=args.show_zero_shot)
    # Always regenerate experiment-specific tables under _tables/
    write_experiment_tables(grouped, args.experiment)


if __name__ == "__main__":
    main()
