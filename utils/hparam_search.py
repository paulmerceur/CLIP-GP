#!/usr/bin/env python3
"""
Experiment runner for CLIP-GP (supports grids and single runs).

- Reads an experiment YAML (now stored under configs/trainers/) specifying dataset(s), seeds, shots,
  optional grid of overrides, and may embed the trainer configuration (DATALOADER/INPUT/OPTIM/TRAIN/TRAINER).
- Expands the Cartesian product and launches `python train.py` runs
- Schedules evenly with per-GPU concurrency via --jobs-per-gpu (default 1)
- Writes results under output/<experiment_name>/...

Notes:
- If you want a single run, omit the `grid` key (or leave it empty).
- The experiment name can be provided via --experiment-name; it defaults to the YAML filename stem.

Usage examples:
  python -m utils.hparam_search --config-file configs/trainers/gp_small.yaml --devices "0,1" --jobs-per-gpu 1
  python -m utils.hparam_search --config-file configs/trainers/baseline_l2.yaml --jobs-per-gpu 1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

import yaml


@dataclass
class Trial:
    index: int
    dataset: str
    seed: int
    shots: int
    dataset_cfg: str
    output_root: Path
    output_template: str
    grid_overrides: Dict[str, Any]
    extra_env: Dict[str, str]
    root_override: str | None
    experiment_name: str
    base_opts: List[str]
    config_file: str

    def signature(self) -> str:
        # Human-readable signature: join "<lastkey><value>" pairs without hashing
        if not self.grid_overrides:
            return "default"
        parts: List[str] = []
        for k, v in sorted(self.grid_overrides.items()):
            short = k.split(".")[-1]
            parts.append(f"{short}{v}")
        return "_".join(parts)

    def format_outdir(self) -> Path:
        # Prepare placeholders
        placeholders = {
            # Backward compatibility: expose both names
            "sweep": self.experiment_name,
            "experiment": self.experiment_name,
            "dataset": self.dataset,
            "shots": self.shots,
            "seed": self.seed,
            "sig": self.signature(),
        }
        # Allow any grid key as placeholder
        placeholders.update({k: v for k, v in self.grid_overrides.items()})

        rel = self.output_template.format(**placeholders)
        return (self.output_root / rel).resolve()

    def to_command(self, python_exe: str = sys.executable) -> Tuple[List[str], Dict[str, str]]:
        out_dir = self.format_outdir()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build opts: base trainer opts first, then overrides to allow overriding
        opts: List[str] = list(self.base_opts)
        for k, v in sorted(self.grid_overrides.items()):
            opts.extend([k, str(v)])
        # Also set shots explicitly
        opts.extend(["DATASET.NUM_SHOTS", str(self.shots)])

        # Assemble base command
        cmd = [
            python_exe,
            "train.py",
            "--dataset-config-file",
            self.dataset_cfg,
            "--config-file",
            self.config_file,
            "--dataset",
            self.dataset_map(self.dataset),
            "--seed",
            str(self.seed),
            "--output-dir",
            str(out_dir),
        ]
        # root override
        if self.root_override:
            cmd.extend(["--root", self.root_override])
        # Append opts at the end
        cmd.extend(opts)

        env = os.environ.copy()
        env.update(self.extra_env)
        return cmd, env

    @staticmethod
    def dataset_map(name: str) -> str:
        # Map YAML names to train.py expected names if they differ
        # current configs use lowercase filenames; train.py expects capitalised names
        mapping = {
            "caltech101": "Caltech101",
            "oxford_pets": "OxfordPets",
            "oxford_flowers": "OxfordFlowers",
            "fgvc_aircraft": "FGVCAircraft",
            "dtd": "DescribableTextures",
            "eurosat": "EuroSAT",
            "stanford_cars": "StanfordCars",
            "food101": "Food101",
            "sun397": "SUN397",
            "ucf101": "UCF101",
            "imagenet": "ImageNet",
            "imagenet_sketch": "ImageNetSketch",
            "imagenetv2": "ImageNetV2",
            "imagenet_a": "ImageNetA",
            "imagenet_r": "ImageNetR",
        }
        return mapping.get(name.lower(), name)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_trials(cfg: Dict[str, Any], cli_devices: str | None) -> Tuple[List[Trial], Dict[str, Any]]:
    name = cfg.get("name") or "experiment"
    datasets = cfg.get("datasets") or [cfg.get("dataset")]
    seeds: List[int] = list(cfg.get("seeds", [1]))
    shots: List[int] = list(cfg.get("shots", [1]))
    dataset_cfg_from_yaml = cfg.get("dataset_config")  # optional
    output_root = Path(cfg.get("output_root", "output"))
    grid: Dict[str, List[Any]] = cfg.get("grid", {})
    template: str = cfg.get("template", "{experiment}/{dataset}/{sig}/seed{seed}")
    root_override = cfg.get("root")
    config_file_path = str(cfg.get("__config_file__", ""))

    # Device hints from config file (fallback to CLI)
    devices = cfg.get("devices") or cli_devices or ""
    devices_list = [d.strip() for d in devices.split(",") if d.strip()]

    # Expand Cartesian product of grid keys
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]

    trials: List[Trial] = []
    idx = 0
    for ds in datasets:
        dataset_cfg = dataset_cfg_from_yaml or f"configs/datasets/{ds}.yaml"
        for seed in seeds:
            for nshot in shots:
                for combo in itertools.product(*values) if keys else [()]:
                    overrides = {keys[i]: combo[i] for i in range(len(keys))}
                    # Prepare per-trial env, including device assignment (filled in later)
                    trial = Trial(
                        index=idx,
                        dataset=ds,
                        seed=int(seed),
                        shots=int(nshot),
                        dataset_cfg=dataset_cfg,
                        output_root=output_root,
                        output_template=template,
                        grid_overrides=overrides,
                        extra_env={},
                        root_override=root_override,
                        experiment_name=name,
                        base_opts=[],
                        config_file=config_file_path,
                    )
                    trials.append(trial)
                    idx += 1
    meta = {
        "experiment_name": name,
        "n_trials": len(trials),
        "devices": devices_list,
    }
    return trials, meta


def assign_devices(trials: List[Trial], devices: List[str]) -> None:
    if not devices:
        return
    for i, t in enumerate(trials):
        dev = devices[i % len(devices)]
        t.extra_env["CUDA_VISIBLE_DEVICES"] = str(dev)


def run_trials(trials: List[Trial], devices: List[str], jobs_per_gpu: int) -> List[Dict[str, Any]]:
    """Run trials enforcing even per-GPU concurrency (jobs per GPU), printing concise progress.

    Child stdout/stderr are suppressed; logs are still written inside each trial's output dir.
    """
    results: List[Dict[str, Any]] = []
    total = len(trials)
    completed = {"n": 0}

    task_q: "queue.Queue[Trial]" = queue.Queue()
    for t in trials:
        task_q.put(t)

    lock = threading.Lock()

    # Per-device semaphores to cap concurrent jobs per GPU
    dev_ids = devices if devices else [""]  # empty string denotes CPU/no explicit device
    semaphores = {d: threading.Semaphore(max(1, jobs_per_gpu)) for d in dev_ids}

    # total threads equals sum of capacities
    n_threads = max(1, len(dev_ids) * max(1, jobs_per_gpu))

    def get_dev_for_trial(t: Trial) -> str:
        return t.extra_env.get("CUDA_VISIBLE_DEVICES", "")

    def worker(worker_id: int):
        while True:
            try:
                trial = task_q.get_nowait()
            except queue.Empty:
                return
            dev = get_dev_for_trial(trial)
            sem = semaphores.get(dev, None)
            if sem is None:
                sem = semaphores[""]
            with sem:
                cmd, env = trial.to_command()
                # Suppress child outputs; logs are still written to file by train.py
                rc = subprocess.call(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                success = (rc == 0)
                with lock:
                    results.append({
                        "index": trial.index,
                        "dataset": trial.dataset,
                        "seed": trial.seed,
                        "shots": trial.shots,
                        "sig": trial.signature(),
                        "out_dir": str(trial.format_outdir()),
                        "overrides": trial.grid_overrides,
                        "return_code": rc,
                    })
                    completed["n"] += 1
                    status = "OK" if success else "FAIL"
                    print(f"[{completed['n']}/{total}] {status} dataset={trial.dataset} shots={trial.shots} seed={trial.seed} config={trial.signature()}")
            task_q.task_done()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return results


def main():
    ap = argparse.ArgumentParser(description="Run experiments for CLIP-GP (grids or single runs).")
    ap.add_argument("--config-file", required=True, help="Path to experiment YAML (e.g., configs/trainers/gp_small.yaml)")
    ap.add_argument("--devices", default=None, help="Comma-separated GPU IDs, e.g., '0,1' (optional)")
    ap.add_argument("--jobs-per-gpu", type=int, default=1, help="Concurrent jobs per GPU (default 1)")
    ap.add_argument("--experiment-name", default=None, help="Optional experiment name (defaults to YAML filename or 'name' field)")
    args = ap.parse_args()

    timer_start = time.time()

    config_path = Path(args.config_file)
    cfg = load_config(config_path)
    # Inject config file path for trainer consumption and naming
    cfg["__config_file__"] = str(config_path)
    # Determine experiment name
    if args.experiment_name:
        cfg["name"] = args.experiment_name
    elif not cfg.get("name"):
        cfg["name"] = config_path.stem

    trials, meta = build_trials(cfg, cli_devices=args.devices)

    devices_list = meta.get("devices", [])
    assign_devices(trials, devices_list)

    results = run_trials(trials, devices=devices_list, jobs_per_gpu=max(1, args.jobs_per_gpu))
    exp_name = meta.get("experiment_name", "experiment")
    print(f"Experiment complete: {exp_name} -> {(trials[0].output_root / exp_name) if trials else (Path('output') / exp_name)}")
    timer_end = time.time()
    print(f"Completed in {timer_end - timer_start} seconds")


if __name__ == "__main__":
    main()

