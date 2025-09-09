#!/usr/bin/env python3
"""
Hyperparameter sweep utility for CLIP-GP.

- Reads a sweep YAML from configs/search/ specifying dataset(s), seeds, shots, and a grid of config overrides
- Expands the Cartesian product and launches `python train.py` runs
- Schedules evenly with per-GPU concurrency via --jobs-per-gpu (default 1)
- Writes a manifest.json and a summary.csv under output_root/<sweep_name>/

Usage examples:
  python -m utils.hparam_search --sweep-file configs/search/gp_small.yaml --devices "0,1" --jobs-per-gpu 1
  python -m utils.hparam_search --sweep-file configs/search/baseline_l2.yaml --jobs-per-gpu 1
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
    base_trainer_cfg: str
    dataset_cfg: str
    output_root: Path
    output_template: str
    grid_overrides: Dict[str, Any]
    extra_env: Dict[str, str]
    root_override: str | None
    sweep_name: str

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
            "sweep": self.sweep_name,
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

        # Build opts overrides as a flat list
        opts: List[str] = []
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
            self.base_trainer_cfg,
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


def load_sweep(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_trials(cfg: Dict[str, Any], cli_devices: str | None) -> Tuple[List[Trial], Dict[str, Any]]:
    name = cfg.get("name", "sweep")
    datasets = cfg.get("datasets") or [cfg.get("dataset")]
    seeds: List[int] = list(cfg.get("seeds", [1]))
    shots: List[int] = list(cfg.get("shots", [1]))
    base_trainer_cfg = cfg.get("trainer_config", "configs/trainers/gp.yaml")
    dataset_cfg_from_yaml = cfg.get("dataset_config")  # optional
    output_root = Path(cfg.get("output_root", "output/search"))
    grid: Dict[str, List[Any]] = cfg.get("grid", {})
    template: str = cfg.get("template", "{sweep}/{dataset}/{sig}/seed{seed}")
    root_override = cfg.get("root")

    # Device hints from sweep file (fallback to CLI)
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
                        base_trainer_cfg=base_trainer_cfg,
                        dataset_cfg=dataset_cfg,
                        output_root=output_root,
                        output_template=template,
                        grid_overrides=overrides,
                        extra_env={},
                        root_override=root_override,
                        sweep_name=name,
                    )
                    trials.append(trial)
                    idx += 1
    meta = {
        "name": name,
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


def run_trials(trials: List[Trial], devices: List[str], jobs_per_gpu: int, retries: int) -> List[Dict[str, Any]]:
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
            attempt = 0
            with sem:
                while True:
                    attempt += 1
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
                            "attempt": attempt,
                        })
                        completed["n"] += 1
                        status = "OK" if success else "FAIL"
                        print(f"[{completed['n']}/{total}] {status} dataset={trial.dataset} shots={trial.shots} seed={trial.seed} config={trial.signature()}")
                    if success or attempt > retries:
                        break
            task_q.task_done()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return results


def _collect_override_keys(trials: List[Trial]) -> List[str]:
    keys: set[str] = set()
    for t in trials:
        keys.update(t.grid_overrides.keys())
    return sorted(list(keys))


def main():
    ap = argparse.ArgumentParser(description="Run hyperparameter sweeps for CLIP-GP.")
    ap.add_argument("--sweep-file", required=True, help="Path to sweep YAML (e.g., configs/search/gp_small.yaml)")
    ap.add_argument("--devices", default=None, help="Comma-separated GPU IDs, e.g., '0,1' (optional)")
    ap.add_argument("--jobs-per-gpu", type=int, default=1, help="Concurrent jobs per GPU (default 1)")
    ap.add_argument("--retries", type=int, default=0, help="Number of retries per failed trial")
    args = ap.parse_args()

    timer_start = time.time()

    sweep_path = Path(args.sweep_file)
    cfg = load_sweep(sweep_path)
    trials, meta = build_trials(cfg, cli_devices=args.devices)

    devices_list = meta.get("devices", [])
    assign_devices(trials, devices_list)

    results = run_trials(trials, devices=devices_list, jobs_per_gpu=max(1, args.jobs_per_gpu), retries=args.retries)
    print(f"Sweep complete: {meta['name']} -> {(trials[0].output_root / meta['name']) if trials else (Path('output/search') / meta['name'])}")
    timer_end = time.time()
    print(f"Sweep completed in {timer_end - timer_start} seconds")


if __name__ == "__main__":
    main()

