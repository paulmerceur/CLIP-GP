import argparse
import os
import shutil
import subprocess
import sys
import tempfile


METHODS = [
    ("baseline", "Adapter", "configs/trainers/baseline.yaml"),
    ("gp", "Adapter", "configs/trainers/gp.yaml"),
    ("coop", "Adapter-CoOp", "configs/trainers/coop.yaml"),
    ("tipa-f", "Adapter-TipA-F", "configs/trainers/tipa_f.yaml"),
]


DATASET_NAME_MAP = {
    "caltech": "Caltech101",
    "caltech101": "Caltech101",
    "eurosat": "EuroSAT",
    "ucf101": "UCF101",
    "pets": "OxfordPets",
    "flowers": "OxfordFlowers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick smoke-check for core methods")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to use (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--dataset", type=str, default="caltech", help="Dataset short name (default: caltech)")
    parser.add_argument("--verbose", action="store_true", help="Print full logs for each run")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone (optional)")
    return parser.parse_args()


def normalize_dataset(name: str) -> str:
    key = (name or "caltech").strip().lower()
    return DATASET_NAME_MAP.get(key, name)


def run_one(method_key: str, trainer_name: str, config_file: str, dataset: str, gpu: str, verbose: bool, backbone: str | None) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tmpdir = tempfile.mkdtemp(prefix=f"clipgp_quick_{method_key}_")
    cmd = [
        sys.executable,
        "-m",
        "train",
        "--config-file",
        config_file,
        "--trainer",
        trainer_name,
        "--dataset",
        dataset,
        "--seed",
        "1",
        "--epochs",
        "10",
        "--output-dir",
        tmpdir,
    ]
    if backbone:
        cmd += ["--backbone", backbone]

    try:
        if verbose:
            rc = subprocess.call(cmd, env=env)
        else:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Consume output to avoid blocking but do not print
            assert proc.stdout is not None
            for _ in proc.stdout:
                pass
            rc = proc.wait()
        return rc == 0
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    dataset = normalize_dataset(args.dataset)
    all_ok = True
    for method_key, trainer_name, cfg in METHODS:
        ok = run_one(method_key, trainer_name, cfg, dataset, args.gpu, args.verbose, args.backbone)
        all_ok = all_ok and ok
        if args.verbose:
            status = "OK" if ok else "FAIL"
            print(f"[{method_key}] {status}")
        else:
            print(f"[{method_key}] {'OK' if ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


