import time
from pathlib import Path

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"
LOGS_DIR = ROOT / "logs"
VALIDATOR = ROOT / "utils" / "validate_submission.py"
SEED = 42


def data_dir(arg=None):
    if arg:
        return Path(arg)
    options = (ROOT / "dataset", ROOT / "student_resource" / "dataset")
    for d in options:
        if (d / "train").is_dir() or (d / "test").is_dir():
            return d
    raise FileNotFoundError(f"no train/ or test/ folder under {' or '.join(map(str, options))}; pass --data-dir")


def source_path(data_dir, split, n):
    return Path(data_dir) / split / f"{split}_source{n}.tsv"


def truth_path(data_dir):
    return Path(data_dir) / "train" / "train_ground_truth.tsv"


def log(msg):
    print(f"[{time.perf_counter() - START:8.1f}s] {msg}", flush=True)
