from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def make_run_dir(name: str) -> Path:
    """Create runs/<date_time>_<name>/ and return its path."""
    run_dir = RUNS_DIR / f"{datetime.now():%Y-%m-%d_%H%M}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
