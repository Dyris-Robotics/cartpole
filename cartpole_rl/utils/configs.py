from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Read a YAML config file and return it as a dict of dicts.

    For example config["terminations"]["cart_pos_limit"].
    """
    with open(Path(path)) as f:
        return yaml.safe_load(f) or {}
