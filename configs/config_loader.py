# Loads YAML configs relative to this file's directory.
#
# All config files live alongside this loader in configs/. Callers
# pass a filename (e.g. "database.yaml") rather than a path so the
# pipeline doesn't care about the caller's working directory.

import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_config(file_name="config.yaml"):
    path = BASE_DIR / file_name
    with open(path, "r") as f:
        return yaml.safe_load(f)
