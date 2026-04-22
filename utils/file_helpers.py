# Small filesystem helpers used across the pipeline.
# Using pathlib-based alternatives (Path.mkdir(exist_ok=True))
# would be equivalent; kept this os-based for backward compat with
# a few older call sites that pass strings rather than Path objects.

import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
