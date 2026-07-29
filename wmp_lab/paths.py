"""Resolve WMP lab asset paths for dev installs and packaged installs."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def lab_root() -> Path:
    """Root directory containing ``resources/``, ``datasets/``, ``generated/``."""
    env_root = os.environ.get("WMP_LAB_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if root.is_dir():
            return root

    # .../wmp_lab/wmp_lab/paths.py -> repo root is parents[1]
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "resources").is_dir() and (candidate / "datasets").is_dir():
        return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "resources").is_dir() and (cwd / "datasets").is_dir():
        return cwd

    raise FileNotFoundError(
        "Could not locate WMP lab root (expected resources/ and datasets/). "
        "Set WMP_LAB_ROOT to the repository root."
    )


def resource_path(*parts: str) -> Path:
    return lab_root().joinpath("resources", *parts)


def dataset_path(*parts: str) -> Path:
    return lab_root().joinpath("datasets", *parts)


def generated_path(*parts: str) -> Path:
    path = lab_root().joinpath("generated", *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
