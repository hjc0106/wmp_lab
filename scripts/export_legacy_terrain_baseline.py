#!/usr/bin/env python3
"""Export golden metrics by executing the original Isaac Gym WMP terrain code.

Run this script with the legacy Isaac Gym Python environment. It stubs only the
unrelated legged-gym package imports; all terrain generation comes from the
reference repository and NVIDIA's ``isaacgym.terrain_utils``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_REFERENCE = Path("/home/hongjiacheng/codes/robot_lab/WMP-go2_version")
ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_legacy_terrain(reference: Path):
    # Isaac Gym still uses this removed NumPy alias during import.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    legged_gym = types.ModuleType("legged_gym")
    legged_gym.__path__ = []
    utils = types.ModuleType("legged_gym.utils")
    utils.__path__ = []
    envs = types.ModuleType("legged_gym.envs")
    envs.__path__ = []
    base = types.ModuleType("legged_gym.envs.base")
    base.__path__ = []
    config = types.ModuleType("legged_gym.envs.base.legged_robot_config")

    class LeggedRobotCfg:
        class terrain:
            pass

    config.LeggedRobotCfg = LeggedRobotCfg
    sys.modules.update(
        {
            "legged_gym": legged_gym,
            "legged_gym.utils": utils,
            "legged_gym.envs": envs,
            "legged_gym.envs.base": base,
            "legged_gym.envs.base.legged_robot_config": config,
        }
    )
    legacy_trimesh = _load_module(
        "legged_gym.utils.trimesh", reference / "legged_gym" / "utils" / "trimesh.py"
    )
    utils.trimesh = legacy_trimesh
    return _load_module("legacy_wmp_terrain", reference / "legged_gym" / "utils" / "terrain.py")


def _terrain_cfg():
    return SimpleNamespace(
        mesh_type="heightfield",
        curriculum=True,
        selected=False,
        terrain_kwargs=None,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=25.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        terrain_proportions=[0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05],
        slope_treshold=0.75,
    )


def collect(reference: Path, seeds: list[int], rows: list[int]) -> dict:
    legacy = _load_legacy_terrain(reference)
    names = (
        "wave", "slope", "stairs_up", "stairs_down", "discrete",
        "gap", "climb", "tilt", "crawl", "rough_flat",
    )
    tiles = []
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        cfg = _terrain_cfg()
        terrain = legacy.Terrain(cfg, num_robots=1)
        border = int(cfg.border_size / cfg.horizontal_scale)
        tile_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        cumulative = np.cumsum(cfg.terrain_proportions)
        for row in rows:
            for col in range(cfg.num_cols):
                choice = col / cfg.num_cols + 0.001
                category = names[int(np.searchsorted(cumulative, choice, side="right"))]
                x0 = border + row * tile_pixels
                y0 = border + col * tile_pixels
                hf = terrain.height_field_raw[x0 : x0 + tile_pixels, y0 : y0 + tile_pixels]
                tiles.append(
                    {
                        "seed": seed,
                        "row": row,
                        "column": col,
                        "category": category,
                        "difficulty": row / cfg.num_rows,
                        "height_min": int(hf.min()),
                        "height_max": int(hf.max()),
                        "origin": terrain.env_origins[row, col].tolist(),
                    }
                )
    return {
        "meta": {
            "source": "legacy_isaacgym",
            "reference": str(reference / "legged_gym" / "utils" / "terrain.py"),
            "num_rows": 10,
            "num_cols": 20,
            "seeds": seeds,
            "rows": rows,
        },
        "tiles": tiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/data/legacy_terrain_metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7])
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 3, 6, 9])
    args = parser.parse_args()
    result = collect(args.reference, args.seeds, args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result['tiles'])} legacy tiles to {args.output}")


if __name__ == "__main__":
    main()
