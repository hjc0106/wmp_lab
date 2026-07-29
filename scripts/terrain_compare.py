#!/usr/bin/env python3
"""Capture metrics from the ported generator for comparison with legacy goldens.

The golden file must be produced separately by ``export_legacy_terrain_baseline.py``
inside the Isaac Gym environment; this script never overwrites that baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap():
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _mesh_metrics(mesh) -> dict:
    verts = mesh.vertices
    return {
        "num_vertices": int(len(verts)),
        "num_faces": int(len(mesh.faces)),
        "bounds_min": verts.min(axis=0).tolist(),
        "bounds_max": verts.max(axis=0).tolist(),
    }


def collect_metrics(
    seeds: list[int],
    difficulties: list[float],
    num_cols: int = 20,
    num_rows: int = 10,
    compat_mode: str = "legacy_exact",
) -> dict:
    from wmp_lab.tasks.go2.legacy_terrain_layout import LEGACY_TERRAIN_NAMES, column_category_names
    from wmp_lab.tasks.go2.legacy_terrain_utils import combine_meshes
    from wmp_lab.tasks.go2.terrains import make_legacy_tile

    out: dict = {
        "meta": {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "compat_mode": compat_mode,
            "seeds": seeds,
            "difficulties": difficulties,
            "categories": list(LEGACY_TERRAIN_NAMES),
            "column_map": column_category_names(num_cols),
        },
        "tiles": [],
    }

    col_names = column_category_names(num_cols)
    for seed in seeds:
        for row in range(num_rows):
            diff = row / num_rows
            if difficulties and diff not in difficulties and not any(abs(diff - d) < 1e-6 for d in difficulties):
                continue
            for col in range(num_cols):
                meshes, origin, cat, tile_diff, _tile_hf = make_legacy_tile(
                    row=row,
                    column=col,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    training_seed=seed,
                    compat_mode=compat_mode,  # type: ignore[arg-type]
                )
                mesh = combine_meshes(meshes)
                hf = None
                entry = {
                    "seed": seed,
                    "row": row,
                    "column": col,
                    "category": cat,
                    "expected_category": col_names[col],
                    "difficulty": tile_diff,
                    "origin": origin.tolist(),
                    "mesh": _mesh_metrics(mesh),
                }
                out["tiles"].append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description="Export legacy terrain comparison metrics")
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "ported_terrain_metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42])
    parser.add_argument("--difficulties", type=float, nargs="*", default=[0.0, 0.3, 0.6, 0.9])
    parser.add_argument("--compat", default="legacy_exact", choices=["legacy_exact", "legacy_fixed"])
    parser.add_argument("--full-grid", action="store_true", help="Record all 10×20 tiles, not just selected difficulties")
    args = parser.parse_args()

    _bootstrap()
    diffs = [] if args.full_grid else args.difficulties
    metrics = collect_metrics(args.seeds, diffs, compat_mode=args.compat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {len(metrics['tiles'])} tile records to {args.output}")


if __name__ == "__main__":
    main()
