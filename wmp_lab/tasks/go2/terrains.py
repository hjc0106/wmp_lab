# Copyright (c) 2026, WMP-Go2-Lab authors.
# SPDX-License-Identifier: BSD-3-Clause
"""Custom trimesh sub-terrain generators for the WMP Go2 curriculum.

These follow the IsaacLab ``SubTerrainBaseCfg.function`` contract
 ``(difficulty, cfg) -> (list[trimesh.Trimesh], np.ndarray origin)``.
They are intentionally simple but valid collision meshes.

The two terrains below emulate the IsaacGym WMP ``tilt`` and ``crawl``
obstacles:
    * ``tilt``  : a sloped ramp the robot climbs (heading reward disabled).
    * ``crawl`` : a low bridge the robot must crawl under (depth camera
                  sees the ceiling) sitting on a flat ground.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.trimesh.utils import make_plane  # noqa: F401


def tilt_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """A constant-slope ramp rising along +x within the sub-terrain tile."""
    slope_range = getattr(cfg, "slope_range", (0.05, 0.35))
    slope = slope_range[0] + difficulty * (slope_range[1] - slope_range[0])
    height = slope * cfg.size[0]

    # Build a sloped plane via a small heightfield extruded into a mesh.
    nx, ny = 32, 16
    xs = np.linspace(0.0, cfg.size[0], nx)
    ys = np.linspace(0.0, cfg.size[1], ny)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    grid_z = (grid_x / cfg.size[0]) * height
    # construct vertices/faces
    verts = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = (i + 1) * ny + j
            c = i * ny + (j + 1)
            d = (i + 1) * ny + (j + 1)
            faces.append([a, b, d])
            faces.append([a, d, c])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces))
    origin = np.asarray([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    return [mesh], origin


def crawl_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """A flat ground with a low overhead bridge forcing a crawl gait."""
    ceiling_height = getattr(cfg, "ceiling_height_range", (0.25, 0.3))
    h = ceiling_height[1] - difficulty * (ceiling_height[1] - ceiling_height[0])
    bridge_len = getattr(cfg, "bridge_length", cfg.size[0] * 0.7)
    bridge_w = getattr(cfg, "bridge_width", cfg.size[1] * 0.9)
    thickness = 0.1

    ground = make_plane(cfg.size, 0.0, center_zero=False)
    # overhead box centered on the tile, gap of `h` below its lower face
    dim = [bridge_len, bridge_w, thickness]
    center = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, h + thickness / 2.0)
    bridge = trimesh.creation.box(
        dim, trimesh.transformations.translation_matrix(center)
    )
    origin = np.asarray([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    return [ground, bridge], origin