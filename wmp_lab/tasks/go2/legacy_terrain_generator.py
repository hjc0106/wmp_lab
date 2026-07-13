"""Legacy WMP terrain generator compatible with Isaac Lab ``TerrainImporter``."""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import trimesh
from isaaclab.utils.configclass import configclass
from isaaclab.utils.timer import Timer

from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_border

from .legacy_terrain_layout import DEFAULT_TERRAIN_PROPORTIONS, tile_translation
from .legacy_terrain_utils import CompatMode, combine_meshes, stable_tile_seed
from .terrains import make_legacy_tile

logger = logging.getLogger(__name__)


@configclass
class LegacyTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration for deterministic legacy WMP terrain generation."""

    # Resolved in __post_init__ because the generator class is declared below.
    class_type: type = None
    compat_mode: str = "legacy_exact"
    ordered_generation: bool = True
    slope_direction: str = "legacy"
    terrain_proportions: list[float] = list(DEFAULT_TERRAIN_PROPORTIONS)

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = LegacyTerrainGenerator


class LegacyTerrainGenerator(TerrainGenerator):
    """Generate terrains using legacy column/difficulty rules with per-tile RNG."""

    cfg: LegacyTerrainGeneratorCfg

    def __init__(self, cfg: LegacyTerrainGeneratorCfg, device: str = "cpu"):
        if cfg.compat_mode not in ("legacy_exact", "legacy_fixed"):
            raise ValueError(f"Unsupported terrain compatibility mode: {cfg.compat_mode!r}")
        if cfg.slope_direction not in ("legacy", "up", "down"):
            raise ValueError(f"Unsupported slope direction: {cfg.slope_direction!r}")
        if tuple(cfg.size) != (8.0, 8.0) or cfg.horizontal_scale != 0.1 or cfg.vertical_scale != 0.005:
            raise ValueError("Legacy WMP terrains require size=8x8, horizontal_scale=0.1, vertical_scale=0.005")
        if len(cfg.terrain_proportions) != 10 or sum(cfg.terrain_proportions) <= 0:
            raise ValueError("Legacy WMP terrain proportions must contain 10 entries with a positive sum")
        if len(cfg.sub_terrains) == 0:
            # Isaac Lab requires at least one sub-terrain entry; legacy path ignores it.
            from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

            cfg.sub_terrains = {"legacy": SubTerrainBaseCfg(proportion=1.0, function=lambda d, c: ([], np.zeros(3)))}
        self.cfg = cfg
        self.device = device
        self.flat_patches = {}
        self.terrain_meshes: list[trimesh.Trimesh] = []
        self.terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        if cfg.use_cache and cfg.seed is None:
            logger.warning("Legacy terrain cache enabled without seed; generation may not be reproducible.")

        with Timer("[INFO] Generating legacy terrains took"):
            self._generate_legacy_tiles()
        self._add_terrain_border()
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        transform = np.eye(4)
        transform[:2, -1] = -cfg.size[0] * cfg.num_rows * 0.5, -cfg.size[1] * cfg.num_cols * 0.5
        self.terrain_mesh.apply_transform(transform)
        self.terrain_origins += transform[:3, -1]

    def _generate_legacy_tiles(self) -> None:
        seed = 0 if self.cfg.seed is None else int(self.cfg.seed)
        compat: CompatMode = self.cfg.compat_mode  # type: ignore[assignment]
        slope_dir: Literal["legacy", "up", "down"] = self.cfg.slope_direction  # type: ignore[assignment]
        props = list(self.cfg.terrain_proportions)

        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = None
                if not self.cfg.ordered_generation:
                    difficulty_seed = stable_tile_seed(seed, "difficulty", row, col, compat)
                    difficulty = float(np.random.default_rng(difficulty_seed).uniform(0.0, 1.0))
                meshes, origin, _cat, _diff = make_legacy_tile(
                    row=row,
                    column=col,
                    num_rows=self.cfg.num_rows,
                    num_cols=self.cfg.num_cols,
                    training_seed=seed,
                    compat_mode=compat,
                    proportions=props,
                    slope_direction=slope_dir,
                    difficulty_override=difficulty,
                )
                mesh = combine_meshes(meshes)
                self._add_sub_terrain_legacy(mesh, origin, row, col)

    def _add_sub_terrain_legacy(self, mesh: trimesh.Trimesh, origin: np.ndarray, row: int, col: int) -> None:
        transform = np.eye(4)
        # Legacy tile meshes use local coordinates in [0, size], not center-zero coordinates.
        transform[0:2, -1] = tile_translation(row, col, self.cfg.size)
        mesh = mesh.copy()
        mesh.apply_transform(transform)
        self.terrain_meshes.append(mesh)
        self.terrain_origins[row, col] = origin + transform[:3, -1]

    def _add_terrain_border(self) -> None:
        border_size = (
            self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
            self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
        )
        inner_size = (self.cfg.num_rows * self.cfg.size[0], self.cfg.num_cols * self.cfg.size[1])
        border_center = (
            self.cfg.num_rows * self.cfg.size[0] / 2,
            self.cfg.num_cols * self.cfg.size[1] / 2,
            -self.cfg.border_height / 2,
        )
        border_meshes = make_border(border_size, inner_size, height=abs(self.cfg.border_height), position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        self.terrain_meshes.append(border)
