"""Geometry and layout tests for legacy WMP terrains."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wmp_lab.tasks.go2.legacy_terrain_layout import (
    EXPECTED_COLUMN_COUNTS,
    column_category_counts,
    column_category_names,
    ordered_row_difficulty,
)
from wmp_lab.tasks.go2.legacy_terrain_utils import (
    TILE_SIZE,
    VERTICAL_SCALE,
    SubTerrain,
    TileRng,
    stable_tile_seed,
    terrain_origin_from_heightfield,
)
from wmp_lab.tasks.go2.terrains import (
    legacy_climb_terrain,
    legacy_crawl_terrain,
    legacy_gap_terrain,
    legacy_slope_terrain,
    legacy_stairs_terrain,
    legacy_tilt_terrain,
    make_legacy_tile,
)

DATA = Path(__file__).resolve().parent / "data" / "legacy_terrain_metrics.json"


class TestLayout:
    def test_twenty_column_allocation(self):
        counts = column_category_counts(20)
        for name, expected in EXPECTED_COLUMN_COUNTS.items():
            assert counts[name] == expected, f"{name}: got {counts[name]}, want {expected}"

    def test_row_difficulties(self):
        diffs = [ordered_row_difficulty(r, 10) for r in range(10)]
        assert diffs == pytest.approx([i / 10 for i in range(10)])


class TestDeterminism:
    def test_same_seed_identical_mesh(self):
        a, oa, _, _ = make_legacy_tile(3, 5, 10, 20, training_seed=99)
        b, ob, _, _ = make_legacy_tile(3, 5, 10, 20, training_seed=99)
        assert np.allclose(oa, ob)
        assert len(a[0].vertices) == len(b[0].vertices)
        assert np.allclose(a[0].vertices, b[0].vertices)

    def test_different_seed_may_differ(self):
        a, _, ca, _ = make_legacy_tile(1, 10, 10, 20, training_seed=1)
        b, _, cb, _ = make_legacy_tile(1, 10, 10, 20, training_seed=2)
        assert ca == cb
        if ca in ("gap", "climb", "tilt", "crawl"):
            assert not np.allclose(a[0].vertices, b[0].vertices)


class TestFormulas:
    def test_heightfield_size(self):
        terrain = SubTerrain.create()
        assert terrain.height_field_raw.shape == (80, 80)

    def test_tile_mesh_covers_full_eight_meters(self):
        meshes, _, _, _ = make_legacy_tile(0, 0, 10, 20, training_seed=1)
        assert meshes[0].bounds[0, :2] == pytest.approx([0.0, 0.0])
        assert meshes[0].bounds[1, :2] == pytest.approx([TILE_SIZE, TILE_SIZE])

    def test_gap_depth(self):
        rng = TileRng(stable_tile_seed(1, "gap", 5, 0, "legacy_exact"))
        meshes, _ = legacy_gap_terrain(0.5, rng)
        zs = meshes[0].vertices[:, 2]
        assert zs.min() < -4.9

    def test_climb_is_positive(self):
        meshes, _ = legacy_climb_terrain(0.5, TileRng(123))
        assert meshes[0].vertices[:, 2].max() >= 0.29

    def test_legacy_slope_column_is_negative(self):
        meshes, _ = legacy_slope_terrain(0.5, TileRng(123), 0, 20)
        assert meshes[0].vertices[:, 2].min() < -0.3
        assert meshes[0].vertices[:, 2].max() <= 0.051

    def test_stair_category_directions_match_legacy_columns(self):
        negative, _ = legacy_stairs_terrain(0.5, TileRng(123), 1, 20, stairs_up=False)
        positive, _ = legacy_stairs_terrain(0.5, TileRng(123), 4, 20, stairs_up=True)
        assert negative[0].vertices[:, 2].min() < 0.0
        assert positive[0].vertices[:, 2].max() > 0.0

    def test_tilt_channel_width(self):
        difficulty = 0.5
        meshes, _ = legacy_tilt_terrain(difficulty, TileRng(123))
        top_vertices = meshes[0].vertices[meshes[0].vertices[:, 2] > 0.9]
        left_edge = top_vertices[top_vertices[:, 1] < 4.0, 1].max()
        right_edge = top_vertices[top_vertices[:, 1] > 4.0, 1].min()
        assert right_edge - left_edge == pytest.approx(0.32 - 0.04 * difficulty)

    def test_crawl_clearance_and_exact_overlap(self):
        difficulty = 0.5
        meshes, _ = legacy_crawl_terrain(difficulty, TileRng(456), "legacy_exact")
        raised = meshes[0].vertices[meshes[0].vertices[:, 2] > 0.0]
        assert raised[:, 2].min() == pytest.approx(0.35 - 0.15 * difficulty)
        assert len(np.unique(np.round(raised[:, 0], 6))) == 2

    def test_crawl_fixed_symmetric(self):
        rng = TileRng(456)
        meshes, _ = legacy_crawl_terrain(0.5, rng, "legacy_fixed")
        verts = meshes[0].vertices
        cx = 4.0
        front = verts[(verts[:, 0] > cx) & (verts[:, 2] > 0.3)][:, 0].mean()
        back = verts[(verts[:, 0] < cx) & (verts[:, 2] > 0.3)][:, 0].mean()
        assert front > cx
        assert back < cx
        assert front - cx == pytest.approx(cx - back)

    def test_origin_center_2m_max(self):
        hf = np.zeros((80, 80), dtype=np.int16)
        hf[35:45, 35:45] = 100
        origin = terrain_origin_from_heightfield(hf)
        assert origin[2] == pytest.approx(100 * VERTICAL_SCALE)


class TestCategoryMaskConsistency:
    def test_generator_matches_column_map(self):
        names = column_category_names(20)
        for col, expected in enumerate(names):
            _, _, cat, _ = make_legacy_tile(0, col, 10, 20, training_seed=1)
            assert cat == expected


@pytest.mark.skipif(not DATA.exists(), reason="baseline metrics not generated")
class TestBaselineFile:
    def test_metrics_file_loads(self):
        data = json.loads(DATA.read_text())
        assert "tiles" in data
        assert data["meta"]["num_cols"] == 20
        assert data["meta"]["source"] == "legacy_isaacgym"

    def test_height_extrema_match_real_legacy_output(self):
        data = json.loads(DATA.read_text())
        heightfield_categories = {
            "slope", "stairs_up", "stairs_down", "gap", "climb", "rough_flat"
        }
        for entry in data["tiles"]:
            if entry["category"] not in heightfield_categories:
                continue
            meshes, _, category, _ = make_legacy_tile(
                entry["row"], entry["column"], 10, 20, training_seed=entry["seed"]
            )
            heights = np.rint(meshes[0].vertices[:, 2] / VERTICAL_SCALE).astype(int)
            assert category == entry["category"]
            # Per-tile RNG deliberately differs from the legacy global stream;
            # extrema may differ by one quantization unit after interpolation.
            assert abs(int(heights.min()) - entry["height_min"]) <= 1
            assert abs(int(heights.max()) - entry["height_max"]) <= 1
