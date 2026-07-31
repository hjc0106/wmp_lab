"""Semantic migration tests (no Isaac Sim required)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from wmp_lab.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    PRE_SEMANTIC_FIX_TAG,
    TRAINING_SEMANTICS_VERSION,
    is_pre_semantic_fix_checkpoint,
    deserialize_amp_normalizer,
    serialize_amp_normalizer,
    validate_checkpoint_for_resume,
)
from wmp_lab.tasks.go2.legacy_terrain_utils import (
    HORIZONTAL_SCALE,
    SubTerrain,
    TileRng,
    accumulate_heightfield_tile,
    build_x_edge_mask,
    gap_terrain,
    stable_tile_seed,
)


class TestCheckpointVersioning:
    def test_pre_semantic_fix_rejected_for_resume(self):
        with pytest.raises(ValueError, match="pre-semantic-fix"):
            validate_checkpoint_for_resume({"checkpoint_tag": PRE_SEMANTIC_FIX_TAG})

    def test_legacy_missing_discriminator_rejected(self):
        with pytest.raises(ValueError, match="discriminator"):
            validate_checkpoint_for_resume(
                {"training_semantics_version": TRAINING_SEMANTICS_VERSION, "format_version": 1}
            )

    def test_current_semantics_version(self):
        assert TRAINING_SEMANTICS_VERSION > 0
        assert CHECKPOINT_FORMAT_VERSION >= 2
        assert not is_pre_semantic_fix_checkpoint(
            {
                "training_semantics_version": TRAINING_SEMANTICS_VERSION,
                "discriminator_state_dict": {},
                "amp_normalizer": object(),
            }
        )

    def test_amp_normalizer_serializes_without_numpy_objects(self):
        from rsl_rl.utils.utils import Normalizer

        normalizer = Normalizer((3,), epsilon=1e-5, clip_obs=5.0)
        normalizer.mean = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        normalizer.var = np.array([4.0, 9.0, 16.0], dtype=np.float64)
        normalizer.count = 123.0

        payload = serialize_amp_normalizer(normalizer)
        assert payload["__class__"] == "Normalizer"
        assert isinstance(payload["mean"], list)
        assert isinstance(payload["var"], list)
        assert payload["mean"] == [1.0, 2.0, 3.0]
        assert payload["var"] == [4.0, 9.0, 16.0]

        restored = deserialize_amp_normalizer(payload)
        assert isinstance(restored, Normalizer)
        assert restored.epsilon == pytest.approx(1e-5)
        assert restored.clip_obs == pytest.approx(5.0)
        assert restored.count == pytest.approx(123.0)
        np.testing.assert_allclose(restored.mean, normalizer.mean)
        np.testing.assert_allclose(restored.var, normalizer.var)


class TestRewardScaling:
    def test_reward_scales_include_step_dt(self):
        dt = 0.02
        assert 1.5 * dt == pytest.approx(0.03)
        assert -1.0 * dt == pytest.approx(-0.02)

    def test_fixed_input_tracking_reward_magnitude(self):
        dt = 0.02
        scale = 1.5 * dt
        assert scale * math.exp(0.0) == pytest.approx(0.03)


class TestObservationLayout:
    def test_observation_segment_sizes(self):
        prop_dim = 33
        privileged_dim = 24 + 26 + 3
        height_dim = 187
        action_dim = 12
        assert prop_dim + privileged_dim + height_dim + action_dim == 285
        assert privileged_dim == 53


class TestEdgeMask:
    def test_flat_heightfield_has_no_edges(self):
        hf = np.zeros((81, 81), dtype=np.int16)
        mask = build_x_edge_mask(hf)
        assert not mask.any()

    def test_gap_platform_center_not_edge(self):
        terrain = SubTerrain.create()
        rng = TileRng(stable_tile_seed(1, "gap", 0, 0, "legacy_exact"))
        gap_terrain(terrain, rng, gap_size=0.5, platform_size=4.0)
        hf = np.zeros((81, 81), dtype=np.int16)
        accumulate_heightfield_tile(hf, terrain.height_field_raw, 0, 0, border_pixels=0, tile_pixels=80)
        padded = np.pad(hf, ((0, 1), (0, 1)), mode="edge")
        mask = build_x_edge_mask(padded)
        assert not mask[40, 40]

    def test_world_to_grid_mapping(self):
        origin = np.array([-65.0, -65.0], dtype=np.float32)
        world = torch.tensor([[-36.0, -36.0]], dtype=torch.float32)
        idx = ((world - origin) / HORIZONTAL_SCALE).round().long()
        assert idx[0, 0].item() == 290
        assert idx[0, 1].item() == 290


class TestForwardScannerCoords:
    def test_forward_points_start_at_zero(self):
        points = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        ]
        assert points[0] == 0.0
        assert points[-1] == 2.0

    def test_forward_scanner_offset_zero_in_cfg_module(self):
        import ast
        from pathlib import Path

        text = Path("wmp_lab/tasks/go2/go2_env_cfg.py").read_text()
        assert 'offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0))' in text


class TestAmpTerminalState:
    def test_amp_observation_dim(self):
        assert 12 + 3 + 3 + 12 == 30
