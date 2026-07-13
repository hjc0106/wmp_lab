"""Curriculum difficulty ordering tests (no Isaac Sim)."""
from __future__ import annotations

from wmp_lab.tasks.go2.legacy_terrain_layout import ordered_row_difficulty


def test_ordered_difficulties_span_unit_interval():
    diffs = [ordered_row_difficulty(r, 10) for r in range(10)]
    assert diffs[0] == 0.0
    assert diffs[-1] == 0.9
    assert len(set(round(d, 1) for d in diffs)) == 10
