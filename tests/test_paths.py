"""Asset path resolution tests."""
from __future__ import annotations

from wmp_lab.paths import dataset_path, lab_root, resource_path


def test_lab_root_has_assets():
    root = lab_root()
    assert (root / "resources" / "robots" / "go2" / "urdf" / "go2.urdf").is_file()
    assert (root / "datasets" / "mocap_motions" / "trot1.txt").is_file()


def test_helpers():
    assert resource_path("robots", "go2", "urdf", "go2.urdf").is_file()
    assert dataset_path("mocap_motions", "trot1.txt").is_file()
