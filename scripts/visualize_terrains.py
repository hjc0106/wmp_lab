#!/usr/bin/env python3
"""Visualize the legacy WMP terrain generator without loading a robot or policy."""
from __future__ import annotations

import argparse

from _common import add_common_args, bootstrap_paths, launch_isaac_app


def _camera_pose(rows: int, cols: int) -> tuple[list[float], list[float]]:
    extent_x = rows * 8.0
    extent_y = cols * 8.0
    span = max(extent_x, extent_y)
    eye = [0.55 * extent_x, -0.75 * extent_y, max(12.0, 0.7 * span)]
    return eye, [0.0, 0.0, 0.0]


def visualize(args, simulation_app) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

    from wmp_lab.tasks.go2.go2_env_cfg import GO2_TERRAIN_NAMES, _build_go2_terrain_generator
    from wmp_lab.tasks.go2.legacy_terrain_layout import column_category_names

    selected = args.terrain != "all"
    rows = args.rows if args.rows is not None else (4 if selected else 10)
    cols = args.cols if args.cols is not None else (1 if selected else 20)
    if rows < 1 or cols < 1:
        raise ValueError("--rows and --cols must be positive")

    proportions = None
    if selected:
        proportions = {name: 0.0 for name in GO2_TERRAIN_NAMES}
        proportions[args.terrain] = 1.0

    generator_cfg = _build_go2_terrain_generator(
        num_rows=rows,
        num_cols=cols,
        curriculum=False,
        proportions=proportions,
        seed=args.seed,
        compat_mode=args.compat,
        slope_direction=args.slope_direction,
    )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 60.0,
            device=args.device,
            gravity=(0.0, 0.0, 0.0),
        )
    )
    eye, target = _camera_pose(rows, cols)
    if args.camera_eye is not None:
        eye = args.camera_eye
    if args.camera_target is not None:
        target = args.camera_target
    sim.set_camera_view(eye=eye, target=target)

    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        num_envs=1,
        terrain_type="generator",
        terrain_generator=generator_cfg,
        use_terrain_origins=True,
        max_init_terrain_level=0,
        debug_vis=args.show_origins,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.24, 0.36, 0.16),
            roughness=0.86,
            metallic=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporter(terrain_cfg)
    sim.reset()

    column_names = column_category_names(cols, generator_cfg.terrain_proportions)
    print(
        f"[terrain-viewer] terrain={args.terrain} mode={args.compat} "
        f"seed={args.seed} grid={rows}x{cols}",
        flush=True,
    )
    print(f"[terrain-viewer] row difficulties: {[row / rows for row in range(rows)]}", flush=True)
    print(f"[terrain-viewer] column categories: {column_names}", flush=True)
    print("[terrain-viewer] close the viewer window or press Ctrl+C to exit", flush=True)

    # Keep a live reference to the importer while the viewer is open.
    _ = terrain
    step = 0
    while simulation_app.is_running() and (args.steps <= 0 or step < args.steps):
        sim.step()
        step += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(headless=False, seed=1)
    parser.add_argument(
        "--terrain",
        default="all",
        choices=[
            "all", "wave", "slope", "stairs_up", "stairs_down", "discrete",
            "gap", "climb", "tilt", "crawl", "rough_flat",
        ],
        help="Show the complete distribution or repeat one terrain category",
    )
    parser.add_argument("--rows", type=int, default=None, help="Defaults to 10 for all, 4 for one category")
    parser.add_argument("--cols", type=int, default=None, help="Defaults to 20 for all, 1 for one category")
    parser.add_argument(
        "--compat",
        default="legacy_exact",
        choices=["legacy_exact", "legacy_fixed"],
    )
    parser.add_argument(
        "--slope_direction",
        default="legacy",
        choices=["legacy", "up", "down"],
    )
    parser.add_argument("--show_origins", action="store_true", help="Draw each tile origin frame")
    parser.add_argument("--steps", type=int, default=0, help="Exit after N frames; 0 keeps the viewer open")
    parser.add_argument("--camera_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera_target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    args = parser.parse_args()

    bootstrap_paths()
    app_launcher = launch_isaac_app(args)
    simulation_app = app_launcher.app
    try:
        visualize(args, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
