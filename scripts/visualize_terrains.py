#!/usr/bin/env python3
"""Visualize the legacy WMP terrain generator without loading a robot or policy.

Uses the same terrain presets and generator settings as ``play_wmp.py`` /
``Go2AmpLabCfg`` (``PLAY_TERRAIN_PRESETS``, ``_build_go2_terrain_generator``).
"""
from __future__ import annotations

import argparse

from _common import (
    add_common_args,
    apply_distributed_device,
    bootstrap_paths,
    launch_isaac_app,
    resolve_distributed_flag,
)


from wmp_lab.tasks.go2.legacy_terrain_layout import (
    LEGACY_TERRAIN_NAMES,
    PLAY_TERRAIN_PRESETS,
    normalize_terrain_name,
)


def _terrain_choice_list() -> list[str]:
    choices: list[str] = ["all"]
    for name in sorted(PLAY_TERRAIN_PRESETS.keys()):
        if name not in choices:
            choices.append(name)
    for name in LEGACY_TERRAIN_NAMES:
        if name not in choices:
            choices.append(name)
    return choices


def _resolve_terrain_request(args) -> tuple[dict | None, int, int, str, str]:
    """Return (proportions, rows, cols, mesh_type, canonical_name_for_log)."""
    key = args.terrain
    default_rows = 10
    default_cols = 20

    if key == "all":
        rows = args.rows if args.rows is not None else default_rows
        cols = args.cols if args.cols is not None else default_cols
        return None, rows, cols, "generator", "all"

    if key == "plane":
        return None, 1, 1, "plane", "plane"

    if key in PLAY_TERRAIN_PRESETS:
        mapped = PLAY_TERRAIN_PRESETS[key]
        if mapped is None:
            return None, 1, 1, "plane", "plane"
        canonical = mapped
    else:
        canonical = normalize_terrain_name(key)
        if canonical not in LEGACY_TERRAIN_NAMES:
            raise ValueError(
                f"Unknown terrain {key!r}; expected one of {_terrain_choice_list()}"
            )

    rows = args.rows if args.rows is not None else default_rows
    cols = args.cols if args.cols is not None else default_cols
    proportions = {name: 0.0 for name in LEGACY_TERRAIN_NAMES}
    proportions[canonical] = 1.0
    return proportions, rows, cols, "generator", canonical


def _camera_pose(rows: int, cols: int) -> tuple[list[float], list[float]]:
    extent_x = rows * 8.0
    extent_y = cols * 8.0
    span = max(extent_x, extent_y)
    eye = [0.55 * extent_x, -0.75 * extent_y, max(12.0, 0.7 * span)]
    return eye, [0.0, 0.0, 0.0]


def _spawn_diagnostic_grid(sim_utils, rows: int, cols: int) -> None:
    extent_x = rows * 8.0
    extent_y = cols * 8.0
    material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.055, 0.06, 0.065),
        roughness=0.9,
        metallic=0.0,
    )
    thickness = 0.035
    height = 0.018
    x_line = sim_utils.CuboidCfg(size=(thickness, extent_y, height), visual_material=material)
    y_line = sim_utils.CuboidCfg(size=(extent_x, thickness, height), visual_material=material)
    for row in range(rows + 1):
        x = -extent_x / 2.0 + row * 8.0
        x_line.func(f"/World/Diagnostics/GridX_{row}", x_line, translation=(x, 0.0, 0.02))
    for col in range(cols + 1):
        y = -extent_y / 2.0 + col * 8.0
        y_line.func(f"/World/Diagnostics/GridY_{col}", y_line, translation=(0.0, y, 0.02))


def visualize(args, simulation_app) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

    from wmp_lab.tasks.go2.go2_env_cfg import _build_go2_terrain_generator
    from wmp_lab.tasks.go2.legacy_terrain_layout import TERRAIN_CATEGORY_COLORS, column_category_names

    proportions, rows, cols, mesh_type, canonical = _resolve_terrain_request(args)
    if rows < 1 or cols < 1:
        raise ValueError("--rows/--terrain_rows and --cols/--terrain_cols must be positive")

    generator_cfg = None
    if mesh_type == "generator":
        generator_cfg = _build_go2_terrain_generator(
            num_rows=rows,
            num_cols=cols,
            curriculum=args.curriculum,
            proportions=proportions,
            seed=args.seed,
            compat_mode=args.compat,
            slope_direction=args.slope_direction,
        )
        generator_cfg.border_width = 25.0 if args.show_border else 0.0
        generator_cfg.color_by_category = not args.uniform_color

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

    dome_light = sim_utils.DomeLightCfg(intensity=3200.0, color=(0.82, 0.84, 0.86))
    dome_light.func("/World/Lighting/Dome", dome_light)
    key_light = sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.92, 0.8), angle=2.0)
    key_light.func("/World/Lighting/Key", key_light, translation=(0.0, 0.0, 20.0))

    if mesh_type == "plane":
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            num_envs=1,
            terrain_type="plane",
            use_terrain_origins=False,
            debug_vis=False,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.43, 0.44, 0.41), roughness=0.82, metallic=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
    else:
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            num_envs=1,
            terrain_type="generator",
            terrain_generator=generator_cfg,
            use_terrain_origins=True,
            max_init_terrain_level=rows - 1 if args.curriculum else 0,
            debug_vis=args.show_origins,
            visual_material=None
            if not args.uniform_color
            else sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.43, 0.44, 0.41), roughness=0.82, metallic=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
    terrain = TerrainImporter(terrain_cfg)
    if mesh_type == "generator" and not args.no_grid:
        _spawn_diagnostic_grid(sim_utils, rows, cols)
    sim.reset()

    play_key = args.terrain if args.terrain in {"slope", "stair", "gap", "climb", "tilt", "crawl", "plane"} else None
    print(
        f"[terrain-viewer] request={args.terrain} canonical={canonical} "
        f"mesh={mesh_type} mode={args.compat} seed={args.seed} grid={rows}x{cols} "
        f"curriculum={args.curriculum}",
        flush=True,
    )
    if mesh_type == "generator":
        column_names = column_category_names(cols, generator_cfg.terrain_proportions)
        print(
            f"[terrain-viewer] row difficulties: "
            f"{[row / rows for row in range(rows)]}",
            flush=True,
        )
        print(f"[terrain-viewer] column categories: {column_names}", flush=True)
        if not args.uniform_color:
            visible = dict.fromkeys(column_names)
            legend = {name: TERRAIN_CATEGORY_COLORS[name][:3] for name in visible}
            print(f"[terrain-viewer] category RGB legend: {legend}", flush=True)
        print(
            f"[terrain-viewer] border={'shown' if args.show_border else 'hidden'} "
            f"tile_grid={'hidden' if args.no_grid else 'shown'} "
            f"origins={'shown' if args.show_origins else 'hidden'}",
            flush=True,
        )
    if play_key is not None:
        print(
            f"[terrain-viewer] matching play command: "
            f"python scripts/play_wmp.py --terrain {play_key} "
            f"--terrain_rows {rows} --terrain_cols {cols} "
            f"--terrain_compat {args.compat} --terrain_seed {args.seed}",
            flush=True,
        )
    print("[terrain-viewer] close the viewer window or press Ctrl+C to exit", flush=True)

    _ = terrain
    step = 0
    while simulation_app.is_running() and (args.steps <= 0 or step < args.steps):
        sim.step()
        step += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/visualize_terrains.py --terrain all\n"
            "  python scripts/visualize_terrains.py --terrain climb --show_border\n"
            "  python scripts/visualize_terrains.py --terrain stair --compat legacy_fixed\n"
            "  python scripts/visualize_terrains.py --terrain plane\n"
            "\n"
            "Play presets (same names as play_wmp.py): "
            "slope, stair, gap, climb, tilt, crawl, plane\n"
            "Full category names: wave, slope, stairs_up, stairs_down, discrete, "
            "gap, climb, tilt, crawl, rough_flat\n"
        ),
    )
    add_common_args(parser)
    parser.set_defaults(headless=False, seed=1, num_envs=1)
    parser.add_argument(
        "--terrain",
        default="all",
        choices=_terrain_choice_list(),
        help="Training distribution (all), a play preset, or one legacy category",
    )
    parser.add_argument(
        "--rows",
        "--terrain_rows",
        dest="rows",
        type=int,
        default=None,
        help="Grid rows (default: 10; matches play_wmp / Go2AmpLabCfg)",
    )
    parser.add_argument(
        "--cols",
        "--terrain_cols",
        dest="cols",
        type=int,
        default=None,
        help="Grid columns (default: 20 for all/single-category views)",
    )
    parser.add_argument(
        "--compat",
        "--terrain_compat",
        dest="compat",
        default="legacy_exact",
        choices=["legacy_exact", "legacy_fixed"],
        help="legacy_exact reproduces old crawl overlap; legacy_fixed symmetric crawl bars",
    )
    parser.add_argument(
        "--slope_direction",
        default="legacy",
        choices=["legacy", "up", "down"],
        help="Slope sign override (legacy splits slope columns by direction)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable row difficulty curriculum (training layout)",
    )
    parser.add_argument("--show_origins", action="store_true", help="Draw each tile origin frame")
    parser.add_argument("--show_border", action="store_true", help="Show the 25 m training border")
    parser.add_argument("--no_grid", action="store_true", help="Hide the 8 m tile boundary grid")
    parser.add_argument("--uniform_color", action="store_true", help="Use one neutral gray material")
    parser.add_argument("--steps", type=int, default=0, help="Exit after N frames; 0 keeps the viewer open")
    parser.add_argument("--camera_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera_target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    args = parser.parse_args()

    bootstrap_paths()
    resolve_distributed_flag(args)
    args.distributed = False

    app_launcher = launch_isaac_app(args)
    simulation_app = app_launcher.app
    try:
        args = apply_distributed_device(args, app_launcher)
        visualize(args, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
