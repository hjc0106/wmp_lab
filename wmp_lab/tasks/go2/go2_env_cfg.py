from __future__ import annotations

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    RayCasterCfg,
    RayCasterCameraCfg,
)
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import (
    PatternBaseCfg,
    PinholeCameraPatternCfg,
)
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .legacy_terrain_generator import LegacyTerrainGenerator, LegacyTerrainGeneratorCfg
from .legacy_terrain_layout import DEFAULT_TERRAIN_PROPORTIONS, LEGACY_TERRAIN_NAMES, normalize_terrain_name


WMP_LAB_ROOT = Path(__file__).resolve().parents[3]
GO2_URDF = WMP_LAB_ROOT / "resources" / "robots" / "go2" / "urdf" / "go2.urdf"
USD_DIR = WMP_LAB_ROOT / "generated" / "usd"
TERRAIN_PRIM = "/World/ground"

# Non-uniform height sample point sets (module-level so they can be reused by
# the ray-cast pattern builders without relying on configclass introspection).
MEASURED_POINTS_X = [
    -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3,
    0.4, 0.5, 0.6, 0.7, 0.8,
]
MEASURED_POINTS_Y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MEASURED_FORWARD_POINTS_X = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
]
MEASURED_FORWARD_POINTS_Y = [
    -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
    -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
]


# --------------------------------------------------------------------------- #
# Custom ray-cast patterns for the legacy non-uniform height sample points.
# --------------------------------------------------------------------------- #
def points_pattern(cfg: PatternBaseCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    pts = torch.as_tensor(cfg.points, device=device, dtype=torch.float32)  # (N, 2)
    starts = torch.zeros(pts.shape[0], 3, device=device)
    starts[:, :2] = pts
    dirs = torch.zeros(pts.shape[0], 3, device=device)
    dirs[:, 2] = -1.0
    return starts, dirs


@configclass
class PointsPatternCfg(PatternBaseCfg):
    func = points_pattern
    points: tuple = ()


# --------------------------------------------------------------------------- #
# Legacy-style meta configs (kept as nested dataclasses so the unchanged
# ``rsl_rl.WMPRunner`` can read ``self.cfg.env.*/terrain_meta.*/depth.*/...``).
# --------------------------------------------------------------------------- #
@configclass
class Go2DepthCfg:
    use_camera: bool = False
    camera_num_envs: int = 128
    camera_terrain_num_rows: int = 10
    camera_terrain_num_cols: int = 20
    position: list = [0.33, 0.0, 0.08]
    y_angle: list = [-5.0, 5.0]
    z_angle: list = [0.0, 0.0]
    x_angle: list = [0.0, 0.0]
    update_interval: int = 5
    original: tuple = (64, 64)
    resized: tuple = (64, 64)
    horizontal_fov: int = 58
    buffer_len: int = 2
    near_clip: float = 0.0
    far_clip: float = 2.0
    dis_noise: float = 0.0
    scale: int = 1
    invert: bool = True


@configclass
class Go2EnvMetaCfg:
    num_envs: int = 4
    include_history_steps = None
    prop_dim: int = 33
    action_dim: int = 12
    privileged_dim: int = 24 + 26 + 3
    height_dim: int = 187
    forward_height_dim: int = 525
    num_observations: int = 33 + (24 + 26 + 3) + 187 + 12
    num_privileged_obs: int = 33 + (24 + 26 + 3) + 187 + 12
    num_actions: int = 12
    privileged_obs: bool = True
    env_spacing: float = 3.0
    send_timeouts: bool = True
    episode_length_s: float = 20.0
    reference_state_initialization: bool = False
    reference_state_initialization_prob: float = 0.85
    amp_motion_files: list = [
        "datasets/mocap_motions/hop1.txt",
        "datasets/mocap_motions/hop2.txt",
        "datasets/mocap_motions/trot1.txt",
        "datasets/mocap_motions/trot2.txt",
    ]


@configclass
class Go2TerrainMetaCfg:
    mesh_type: str = "plane"
    measure_heights: bool = True
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_size: float = 25.0
    curriculum: bool = False
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    measured_points_x: list = [
        -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7, 0.8,
    ]
    measured_points_y: list = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    measured_forward_points_x: list = [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    ]
    measured_forward_points_y: list = [
        -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
        -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    ]
    num_rows: int = 10
    num_cols: int = 20
    # proportions ordered: wave, slope, stairs_up, stairs_down, discrete, gap,
    # climb, tilt, crawl, rough_flat
    terrain_proportions: list = list(DEFAULT_TERRAIN_PROPORTIONS)
    slope_treshold: float = 0.75
    terrain_length: float = 8.0
    terrain_width: float = 8.0
    max_init_terrain_level: int = 0


@configclass
class Go2CommandRangesCfg:
    lin_vel_x: list = [0.0, 0.8]
    lin_vel_y: list = [0.0, 0.0]
    ang_vel_yaw: list = [-1.0, 1.0]
    heading: list = [0.0, 0.0]
    flat_lin_vel_x: list = [0.0, 0.8]
    flat_lin_vel_y: list = [0.0, 0.0]
    flat_ang_vel_yaw: list = [-1.0, 1.0]
    flat_heading: list = [-0.785398163, 0.785398163]


@configclass
class Go2CommandCfg:
    curriculum: bool = False
    num_commands: int = 4
    resampling_time: float = 10.0
    heading_command: bool = True
    ranges: Go2CommandRangesCfg = Go2CommandRangesCfg()


@configclass
class Go2ControlCfg:
    control_type: str = "P"
    stiffness: dict = {"joint": 40.0}
    damping: dict = {"joint": 1.0}
    action_scale: float = 0.25
    decimation: int = 4


@configclass
class Go2ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    height_measurements: float = 5.0
    contact_force: float = 0.005
    com_pos: float = 20.0
    pd_gains: float = 5.0


@configclass
class Go2NormalizationCfg:
    obs_scales: Go2ObsScalesCfg = Go2ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 6.0
    base_height: float = 0.35


@configclass
class Go2RewardsScalesCfg:
    tracking_lin_vel: float = 1.5
    tracking_ang_vel: float = 0.5
    lin_vel_z: float = -1.0
    ang_vel_xy: float = 0.0
    torques: float = -0.0001
    dof_acc: float = -2.5e-7
    action_rate: float = -0.03
    collision: float = -1.0
    feet_air_time: float = 0.5
    feet_stumble: float = -0.1
    feet_edge: float = -1.0
    dof_error: float = -0.04
    cheat: float = -1.0
    stuck: float = -1.0


@configclass
class Go2RewardsMetaCfg:
    reward_curriculum: bool = True
    reward_curriculum_term: list = ["feet_edge"]
    reward_curriculum_schedule: list = [[4000, 10000, 0.1, 1.0]]
    tracking_sigma: float = 0.15
    base_height_target: float = 0.30
    foot_height_target: float = 0.15
    lin_vel_clip: float = 0.1
    soft_dof_pos_limit: float = 0.9
    soft_dof_vel_limit: float = 0.9
    soft_torque_limit: float = 1.0
    max_contact_force: float = 0.0
    scales: Go2RewardsScalesCfg = Go2RewardsScalesCfg()


@configclass
class Go2DomainRandCfg:
    randomize_friction: bool = True
    friction_range: list = [0.5, 2.0]
    randomize_restitution: bool = True
    restitution_range: list = [0.0, 0.0]
    randomize_base_mass: bool = True
    added_mass_range: list = [0.0, 5.0]
    randomize_link_mass: bool = True
    link_mass_range: list = [0.8, 1.2]
    randomize_com_pos: bool = True
    com_x_pos_range: list = [-0.05, 0.05]
    com_y_pos_range: list = [-0.05, 0.05]
    com_z_pos_range: list = [-0.05, 0.05]
    push_robots: bool = True
    push_interval_s: float = 15.0
    min_push_interval_s: float = 15.0
    max_push_vel_xy: float = 1.0
    randomize_gains: bool = True
    stiffness_multiplier_range: list = [0.8, 1.2]
    damping_multiplier_range: list = [0.8, 1.2]
    randomize_motor_strength: bool = True
    motor_strength_range: list = [0.8, 1.2]
    randomize_action_latency: bool = True
    latency_range: list = [0.0, 0.005]


@configclass
class Go2NoiseCfg:
    add_noise: bool = False
    noise_level: float = 1.0


# Name order matches legacy WMP terrain_proportions indexing.
GO2_TERRAIN_NAMES = LEGACY_TERRAIN_NAMES

# Play presets mirror legacy play.py terrain_proportions.
PLAY_TERRAIN_PRESETS = {
    "slope": "slope",
    "stair": "stairs_up",
    "gap": "gap",
    "climb": "climb",
    "tilt": "tilt",
    "crawl": "crawl",
    "plane": None,
}


def _build_go2_terrain_generator(
    num_rows: int,
    num_cols: int,
    curriculum: bool,
    proportions: dict | None = None,
    seed: int | None = None,
    compat_mode: str = "legacy_exact",
    slope_direction: str = "legacy",
) -> LegacyTerrainGeneratorCfg:
    """Legacy-compatible terrain generator for the WMP 10-type curriculum."""
    default = {name: float(DEFAULT_TERRAIN_PROPORTIONS[i]) for i, name in enumerate(GO2_TERRAIN_NAMES)}
    if proportions is not None:
        normalized = {normalize_terrain_name(name): float(value) for name, value in proportions.items()}
        default = {name: normalized.get(name, 0.0) for name in GO2_TERRAIN_NAMES}
    return LegacyTerrainGeneratorCfg(
        class_type=LegacyTerrainGenerator,
        seed=seed,
        curriculum=curriculum,
        ordered_generation=True,
        compat_mode=compat_mode,
        slope_direction=slope_direction,
        terrain_proportions=[default[n] for n in GO2_TERRAIN_NAMES],
        size=(8.0, 8.0),
        border_width=25.0,
        # A one-meter-deep border leaves its collision top at z=0.
        border_height=1.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={},
    )


def _measured_points_xy() -> list:
    pts = []
    for y in MEASURED_POINTS_Y:
        for x in MEASURED_POINTS_X:
            pts.append((x, y))
    return pts


def _forward_points_xy() -> list:
    pts = []
    for y in MEASURED_FORWARD_POINTS_Y:
        for x in MEASURED_FORWARD_POINTS_X:
            pts.append((x, y))
    return pts


# --------------------------------------------------------------------------- #
# Main env config.
# --------------------------------------------------------------------------- #
@configclass
class Go2LabCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_space = 12
    observation_space = 33 + (24 + 26 + 3) + 187 + 12
    state_space = 33 + (24 + 26 + 3) + 187 + 12
    is_finite_horizon = False

    env: Go2EnvMetaCfg = Go2EnvMetaCfg()
    terrain_meta: Go2TerrainMetaCfg = Go2TerrainMetaCfg()
    commands: Go2CommandCfg = Go2CommandCfg()
    control: Go2ControlCfg = Go2ControlCfg()
    depth: Go2DepthCfg = Go2DepthCfg()
    normalization: Go2NormalizationCfg = Go2NormalizationCfg()
    rewards: Go2RewardsMetaCfg = Go2RewardsMetaCfg()
    domain_rand: Go2DomainRandCfg = Go2DomainRandCfg()
    noise: Go2NoiseCfg = Go2NoiseCfg()

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path=TERRAIN_PRIM,
        terrain_type="plane",
        collision_group=-1,
        use_terrain_origins=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(GO2_URDF),
            usd_dir=str(USD_DIR),
            usd_file_name="go2_wmp_lab.usd",
            force_usd_conversion=False,
            fix_base=False,
            merge_fixed_joints=True,
            replace_cylinders_with_capsules=True,
            self_collision=False,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=40.0, damping=1.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                "FL_hip_joint": 0.1,
                "RL_hip_joint": 0.1,
                "FR_hip_joint": -0.1,
                "RR_hip_joint": -0.1,
                "FL_thigh_joint": 0.8,
                "RL_thigh_joint": 1.0,
                "FR_thigh_joint": 0.8,
                "RR_thigh_joint": 1.0,
                "FL_calf_joint": -1.5,
                "RL_calf_joint": -1.5,
                "FR_calf_joint": -1.5,
                "RR_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "base_legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit_sim=23.5,
                velocity_limit_sim=30.0,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        force_threshold=1.0,
    )

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=PointsPatternCfg(points=_measured_points_xy()),
        debug_vis=False,
        mesh_prim_paths=[TERRAIN_PRIM],
    )

    forward_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=PointsPatternCfg(points=_forward_points_xy()),
        debug_vis=False,
        mesh_prim_paths=[TERRAIN_PRIM],
    )

    # Depth camera (Optional, expected proportional; the WMP depth predictor
    # linearizes this from forward_height_map). Disabled by isaaclab at run
    # time when RTX/tiled sensors are not requested.
    depth_camera: RayCasterCameraCfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.33, 0.0, 0.08),
            rot=(0.7071, 0.0, 0.7071, 0.0),
            convention="ros",
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=2.0,
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=18.9,
            horizontal_aperture=20.955,
            width=64,
            height=64,
        ),
        debug_vis=False,
        mesh_prim_paths=[TERRAIN_PRIM],
    )


@configclass
class Go2RoughLabCfg(Go2LabCfg):
    """Simpler 'rough' task: flat plane, no depth camera, no curriculum."""

    def __post_init__(self):
        self.depth.use_camera = False
        self.terrain_meta.mesh_type = "plane"
        self.terrain_meta.curriculum = False
        self.terrain.use_terrain_origins = False


@configclass
class Go2AmpLabCfg(Go2LabCfg):
    def __post_init__(self):
        self.depth.use_camera = True
        self.terrain_meta.mesh_type = "trimesh"
        self.terrain_meta.curriculum = True
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = _build_go2_terrain_generator(
            num_rows=self.terrain_meta.num_rows,
            num_cols=self.terrain_meta.num_cols,
            curriculum=True,
        )
        self.terrain.use_terrain_origins = True
        self.terrain.max_init_terrain_level = self.terrain_meta.max_init_terrain_level
