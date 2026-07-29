"""PhysX-backed domain randomization for Go2 DirectRLEnv."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from .go2_env_cfg import Go2DomainRandCfg

logger = logging.getLogger(__name__)

# Flags for which randomization terms write through to PhysX.
PHYSX_SUPPORTED = frozenset(
    {
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_mass",
        "randomize_link_mass",
        "randomize_com_pos",
        "randomize_gains",
        "randomize_motor_strength",
        "randomize_action_latency",
        "push_robots",
    }
)


def validate_domain_rand_config(dr: Go2DomainRandCfg, *, play_mode: bool = False) -> None:
    """Warn or disable unsupported DR flags; play_mode turns all DR off."""
    if play_mode:
        return
    for name in dir(dr):
        if not name.startswith("randomize_"):
            continue
        enabled = bool(getattr(dr, name))
        if enabled and name not in PHYSX_SUPPORTED:
            logger.warning("Domain randomization %s is enabled but not implemented; ignoring.", name)


def apply_static_domain_rand(
    robot: Articulation,
    env_ids: torch.Tensor,
    dr: Go2DomainRandCfg,
    *,
    frictions: torch.Tensor,
    restitutions: torch.Tensor,
    added_masses: torch.Tensor,
    com_offsets: torch.Tensor,
    link_mass_scales: torch.Tensor | None,
    default_masses: torch.Tensor,
    default_inertia: torch.Tensor,
) -> None:
    """Write creation-time randomization (friction, mass, COM) into PhysX."""
    if len(env_ids) == 0:
        return
    cpu_ids = env_ids.cpu()
    view = robot.root_physx_view

    if dr.randomize_friction or dr.randomize_restitution:
        materials = view.get_material_properties().clone()
        for e in range(len(cpu_ids)):
            env_i = int(cpu_ids[e])
            if dr.randomize_friction:
                mu = float(frictions[env_i, 0].item())
                materials[env_i, :, 0] = mu
                materials[env_i, :, 1] = mu
            if dr.randomize_restitution:
                rest = float(restitutions[env_i, 0].item())
                materials[env_i, :, 2] = rest
        view.set_material_properties(materials, cpu_ids)

    body_ids = torch.arange(robot.num_bodies, dtype=torch.int, device="cpu")
    masses = view.get_masses().clone()
    masses[cpu_ids[:, None], body_ids] = default_masses[cpu_ids[:, None], body_ids].clone()

    if dr.randomize_base_mass:
        masses[cpu_ids, 0] += added_masses[cpu_ids, 0].cpu()

    if dr.randomize_link_mass and link_mass_scales is not None:
        n_links = link_mass_scales.shape[1]
        for b in range(1, min(robot.num_bodies, n_links + 1)):
            masses[cpu_ids, b] *= link_mass_scales[cpu_ids, b - 1].cpu()

    masses = torch.clamp(masses, min=1e-6)
    view.set_masses(masses, cpu_ids)

    if dr.randomize_base_mass or dr.randomize_link_mass:
        ratios = masses[cpu_ids[:, None], body_ids] / torch.clamp(
            default_masses[cpu_ids[:, None], body_ids], min=1e-6
        )
        inertias = view.get_inertias().clone()
        inertias[cpu_ids[:, None], body_ids] = default_inertia[cpu_ids[:, None], body_ids] * ratios[..., None]
        view.set_inertias(inertias, cpu_ids)

    if dr.randomize_com_pos:
        coms = view.get_coms().clone()
        coms[cpu_ids, 0, :3] += com_offsets[cpu_ids].cpu()
        view.set_coms(coms, cpu_ids)


def read_back_static_domain_rand(
    robot: Articulation,
    env_ids: torch.Tensor,
    device: torch.device,
    default_coms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read friction/restitution/base added mass/COM offset from PhysX for privileged obs."""
    cpu_ids = env_ids.cpu()
    view = robot.root_physx_view
    materials = view.get_material_properties()
    frictions = materials[cpu_ids, 0, 0].to(device=device).unsqueeze(-1)
    restitutions = materials[cpu_ids, 0, 2].to(device=device).unsqueeze(-1)

    masses = view.get_masses()
    default = robot.data.default_mass[cpu_ids, 0].cpu()
    added = (masses[cpu_ids, 0] - default).to(device=device).unsqueeze(-1)

    coms = view.get_coms()
    com_off = (coms[cpu_ids, 0, :3] - default_coms[cpu_ids, 0, :3].cpu()).to(device=device)
    return frictions, restitutions, added, com_off
