"""Builds the simulated robot from its description file, which is never edited.

The robot's file describes the arm as its maker publishes it: links, joints,
masses, inertias and limits. This module adds what a torque-controlled study
needs on top, in code: a torque motor on every joint, limited to the joint's
rated torque, a payload body at the flange, a tool site, and the simulation
settings. The arm moves in free space, so contacts are switched off.
"""
from pathlib import Path

import mujoco
import numpy as np

from src.plant.friction import Friction
from src.plant.workspace import Workspace
from src.paths import SIMULATION_STEP


def build(path: Path, flange: str = "flange") -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(path))
    spec.option.timestep = SIMULATION_STEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    site = spec.site(flange)
    payload = site.parent.add_body(name="payload", pos=site.pos, quat=site.quat)
    payload.mass, payload.inertia = 1e-6, [1e-9, 1e-9, 1e-9]  # empty until a part is held
    payload.add_site(name="tool")

    for joint in spec.joints:
        limit = joint.actfrcrange[1]
        spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          ctrllimited=True, ctrlrange=[-limit, limit])
    return spec.compile()


def friction(model: mujoco.MjModel) -> Friction:
    """Base joint friction, as shares of each joint's rated torque.

    The maker does not publish friction, so typical harmonic-drive values are
    used: sliding friction 2% of rated torque, breakaway 3%, and viscous
    friction 1% of rated torque per rad/s.
    """
    rated = model.actuator_ctrlrange[:, 1]
    return Friction(coulomb=0.02 * rated, static=0.03 * rated, viscous=0.01 * rated)


def workspace() -> Workspace:
    """The LBR iisy 8's working region: around a ready pose with the tool forward
    and down, with the rated joint speeds from the maker's description."""
    return Workspace(
        home=np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0]),
        span=np.array([1.5, 0.5, 0.7, 1.0, 0.8, 1.5]),
        velocity=np.array([3.49, 3.49, 3.49, 4.01, 4.54, 7.50]),
    )
