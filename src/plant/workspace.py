"""Where the arm works: a home pose, how far each joint may swing from it,
each joint's rated speed, and where the tool may go.

Poses are drawn around home and kept only if the tool stays above the mounting
surface and clear of the arm's own base, the region a real cell would allow.
"""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Workspace:
    home: np.ndarray      # rad
    span: np.ndarray      # rad, largest swing from home per joint
    velocity: np.ndarray  # rad/s, rated speed per joint
    floor: float = 0.10   # m, lowest tool height
    clearance: float = 0.30  # m, closest tool distance to the base axis

    def sample(self, rng: np.random.Generator, model: mujoco.MjModel, count: int) -> np.ndarray:
        data, tool = mujoco.MjData(model), model.site("tool").id
        poses = []
        while len(poses) < count:
            pose = np.clip(self.home + rng.uniform(-1, 1, len(self.home)) * self.span,
                           model.jnt_range[:, 0], model.jnt_range[:, 1])
            data.qpos[:] = pose
            mujoco.mj_kinematics(model, data)
            x, y, z = data.site_xpos[tool]
            if z > self.floor and np.hypot(x, y) > self.clearance:
                poses.append(pose)
        return np.array(poses)
