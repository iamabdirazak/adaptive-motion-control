"""The simulated arm: a MuJoCo model whose payload, friction and motors change.

The model comes from src.plant.robot. Its joints are driven by torque motors with unit gear, so a control
command is a joint torque in N·m. A payload body at the flange carries the held
part, wear lowers the motors' gear, and friction is applied as a joint force.
The controller never reads these settings, only the sensed joint angles.
"""
import mujoco
import numpy as np

from src.plant.changes import NOMINAL, Conditions
from src.plant.friction import Friction
from src.paths import CONTROL_PERIOD
from src.plant.workspace import Workspace


class Arm:
    def __init__(self, model: mujoco.MjModel, workspace: Workspace, friction: Friction | None = None,
                 period: float = CONTROL_PERIOD, payload_body: str = "payload", tool_site: str = "tool"):
        self.model, self.workspace = model, workspace
        self.data = mujoco.MjData(self.model)
        self.friction = friction
        self.substeps = round(period / self.model.opt.timestep)
        self.period = self.substeps * self.model.opt.timestep
        self.payload = self.model.body(payload_body).id
        self.tool = self.model.site(tool_site).id
        self.limit = self.model.actuator_ctrlrange[:, 1].copy()
        self.apply(NOMINAL)

    @property
    def joints(self) -> int:
        return self.model.nu

    def apply(self, conditions: Conditions) -> None:
        """Set the payload, friction and wear the arm runs under from now on."""
        mass = max(conditions.payload, 1e-6)  # MuJoCo needs a positive mass
        self.model.body_mass[self.payload] = mass
        self.model.body_inertia[self.payload] = 0.4 * mass * 0.03**2  # a 3 cm solid sphere
        self.model.actuator_gear[:, 0] = 1.0 - conditions.wear
        self.conditions = conditions

    def reset(self, angle: np.ndarray, velocity: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = angle
        if velocity is not None:
            self.data.qvel[:] = velocity
        mujoco.mj_forward(self.model, self.data)

    def step(self, torque: np.ndarray) -> np.ndarray:
        """Apply a torque command for one control period. Returns the command after limits."""
        command = np.clip(torque, -self.limit, self.limit)
        self.data.ctrl[:] = command
        for _ in range(self.substeps):
            if self.friction is not None:
                self.data.qfrc_applied[:] = self.friction.torque(self.data.qvel, self.conditions.friction)
            mujoco.mj_step(self.model, self.data)
        return command

    @property
    def angle(self) -> np.ndarray:
        return self.data.qpos.copy()

    @property
    def tool_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tool].copy()
