"""The arm as a reinforcement-learning environment, for the model-free baseline.

Each episode is a 3 s stretch of a random training scenario, started in step
with its reference. The agent's action is a torque correction on top of
computed torque with the nominal model, held for ten control steps. Its reward
is the negative tool error in millimetres, averaged over those steps, with a
small cost on the correction's size.
"""
import gymnasium as gym
import mujoco
import numpy as np

from src.control.computed_torque import ComputedTorque
from src.control.gains import second_order
from src.control.policy import AUTHORITY, REPEAT, Observer
from src.dynamics.nominal import NominalModel
from src.evaluation.scenarios import Scenario
from src.paths import ROBOT
from src.plant.arm import Arm
from src.plant.robot import build, friction, workspace
from src.plant.sensing import Sensing
from src.training.collect import FREQUENCY

LENGTH = 300  # policy steps per episode, 3 s


class ArmEnvironment(gym.Env):
    def __init__(self, seed: int):
        model = build(ROBOT)
        self.arm = Arm(model, workspace(), friction(model))
        self.base = ComputedTorque(NominalModel(build(ROBOT)), *second_order(model.nu, FREQUENCY))
        self.sensing = Sensing(model.nu, self.arm.period)
        self.observer = Observer(model.nu)
        self.rated = model.actuator_ctrlrange[:, 1]
        self.reference_data = mujoco.MjData(model)
        self.seed_base = 100_000 * (seed + 1)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.observer.size,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (model.nu,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Training scenarios only: their seeds sit far above every validation and test seed.
        scenario = Scenario(self.seed_base + int(self.np_random.integers(100_000)), "training")
        self.path, self.schedule = scenario.build(self.arm.model, self.arm.workspace)
        self.time = float(self.np_random.uniform(0, self.path.duration - LENGTH * REPEAT * self.arm.period))
        start = self.path.at(self.time)
        self.arm.reset(start.angle, start.velocity)
        self.sensing.reset(start.angle, start.velocity)
        self.base.reset()
        self.observer.reset()
        self.applied, self.steps = np.zeros_like(self.rated), 0
        self.last = (start, self.sensing.read(self.arm.angle))
        return self._observe(), {}

    def step(self, action):
        correction = AUTHORITY * self.rated * np.asarray(action)
        self.observer.action = np.asarray(action)
        errors = []
        for _ in range(REPEAT):
            reference = self.path.at(self.time)
            self.arm.apply(self.schedule.at(self.time, reference.holding))
            measured = self.sensing.read(self.arm.angle)
            self.base.learn(measured, self.applied)
            self.applied = self.arm.step(self.base(reference, measured) + correction)
            self.time += self.arm.period
            self.reference_data.qpos[:] = reference.angle
            mujoco.mj_kinematics(self.arm.model, self.reference_data)
            errors.append(1e3 * np.linalg.norm(self.arm.tool_position - self.reference_data.site_xpos[self.arm.tool]))
            self.last = (reference, measured)
        self.steps += 1
        reward = -np.mean(errors) / 10 - 0.01 * float(np.sum(np.square(action)))
        return self._observe(), reward, False, self.steps >= LENGTH, {}

    def _observe(self) -> np.ndarray:
        return self.observer(*self.last)
