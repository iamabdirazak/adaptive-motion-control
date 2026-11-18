"""The model-free baseline: a policy, trained by reinforcement learning, that adds a
torque correction on top of computed torque with the nominal model.

The policy acts at 100 Hz and its correction is held for ten 1 kHz control steps,
the usual split between a learned outer loop and a fast servo loop. It sees the
joint tracking errors, a short history of them, the joint angles and the reference
motion. It never sees the payload, friction or wear, only their effects.
"""
from collections import deque
from pathlib import Path

import numpy as np

from src.plant.sensing import Measurement
from src.plant.trajectories import Reference

REPEAT = 10        # control steps per policy step
AUTHORITY = 0.10   # largest correction, as a share of each joint's rated torque
HISTORY = 2        # earlier policy steps of tracking error the policy sees


class Observer:
    """Builds the policy's observation, the same in training and in evaluation."""

    def __init__(self, joints: int):
        self.joints = joints
        self.reset()

    def reset(self) -> None:
        self.errors = deque([np.zeros(self.joints)] * HISTORY, maxlen=HISTORY)
        self.action = np.zeros(self.joints)

    def __call__(self, reference: Reference, measured: Measurement) -> np.ndarray:
        error = reference.angle - measured.angle
        observation = np.concatenate([
            100 * error, 10 * (reference.velocity - measured.velocity),
            np.sin(measured.angle), np.cos(measured.angle),
            reference.velocity, reference.acceleration / 10,
            self.action, *(100 * past for past in self.errors),
        ]).astype(np.float32)
        self.errors.append(error)
        return observation

    @property
    def size(self) -> int:
        return self.joints * (7 + HISTORY)


class ResidualPolicy:
    def __init__(self, base, path: Path):
        from stable_baselines3 import SAC

        self.base = base
        self.policy = SAC.load(str(path), device="cpu")
        self.rated = base.nominal.model.actuator_ctrlrange[:, 1]
        self.observer = Observer(base.nominal.model.nu)

    def reset(self) -> None:
        self.base.reset()
        self.observer.reset()
        self.count, self.correction = 0, np.zeros_like(self.rated)

    def __call__(self, reference: Reference, measured: Measurement) -> np.ndarray:
        if self.count % REPEAT == 0:
            action, _ = self.policy.predict(self.observer(reference, measured), deterministic=True)
            self.observer.action = action
            self.correction = AUTHORITY * self.rated * action
        self.count += 1
        return self.base(reference, measured) + self.correction

    def learn(self, measured: Measurement, applied: np.ndarray) -> None:
        self.base.learn(measured, applied)
