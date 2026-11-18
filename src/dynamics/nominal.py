"""The arm as its datasheet describes it: rigid-body dynamics with no payload,
no wear and no friction beyond what the model file states.

MuJoCo's inverse dynamics gives the joint torque that produces a requested
acceleration at a given state, so no equations of motion are written by hand.
"""
import mujoco
import numpy as np


class NominalModel:
    def __init__(self, model: mujoco.MjModel):
        self.model = model  # its own copy, built from the robot file, never changed
        self.data = mujoco.MjData(self.model)

    def torque(self, angle: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
        self.data.qpos[:], self.data.qvel[:], self.data.qacc[:] = angle, velocity, acceleration
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()
