"""Independent PID on each joint, with gravity compensation from the nominal model.

This is how most industrial arms are controlled: a fixed loop per joint, tuned
once at commissioning. The integral is clamped so it cannot wind up while a
motor is at its torque limit.
"""
import numpy as np

from src.dynamics.nominal import NominalModel
from src.plant.sensing import Measurement
from src.plant.trajectories import Reference


class PID:
    def __init__(self, nominal: NominalModel, proportional, integral, derivative, period: float, clamp: float = 10.0):
        self.nominal = nominal
        self.kp, self.ki, self.kd = (np.asarray(gain, dtype=float) for gain in (proportional, integral, derivative))
        self.period, self.clamp = period, clamp
        self.reset()

    def reset(self) -> None:
        self.accumulated = np.zeros_like(self.kp)

    def __call__(self, reference: Reference, measured: Measurement) -> np.ndarray:
        error = reference.angle - measured.angle
        self.accumulated = np.clip(self.accumulated + error * self.period, -self.clamp, self.clamp)
        gravity = self.nominal.torque(measured.angle, np.zeros_like(error), np.zeros_like(error))
        return (gravity + self.kp * error + self.ki * self.accumulated
                + self.kd * (reference.velocity - measured.velocity))

    def learn(self, measured: Measurement, applied: np.ndarray) -> None:
        pass  # fixed gains
