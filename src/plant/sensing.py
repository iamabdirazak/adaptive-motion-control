"""What the controller can measure: encoder angles, and the rates derived from them.

Angles are quantised to the encoder's resolution. Velocity and acceleration are
finite differences, smoothed by first-order filters, as on a real drive. Velocity
feeds the feedback loop, so its filter is light; acceleration only feeds the
model's adaptation, where noise matters more than delay, so its filter is heavier.
"""
from dataclasses import dataclass

import numpy as np


def weight(cutoff: float, period: float) -> float:
    """The weight of a first-order low-pass filter with this cutoff (Hz) at this rate."""
    return 1 - np.exp(-2 * np.pi * cutoff * period)


@dataclass(frozen=True)
class Measurement:
    angle: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class Sensing:
    def __init__(self, joints: int, period: float, bits: int = 19,
                 velocity_cutoff: float = 150.0, acceleration_cutoff: float = 40.0):
        self.period = period
        self.step = 2 * np.pi / 2**bits  # radians per encoder count
        self.velocity_weight, self.acceleration_weight = weight(velocity_cutoff, period), weight(acceleration_cutoff, period)
        self.reset(np.zeros(joints))

    def reset(self, angle: np.ndarray, velocity: np.ndarray | None = None) -> None:
        self.angle = self._quantise(angle)
        self.velocity = np.zeros_like(angle) if velocity is None else velocity.copy()
        self.acceleration = np.zeros_like(angle)

    def read(self, true_angle: np.ndarray) -> Measurement:
        angle = self._quantise(true_angle)
        velocity = self._smooth(self.velocity, (angle - self.angle) / self.period, self.velocity_weight)
        self.acceleration = self._smooth(self.acceleration, (velocity - self.velocity) / self.period,
                                         self.acceleration_weight)
        self.angle, self.velocity = angle, velocity
        return Measurement(angle, velocity, self.acceleration)

    def _quantise(self, angle: np.ndarray) -> np.ndarray:
        return np.round(angle / self.step) * self.step

    @staticmethod
    def _smooth(previous: np.ndarray, raw: np.ndarray, weight: float) -> np.ndarray:
        return previous + weight * (raw - previous)


class Lag:
    """Delays any signal exactly as the acceleration estimate is delayed: a finite
    difference, then the velocity filter, then the acceleration filter.

    The measured acceleration describes the motion a few milliseconds late. The torque
    that caused it is known at once. Passing the torque through the same chain lines
    the two up in time, so their difference measures the model's error, not the delay.
    """

    def __init__(self, joints: int, period: float, velocity_cutoff: float = 150.0, acceleration_cutoff: float = 40.0):
        self.weights = (weight(velocity_cutoff, period), weight(acceleration_cutoff, period))
        self.joints = joints
        self.reset()

    def reset(self) -> None:
        self.previous = None
        self.stages = [np.zeros(self.joints), np.zeros(self.joints)]

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        # A finite difference followed by its inverse sum is a half-step average.
        averaged = signal if self.previous is None else (signal + self.previous) / 2
        self.previous = signal
        for index, gain in enumerate(self.weights):
            self.stages[index] = self.stages[index] + gain * (averaged - self.stages[index])
            averaged = self.stages[index]
        return averaged
