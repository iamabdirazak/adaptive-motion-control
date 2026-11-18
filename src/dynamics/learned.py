"""The learned dynamics: the nominal model plus a residual that adapts online.

At each control step the controller asks for the residual torque at the state
and acceleration it wants. Once the step has run, the measured acceleration
shows what the torque actually did, and the difference to the nominal model
is a fresh measurement of the residual, which updates the head. The applied
torque is delayed by the same lag as the acceleration estimate, so the two
describe the same moment.
"""
import numpy as np

from src.dynamics.adaptation import RecursiveLeastSquares
from src.dynamics.nominal import NominalModel
from src.dynamics.residual import encode
from src.plant.sensing import Lag, Measurement


class LearnedResidual:
    def __init__(self, nominal: NominalModel, features, estimator: RecursiveLeastSquares, lag: Lag, adapt: bool = True):
        self.nominal, self.features, self.estimator, self.lag, self.adapt = nominal, features, estimator, lag, adapt
        self.previous = None

    def reset(self) -> None:
        self.estimator.reset()
        self.lag.reset()
        self.previous = None

    def predict(self, angle: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self.features(encode(angle, velocity, acceleration)))

    def learn(self, measured: Measurement, applied: np.ndarray) -> None:
        """`applied` is the torque that acted since the previous measurement."""
        delayed = self.lag(applied)
        if self.adapt and self.previous is not None:
            state = (self.previous.angle, self.previous.velocity, measured.acceleration)
            residual = delayed - self.nominal.torque(*state)
            self.estimator.update(self.features(encode(*state)), residual)
        self.previous = measured
