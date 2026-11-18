"""Computed-torque control: the dynamics model turns a desired acceleration into torque.

The desired acceleration is the reference's acceleration plus a PD correction
on the tracking error. If the model were exact, the error would decay exactly
as the gains prescribe. Whatever the model gets wrong appears as tracking error,
which is why a better model, not higher gains, is what keeps the arm accurate.

An optional learned residual adds what the nominal model misses. It is evaluated
on the reference path, not on the measured state (desired compensation, Sadegh
and Horowitz, 1990). Evaluated on the measured state, the adapting model sits
inside the feedback loop: an error in its estimate moves the arm, the movement
changes the estimate, and the two can feed each other. On the reference, the
residual is pure feedforward, and feedback alone corrects what remains.
"""
import numpy as np

from src.dynamics.nominal import NominalModel
from src.plant.sensing import Measurement
from src.plant.trajectories import Reference


class ComputedTorque:
    def __init__(self, nominal: NominalModel, stiffness, damping, residual=None, on_reference: bool = True):
        self.nominal, self.kp, self.kd, self.residual = nominal, stiffness, damping, residual
        self.on_reference = on_reference  # False only for the ablation that puts the residual in the feedback loop

    def reset(self) -> None:
        if self.residual is not None:
            self.residual.reset()

    def __call__(self, reference: Reference, measured: Measurement) -> np.ndarray:
        desired = (reference.acceleration
                   + self.kp * (reference.angle - measured.angle)
                   + self.kd * (reference.velocity - measured.velocity))
        torque = self.nominal.torque(measured.angle, measured.velocity, desired)
        if self.residual is not None:
            state = ((reference.angle, reference.velocity, reference.acceleration) if self.on_reference
                     else (measured.angle, measured.velocity, desired))
            torque += self.residual.predict(*state)
        return torque

    def learn(self, measured: Measurement, applied: np.ndarray) -> None:
        if self.residual is not None:
            self.residual.learn(measured, applied)
