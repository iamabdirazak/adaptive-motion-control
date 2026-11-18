"""The ceiling: computed torque that is told the arm's true conditions.

It reads the payload, friction and wear the simulation is running under, which
no real controller can, and inverts them exactly. Its error is what remains
from sampling, sensing and feedback alone, the best any model can do here.
"""
import numpy as np

from src.control.computed_torque import ComputedTorque
from src.dynamics.nominal import NominalModel
from src.plant.arm import Arm


class Oracle(ComputedTorque):
    def __init__(self, nominal: NominalModel, stiffness, damping, arm: Arm):
        super().__init__(nominal, stiffness, damping)
        self.arm = arm
        self.payload = nominal.model.body("payload").id

    def __call__(self, reference, measured) -> np.ndarray:
        truth = self.arm.conditions
        self.nominal.model.body_mass[self.payload] = self.arm.model.body_mass[self.arm.payload]
        self.nominal.model.body_inertia[self.payload] = self.arm.model.body_inertia[self.arm.payload]
        torque = super().__call__(reference, measured)
        if self.arm.friction is not None:
            torque -= self.arm.friction.torque(measured.velocity, truth.friction)
        return torque / (1.0 - truth.wear)
