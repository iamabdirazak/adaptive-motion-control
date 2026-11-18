"""Classical adaptive control of the payload mass (Slotine and Li, 1987).

The payload enters the dynamics linearly: the torque it needs is its mass times
the torque a 1 kg payload would need. That regressor is computed with the
nominal model, by inverse dynamics with and without a 1 kg payload. The mass
estimate follows a gradient law driven by the sliding variable s, and is kept
non-negative. This baseline adapts to payload only, so it shows how much of
the problem a known structure already solves, and what it cannot reach.

The regressor uses the measured velocity in the Coriolis terms, a common
simplification of the original law that leaves its behaviour at these speeds unchanged.
"""
import numpy as np

from src.dynamics.nominal import NominalModel
from src.plant.sensing import Measurement
from src.plant.trajectories import Reference


class SlotineLi:
    def __init__(self, nominal: NominalModel, unit_payload: NominalModel, slope, damping, rate: float, period: float):
        self.nominal, self.unit = nominal, unit_payload
        self.slope, self.kd = np.asarray(slope, dtype=float), np.asarray(damping, dtype=float)
        self.rate, self.period = rate, period
        self.reset()

    def reset(self) -> None:
        self.mass = 0.0

    def __call__(self, reference: Reference, measured: Measurement) -> np.ndarray:
        error = reference.angle - measured.angle
        rate_error = reference.velocity - measured.velocity
        acceleration = reference.acceleration + self.slope * rate_error
        sliding = -(rate_error + self.slope * error)  # s = q̇ − q̇ᵣ
        base = self.nominal.torque(measured.angle, measured.velocity, acceleration)
        regressor = self.unit.torque(measured.angle, measured.velocity, acceleration) - base
        self.mass = max(0.0, self.mass - self.rate * float(regressor @ sliding) * self.period)
        return base + self.mass * regressor - self.kd * sliding

    def learn(self, measured: Measurement, applied: np.ndarray) -> None:
        pass  # the estimate is updated inside the control law
