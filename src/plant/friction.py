"""Joint friction that the nominal model does not contain.

Coulomb, viscous and Stribeck terms, typical of a harmonic drive. The sign of
the velocity is smoothed with tanh, so the force is continuous through zero.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Friction:
    coulomb: np.ndarray   # N·m, sliding friction
    static: np.ndarray    # N·m, breakaway friction, above coulomb
    viscous: np.ndarray   # N·m·s/rad
    stribeck: float = 0.05  # rad/s, how fast breakaway friction fades with speed
    smoothing: float = 0.005  # rad/s, width of the smoothed sign

    def torque(self, velocity: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """The torque friction applies to each joint, opposing its motion."""
        breakaway = self.coulomb + (self.static - self.coulomb) * np.exp(-(velocity / self.stribeck) ** 2)
        return -scale * (np.tanh(velocity / self.smoothing) * breakaway + self.viscous * velocity)
