"""How the arm changes while it works: payload, friction and actuator wear.

Each change is a slow drift or a step that the controller is never told about.
A schedule holds the start and end of each drift over one episode, and the
payload carried between a pick and the next place.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Conditions:
    """The state of the arm at one instant."""

    payload: float = 0.0   # kg held at the flange
    friction: float = 1.0  # multiple of the arm's base joint friction
    wear: float = 0.0      # share of motor torque lost to wear


NOMINAL = Conditions()

# Ranges seen while collecting training data, and the wider ranges kept for testing.
RANGES = {
    "training": {"payload": (0.0, 4.0), "friction": (0.7, 1.3), "wear": (0.0, 0.15)},
    "beyond": {"payload": (4.0, 8.0), "friction": (0.4, 1.6), "wear": (0.15, 0.30)},
}


@dataclass(frozen=True)
class Schedule:
    """The conditions over one episode of length `duration` seconds."""

    payload: float
    friction: tuple[float, float]
    wear: tuple[float, float]
    duration: float

    def at(self, time: float, holding: bool) -> Conditions:
        share = min(max(time / self.duration, 0.0), 1.0)
        return Conditions(
            payload=self.payload if holding else 0.0,
            friction=float(np.interp(share, [0, 1], self.friction)),
            wear=float(np.interp(share, [0, 1], self.wear)),
        )


def sample(rng: np.random.Generator, duration: float, ranges: str = "training") -> Schedule:
    """A schedule drawn uniformly from one set of ranges."""
    bounds = RANGES[ranges]
    draw = lambda name: tuple(sorted(rng.uniform(*bounds[name], size=2)))
    friction = draw("friction")
    return Schedule(
        payload=float(rng.uniform(*bounds["payload"])),
        friction=friction if rng.random() < 0.5 else friction[::-1],  # joints warm up or cool down
        wear=draw("wear"),                                              # wear only grows
        duration=duration,
    )
