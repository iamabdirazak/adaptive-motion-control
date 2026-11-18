"""Pick-and-place motion in joint space.

The arm moves between waypoints with minimum-jerk profiles, the smooth moves an
industrial planner produces, and pauses at each one. Each move takes long enough
that no joint passes a set share of its rated speed. The arm picks at one
waypoint and places at the next, so it holds the payload on every other move.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from src.plant.workspace import Workspace

PEAK = 1.875  # peak speed of a minimum-jerk move, as a multiple of its average speed


def minimum_jerk(share: float) -> tuple[float, float, float]:
    """Position, velocity and acceleration of a unit move at a share of its duration."""
    s = min(max(share, 0.0), 1.0)
    return (10 * s**3 - 15 * s**4 + 6 * s**5,
            30 * s**2 - 60 * s**3 + 30 * s**4,
            60 * s - 180 * s**2 + 120 * s**3)


@dataclass(frozen=True)
class Reference:
    angle: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    holding: bool


class PickAndPlace:
    def __init__(self, waypoints: np.ndarray, durations: np.ndarray, dwell: float):
        self.waypoints, self.durations, self.dwell = waypoints, durations, dwell
        self.starts = np.concatenate([[0.0], np.cumsum(durations + dwell)])
        self.duration = float(self.starts[-1])

    def at(self, time: float) -> Reference:
        index = int(np.clip(np.searchsorted(self.starts, time, side="right") - 1, 0, len(self.durations) - 1))
        start, span, move = self.waypoints[index], self.waypoints[index + 1] - self.waypoints[index], self.durations[index]
        position, velocity, acceleration = minimum_jerk((time - self.starts[index]) / move)
        # Even moves carry nothing to a pick; odd moves carry the part to its place.
        return Reference(start + position * span, velocity * span / move,
                         acceleration * span / move**2, holding=index % 2 == 1)

    @classmethod
    def sample(cls, rng: np.random.Generator, model: mujoco.MjModel, workspace: Workspace,
               moves: int = 16, speed: float = 0.7, shortest: float = 0.6, dwell: float = 0.35) -> "PickAndPlace":
        """Random waypoints in the workspace, starting from home. `speed` is the share of
        rated joint speed the fastest joint reaches at the middle of each move."""
        waypoints = np.vstack([workspace.home, workspace.sample(rng, model, moves)])
        needed = PEAK * np.abs(np.diff(waypoints, axis=0)) / (speed * workspace.velocity)
        return cls(waypoints, np.maximum(needed.max(axis=1), shortest), dwell)
