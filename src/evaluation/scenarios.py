"""Fixed banks of scenarios, so every controller faces exactly the same episodes.

A scenario is a seed and a set of ranges. The seed fixes the pick-and-place
path and the schedule of changes, so a comparison between controllers on the
same bank is paired. Validation scenarios tune settings and test scenarios
report results, and their seeds never overlap.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from src.plant.changes import Schedule, sample
from src.plant.trajectories import PickAndPlace
from src.plant.workspace import Workspace


@dataclass(frozen=True)
class Scenario:
    seed: int
    ranges: str  # "training" or "beyond"

    def build(self, model: mujoco.MjModel, workspace: Workspace) -> tuple[PickAndPlace, Schedule]:
        rng = np.random.default_rng(self.seed)
        path = PickAndPlace.sample(rng, model, workspace)
        return path, sample(rng, path.duration, self.ranges)


BANKS = {
    "validation": [Scenario(1000 + i, "training") for i in range(20)],
    "test": [Scenario(2000 + i, "training") for i in range(20)] + [Scenario(3000 + i, "beyond") for i in range(20)],
}
