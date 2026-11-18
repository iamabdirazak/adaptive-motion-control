"""Collect operating logs: the arm doing pick-and-place under changing conditions.

The logs come from the controller a machine already runs, computed torque on
the datasheet model, as a company would record them from its own cells. Each
record is exactly what the adaptive model sees online: the state before a
control step, the acceleration measured after it, the torque applied, and the
residual torque the nominal model failed to explain. Records are kept every
fifth step (200 Hz), which keeps the logs small without losing the dynamics.

    python -m src.training.collect --seed 0 --episodes 40
"""
import argparse

import numpy as np

from src.control.computed_torque import ComputedTorque
from src.control.gains import second_order
from src.dynamics.nominal import NominalModel
from src.dynamics.residual import encode
from src.evaluation.rollout import run
from src.evaluation.scenarios import Scenario
from src.paths import DATA, ROBOT
from src.plant.arm import Arm
from src.plant.robot import build, friction, workspace
from src.plant.sensing import Lag

FREQUENCY = 30.0  # rad/s, closed-loop bandwidth of every model-based controller in this project


class Recorder:
    """Wraps a controller and records what an adaptive model would learn from,
    computed exactly as LearnedResidual computes it online."""

    def __init__(self, controller, nominal: NominalModel, lag: Lag, every: int = 5):
        self.controller, self.nominal, self.lag, self.every = controller, nominal, lag, every
        self.inputs, self.residuals = [], []

    def reset(self) -> None:
        self.controller.reset()
        self.lag.reset()
        self.previous, self.count = None, 0

    def __call__(self, reference, measured):
        return self.controller(reference, measured)

    def learn(self, measured, applied) -> None:
        self.controller.learn(measured, applied)
        delayed = self.lag(applied)
        if self.previous is not None and self.count % self.every == 0:
            state = (self.previous.angle, self.previous.velocity, measured.acceleration)
            self.inputs.append(encode(*state))
            self.residuals.append((delayed - self.nominal.torque(*state)).astype(np.float32))
        self.previous, self.count = measured, self.count + 1


def scenarios(seed: int, episodes: int) -> list[Scenario]:
    """Training episodes for one training seed. Their seeds never meet a validation or test seed."""
    return [Scenario(10_000 * (seed + 1) + index, "training") for index in range(episodes)]


def collect(seed: int, episodes: int) -> dict[str, np.ndarray]:
    model = build(ROBOT)
    arm = Arm(model, workspace(), friction(model))
    nominal = NominalModel(build(ROBOT))
    inputs, residuals, episode = [], [], []
    for index, scenario in enumerate(scenarios(seed, episodes)):
        recorder = Recorder(ComputedTorque(nominal, *second_order(model.nu, FREQUENCY)), nominal, Lag(model.nu, arm.period))
        run(recorder, arm, scenario)
        inputs += recorder.inputs
        residuals += recorder.residuals
        episode += [index] * len(recorder.inputs)
    return {"inputs": np.array(inputs), "residuals": np.array(residuals), "episode": np.array(episode)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=40)
    options = parser.parse_args()
    logs = collect(options.seed, options.episodes)
    (DATA / "logs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / "logs" / f"seed-{options.seed}.npz", **logs)
    print(f"seed {options.seed}: {len(logs['inputs']):,} records from {options.episodes} episodes")


if __name__ == "__main__":
    main()
