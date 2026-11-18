"""Every controller in the study, built by name with the same feedback bandwidth.

All model-based controllers share one closed-loop bandwidth, so they differ
only in the model they use. The PID's gains come from the same bandwidth and
each joint's nominal inertia at the home pose, the way a drive is commissioned.
Settings that need choosing are chosen on the validation bank by tune.py and
read from reports/tuning.json.
"""
import json

import numpy as np

from src.control.computed_torque import ComputedTorque
from src.control.gains import second_order
from src.control.oracle import Oracle
from src.control.pid import PID
from src.control.slotine_li import SlotineLi
from src.dynamics.adaptation import RecursiveLeastSquares
from src.dynamics.learned import LearnedResidual
from src.dynamics.nominal import NominalModel
from src.paths import REPORTS, ROBOT, RUNS
from src.plant.arm import Arm
from src.plant.robot import build
from src.plant.sensing import Lag
from src.system import OnnxFeatures
from src.training.collect import FREQUENCY

DETERMINISTIC = ["pid", "computed-torque", "slotine-li", "oracle"]
LEARNED = ["learned-fixed", "learned-adaptive", "learned-feedback", "residual-sac"]
DEFAULTS = {"forgetting": 0.999, "slotine_rate": 20.0}


def settings() -> dict:
    tuned = REPORTS / "tuning.json"
    return {**DEFAULTS, **(json.loads(tuned.read_text())["chosen"] if tuned.exists() else {})}


def joint_inertia(nominal: NominalModel, pose: np.ndarray) -> np.ndarray:
    """Each joint's own inertia at a pose: the torque a unit acceleration of that joint
    alone needs, beyond what holding the pose needs."""
    still = np.zeros_like(pose)
    hold = nominal.torque(pose, still, still)
    return np.array([nominal.torque(pose, still, unit)[index] - hold[index]
                     for index, unit in enumerate(np.eye(len(pose)))])


def make(name: str, arm: Arm, seed: int | None = None, **overrides):
    chosen = {**settings(), **overrides}
    nominal = NominalModel(build(ROBOT))
    joints = nominal.model.nu
    stiffness, damping = second_order(joints, FREQUENCY)

    if name == "computed-torque":
        return ComputedTorque(nominal, stiffness, damping)
    if name == "oracle":
        return Oracle(nominal, stiffness, damping, arm)
    if name == "pid":
        inertia = joint_inertia(nominal, arm.workspace.home)
        return PID(nominal, inertia * stiffness, inertia * stiffness * FREQUENCY / 10, inertia * damping, arm.period)
    if name == "slotine-li":
        unit = NominalModel(build(ROBOT))
        payload = unit.model.body("payload").id
        unit.model.body_mass[payload], unit.model.body_inertia[payload] = 1.0, 0.4 * 0.03**2
        inertia = joint_inertia(nominal, arm.workspace.home)
        return SlotineLi(nominal, unit, np.full(joints, FREQUENCY), inertia * FREQUENCY, chosen["slotine_rate"], arm.period)
    if name in ("learned-fixed", "learned-adaptive", "learned-feedback"):
        folder = RUNS / f"seed-{seed}"
        prior = json.loads((folder / "export.json").read_text())["prior"]
        estimator = RecursiveLeastSquares(np.load(folder / "head.npy"), chosen["forgetting"], prior)
        residual = LearnedResidual(nominal, OnnxFeatures(folder / "dynamics.onnx"), estimator,
                                   Lag(joints, arm.period), adapt=name != "learned-fixed")
        return ComputedTorque(nominal, stiffness, damping, residual, on_reference=name != "learned-feedback")
    if name == "residual-sac":
        from src.control.policy import ResidualPolicy  # needs stable-baselines3, only for this baseline
        return ResidualPolicy(ComputedTorque(nominal, stiffness, damping), RUNS / f"seed-{seed}" / "sac.zip")
    raise ValueError(f"unknown controller {name}")
