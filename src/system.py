"""The released controller: computed torque on the nominal model plus a learned
residual whose head adapts at every control step.

The feature network runs from its ONNX file, so using the controller needs only
NumPy, MuJoCo and ONNX Runtime. The nominal model is the robot's model file.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime

from src.control.computed_torque import ComputedTorque
from src.control.gains import second_order
from src.dynamics.adaptation import RecursiveLeastSquares
from src.dynamics.learned import LearnedResidual
from src.dynamics.nominal import NominalModel
from src.paths import CONTROL_PERIOD
from src.plant.robot import build
from src.plant.sensing import Lag


class OnnxFeatures:
    def __init__(self, path: Path):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1  # one small call per control step; threads cost more than they save
        self.session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input: x[None]})[0][0].astype(np.float64)


def adaptive_controller(robot: Path, models: Path, adapt: bool = True) -> ComputedTorque:
    card = json.loads((models / "model.json").read_text())
    nominal = NominalModel(build(robot))
    adaptation = card["adaptation"]
    estimator = RecursiveLeastSquares(np.load(models / "head.npy"), adaptation["forgetting"], adaptation["prior"])
    residual = LearnedResidual(nominal, OnnxFeatures(models / "dynamics.onnx"), estimator,
                               Lag(nominal.model.nu, CONTROL_PERIOD), adapt)
    stiffness, damping = second_order(nominal.model.nu, card["control"]["bandwidth_rad_s"])
    return ComputedTorque(nominal, stiffness, damping, residual)
