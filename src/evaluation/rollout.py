"""Run one controller through one scenario and record what happened.

Each control step reads the sensors, lets the controller learn from the last
step, computes a torque, and runs the arm for one control period under the
scenario's conditions. The tool position is recorded alongside where the
reference path would have put it, so errors are measured in millimetres at the tool.
"""
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from src.evaluation.scenarios import Scenario
from src.plant.arm import Arm
from src.plant.sensing import Sensing


@dataclass
class Log:
    time: np.ndarray       # s
    error: np.ndarray      # mm, distance between the tool and its reference position
    torque: np.ndarray     # N·m, command after limits, per joint
    saturated: np.ndarray  # bool, any joint at its limit
    holding: np.ndarray    # bool, a part is held
    compute: np.ndarray    # µs, time the controller took per step
    period: float


def run(controller, arm: Arm, scenario: Scenario) -> Log:
    path, schedule = scenario.build(arm.model, arm.workspace)
    sensing = Sensing(arm.joints, arm.period)
    reference_data = mujoco.MjData(arm.model)  # kinematics only, for the reference tool position
    start = path.at(0.0)
    arm.reset(start.angle)
    sensing.reset(start.angle)
    controller.reset()

    steps = int(path.duration / arm.period)
    records = {name: [] for name in ("error", "torque", "saturated", "holding", "compute")}
    applied = np.zeros(arm.joints)
    for step in range(steps):
        now = step * arm.period
        reference = path.at(now)
        arm.apply(schedule.at(now, reference.holding))
        measured = sensing.read(arm.angle)

        tick = time.perf_counter()
        controller.learn(measured, applied)
        torque = controller(reference, measured)
        records["compute"].append((time.perf_counter() - tick) * 1e6)

        applied = arm.step(torque)
        reference_data.qpos[:] = reference.angle
        mujoco.mj_kinematics(arm.model, reference_data)
        records["error"].append(1e3 * np.linalg.norm(arm.tool_position - reference_data.site_xpos[arm.tool]))
        records["torque"].append(applied)
        records["saturated"].append(bool(np.any(np.abs(applied) >= arm.limit * 0.999)))
        records["holding"].append(reference.holding)

    return Log(np.arange(steps) * arm.period, *(np.asarray(records[name]) for name in
               ("error", "torque", "saturated", "holding", "compute")), arm.period)
