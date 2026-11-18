import numpy as np

from src.control.computed_torque import ComputedTorque
from src.control.gains import second_order
from src.control.pid import PID
from src.control.slotine_li import SlotineLi
from src.dynamics.nominal import NominalModel
from src.evaluation.metrics import score
from src.evaluation.rollout import run
from src.evaluation.scenarios import Scenario
from src.plant.arm import Arm
from src.plant.changes import Schedule
from src.plant.robot import build
from tests.conftest import STAND_IN


class Unchanged(Scenario):
    """A scenario whose arm never changes, so a controller with the true model should track exactly."""

    def build(self, model, workspace):
        path, schedule = super().build(model, workspace)
        return path, Schedule(0.0, (1.0, 1.0), (0.0, 0.0), schedule.duration)


def computed_torque(model):
    return ComputedTorque(NominalModel(model), *second_order(model.nu, 30.0))


def test_true_model_tracks_below_half_a_millimetre(model, nominal_model, workspace):
    result = score(run(computed_torque(nominal_model), Arm(model, workspace), Unchanged(1, "training")))
    assert result["rms_mm"] < 0.5


def test_payload_and_friction_degrade_the_fixed_model(model, nominal_model, workspace, friction):
    result = score(run(computed_torque(nominal_model), Arm(model, workspace, friction), Scenario(1, "beyond")))
    assert result["rms_mm"] > 1.0


def test_pid_and_slotine_li_run(model, nominal_model, workspace):
    nominal = NominalModel(nominal_model)
    unit = NominalModel(build(STAND_IN))
    unit.model.body_mass[unit.model.body("payload").id] = 1.0
    unit.model.body_inertia[unit.model.body("payload").id] = 0.4 * 0.03**2
    pid = PID(nominal, [900, 500], [200, 100], [60, 30], period=0.001)
    adaptive = SlotineLi(nominal, unit, slope=[10, 10], damping=[60, 30], rate=0.5, period=0.001)
    for controller in (pid, adaptive):
        assert np.isfinite(score(run(controller, Arm(model, workspace), Scenario(2, "training")))["rms_mm"])
