import numpy as np

from src.paths import ROBOT
from src.plant.arm import Arm
from src.plant.changes import Conditions, Schedule
from src.plant.robot import build, workspace as iisy_workspace
from src.plant.trajectories import PickAndPlace, minimum_jerk


def test_minimum_jerk_starts_and_ends_at_rest():
    assert minimum_jerk(0.0) == (0.0, 0.0, 0.0)
    assert np.allclose(minimum_jerk(1.0), (1.0, 0.0, 0.0))


def test_path_reaches_each_waypoint_and_alternates_holding():
    path = PickAndPlace(np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]]), np.array([1.0, 1.0]), dwell=0.5)
    assert np.allclose(path.at(1.2).angle, [1.0, -1.0])
    assert not path.at(1.2).holding and path.at(1.6).holding


def test_paths_respect_rated_joint_speed(model, workspace):
    path = PickAndPlace.sample(np.random.default_rng(0), model, workspace)
    peak = np.max([np.abs(path.at(t).velocity) / workspace.velocity for t in np.arange(0, path.duration, 0.01)])
    assert peak <= 0.7 + 1e-6


def test_conditions_reach_the_model(model, workspace):
    arm = Arm(model, workspace)
    arm.apply(Conditions(payload=2.0, wear=0.2))
    assert np.isclose(arm.model.body_mass[arm.payload], 2.0)
    assert np.allclose(arm.model.actuator_gear[:, 0], 0.8)


def test_schedule_holds_payload_only_while_holding():
    schedule = Schedule(payload=1.0, friction=(1.0, 1.4), wear=(0.0, 0.1), duration=10.0)
    assert schedule.at(5.0, holding=False).payload == 0.0
    assert np.isclose(schedule.at(5.0, holding=True).friction, 1.2)


def test_friction_opposes_motion(friction):
    velocity = np.array([0.5, -0.5])
    assert np.all(friction.torque(velocity) * velocity < 0)


def test_robot_gets_rated_torque_motors_and_a_payload():
    model = build(ROBOT)
    assert model.nu == 6
    assert np.allclose(model.actuator_ctrlrange[:, 1], model.jnt_actfrcrange[:, 1])
    assert model.body("payload").parentid == model.body("link_6").id
    assert len(iisy_workspace().sample(np.random.default_rng(0), model, 5)) == 5
