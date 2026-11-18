from pathlib import Path

import numpy as np
import pytest

from src.plant.friction import Friction
from src.plant.robot import build
from src.plant.workspace import Workspace

STAND_IN = Path(__file__).with_name("arm.xml")


@pytest.fixture
def model():
    return build(STAND_IN)


@pytest.fixture
def nominal_model():
    return build(STAND_IN)  # a separate copy, as the controller would hold


@pytest.fixture
def workspace() -> Workspace:
    return Workspace(home=np.array([0.0, 0.5]), span=np.array([1.2, 1.2]),
                     velocity=np.array([3.0, 3.0]), floor=-10.0, clearance=0.0)


@pytest.fixture
def friction() -> Friction:
    return Friction(coulomb=np.array([2.0, 1.0]), static=np.array([3.0, 1.5]), viscous=np.array([1.0, 0.5]))
