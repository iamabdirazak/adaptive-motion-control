"""Where every file lives."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

ROBOT = ASSETS / "lbr-iisy" / "lbr_iisy8_r930.xml"  # KUKA LBR iisy 8 R930, as KUKA describes it

CONTROL_PERIOD = 0.001  # 1 kHz, the servo rate of industrial arm controllers
SIMULATION_STEP = 0.0005  # 2 kHz, the rate of the physics, twice the control rate
