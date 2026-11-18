"""Record how the tool error evolves through one pick, for the time-trace figure.

One beyond-range test scenario is run with each controller, and the tool error
is kept for a few seconds around the first time the arm picks up its part.

    python -m src.evaluation.trace
"""
import json

import numpy as np

from src.evaluation.controllers import make
from src.evaluation.rollout import run
from src.evaluation.scenarios import BANKS
from src.paths import REPORTS, ROBOT
from src.plant.arm import Arm
from src.plant.robot import build, friction, workspace

CONTROLLERS = ["computed-torque", "slotine-li", "residual-sac", "learned-adaptive", "oracle"]
BEFORE, AFTER = 0.5, 2.5  # seconds kept either side of the pick


def main(scenario_index: int = 20) -> None:
    scenario = BANKS["test"][scenario_index]
    traces = {}
    for name in CONTROLLERS:
        model = build(ROBOT)
        arm = Arm(model, workspace(), friction(model))
        log = run(make(name, arm, seed=0), arm, scenario)
        pick = int(np.argmax(log.holding))
        window = slice(pick - int(BEFORE / log.period), pick + int(AFTER / log.period))
        traces[name] = log.error[window][::5].round(3).tolist()  # 200 Hz is plenty for a figure
    schedule = scenario.build(model, workspace())[1]
    (REPORTS / "trace.json").write_text(json.dumps({
        "scenario": scenario.seed, "payload_kg": schedule.payload, "period": 0.005, "pick_at": BEFORE,
        "errors": traces}))
    print({name: round(float(np.sqrt(np.mean(np.square(e)))), 2) for name, e in traces.items()})


if __name__ == "__main__":
    main()
