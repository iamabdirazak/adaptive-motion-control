"""Choose the settings that need choosing, on the validation bank only.

The forgetting factor sets how far back the adaptive model remembers: too
short and it chases noise, too long and it lags the arm's changes. The
Slotine-Li rate sets how fast its payload estimate moves. Each candidate is
run on every validation scenario, and the lowest mean RMS error wins. The
test bank is never used here.

    python -m src.evaluation.tune --seeds 2
"""
import argparse
import json

import numpy as np

from src.evaluation.scenarios import BANKS
from src.evaluation.study import evaluate
from src.paths import REPORTS

CANDIDATES = {
    "forgetting": ("learned-adaptive", [0.995, 0.998, 0.999, 0.9995]),
    "slotine_rate": ("slotine-li", [1.0, 5.0, 20.0, 80.0]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None)
    options = parser.parse_args()
    trials, chosen = {}, {}
    for setting, (controller, values) in CANDIDATES.items():
        errors = {}
        for value in values:
            result = evaluate(controller, BANKS["validation"], list(range(options.seeds)), {setting: value}, options.workers)
            errors[value] = float(np.mean(result["metrics"]["rms_mm"]))
            print(f"{setting} = {value}: mean RMS error {errors[value]:.2f} mm")
        trials[setting] = errors
        chosen[setting] = min(errors, key=errors.get)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "tuning.json").write_text(json.dumps({"chosen": chosen, "trials": trials}, indent=2))
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
