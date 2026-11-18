"""Run controllers through a bank of scenarios, in parallel, and save every score.

Each (controller, training seed, scenario) is one task. Controllers without
training run once per scenario; learned controllers run once per training seed.
Results go to reports/<bank>/<controller>.json as arrays shaped (seeds, scenarios),
ready for the hierarchical bootstrap.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.evaluation.controllers import DETERMINISTIC, make
from src.evaluation.metrics import score
from src.evaluation.rollout import run
from src.evaluation.scenarios import BANKS, Scenario
from src.paths import REPORTS, ROBOT
from src.plant.arm import Arm
from src.plant.robot import build, friction, workspace


def episode(task: tuple[str, int | None, Scenario, dict]) -> dict[str, float]:
    name, seed, scenario, overrides = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    model = build(ROBOT)
    arm = Arm(model, workspace(), friction(model))
    return score(run(make(name, arm, seed, **overrides), arm, scenario))


def evaluate(name: str, bank: list[Scenario], seeds: list[int], overrides: dict | None = None,
             workers: int | None = None) -> dict:
    seeds = [None] if name in DETERMINISTIC else seeds
    tasks = [(name, seed, scenario, overrides or {}) for seed in seeds for scenario in bank]
    with ProcessPoolExecutor(workers) as pool:
        scores = list(pool.map(episode, tasks))
    metrics = {key: np.array([s[key] for s in scores]).reshape(len(seeds), len(bank)).tolist() for key in scores[0]}
    return {"controller": name, "seeds": seeds, "scenarios": [[s.seed, s.ranges] for s in bank], "metrics": metrics}


def save(result: dict, bank_name: str, label: str | None = None) -> Path:
    folder = REPORTS / bank_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label or result['controller']}.json"
    path.write_text(json.dumps(result))
    return path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("controllers", nargs="+")
    parser.add_argument("--bank", default="test", choices=list(BANKS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    options = parser.parse_args()
    for name in options.controllers:
        result = evaluate(name, BANKS[options.bank], list(range(options.seeds)), workers=options.workers)
        rms = np.array(result["metrics"]["rms_mm"])
        print(f"{name:18s} RMS error {rms.mean():6.2f} mm  -> {save(result, options.bank)}")


if __name__ == "__main__":
    main()
