"""Train the model-free baseline: Soft Actor-Critic on the residual environment.

SAC is the standard model-free method for continuous control. It learns a
policy (the actor) and an estimate of each action's long-run value (the
critic) from replayed experience, and keeps exploring by rewarding randomness.
It trains on the CPU, on training scenarios only, with domain randomisation
coming from the scenarios' changing payload, friction and wear.

    python -m src.training.rl --seed 0 --steps 150000
"""
import argparse
import json
import time

import torch
from stable_baselines3 import SAC

from src.paths import RUNS
from src.training.environment import LENGTH, ArmEnvironment
from src.control.policy import REPEAT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=150_000)
    options = parser.parse_args()
    torch.set_num_threads(2)
    started = time.time()
    agent = SAC("MlpPolicy", ArmEnvironment(options.seed), seed=options.seed, device="cpu",
                learning_starts=5_000, buffer_size=300_000, batch_size=256,
                policy_kwargs={"net_arch": [256, 256]}, verbose=0)
    agent.learn(total_timesteps=options.steps, log_interval=10, progress_bar=False)
    folder = RUNS / f"seed-{options.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    agent.save(folder / "sac.zip")
    operating = options.steps * REPEAT / 1000  # seconds of arm operation the agent experienced
    (folder / "sac.json").write_text(json.dumps({"steps": options.steps, "operating_seconds": operating,
                                                  "episodes": options.steps // LENGTH,
                                                  "wall_seconds": time.time() - started}, indent=2))
    print(f"seed {options.seed}: {options.steps:,} steps, {operating / 60:.0f} min of operation, "
          f"{(time.time() - started) / 60:.0f} min to train")


if __name__ == "__main__":
    main()
