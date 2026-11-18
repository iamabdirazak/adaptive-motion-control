"""Train the feature network so that a few moments of data are enough to adapt.

Training mimics deployment. The logs are cut into short windows. In each window
the head is fitted in closed form to the first 0.3 s (the context), starting
from a learned prior head, exactly as recursive least squares would from its
prior. The loss is the error of that fitted head on the next 0.2 s (the query).
So the network learns features in which the residual under any condition is
linear and quick to estimate, and the prior head is the best guess before any
data arrives. The prior's strength is learned too, and becomes the starting
covariance of recursive least squares.

Targets are divided by each joint's spread, so the base joints, whose torques
are a hundred times the wrist's, do not dominate the loss.

    python -m src.training.fit --seed 0
"""
import argparse
import json

import numpy as np
import torch
from torch import nn

from src.dynamics.residual import Features
from src.paths import DATA, RUNS

CONTEXT, QUERY = 60, 40  # samples at 200 Hz: 0.3 s to adapt on, 0.2 s to be judged on


class ResidualModel(nn.Module):
    """Normalises the encoded state and maps it to features. This is what ships as ONNX."""

    def __init__(self, mean: np.ndarray, spread: np.ndarray, joints: int, width: int):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("spread", torch.tensor(spread, dtype=torch.float32))
        self.features = Features(joints, width)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.features((state - self.mean) / self.spread)


def windows(logs: dict, episodes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Consecutive windows of CONTEXT + QUERY samples from the chosen episodes."""
    size = CONTEXT + QUERY
    inputs, targets = [], []
    for episode in episodes:
        rows = np.flatnonzero(logs["episode"] == episode)
        for start in range(0, len(rows) - size + 1, size):
            inputs.append(logs["inputs"][rows[start:start + size]])
            targets.append(logs["residuals"][rows[start:start + size]])
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def adapted_error(model: ResidualModel, prior: torch.Tensor, log_strength: torch.Tensor,
                  inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared query error after fitting each window's head to its context."""
    features = model(inputs)
    context, query = features[:, :CONTEXT], features[:, CONTEXT:]
    strength = log_strength.exp()
    identity = torch.eye(features.shape[-1])
    # Ridge regression towards the prior head: (ΦᵀΦ + λI)⁻¹ (ΦᵀY + λ·prior), per window.
    gram = context.transpose(1, 2) @ context + strength * identity
    moment = context.transpose(1, 2) @ targets[:, :CONTEXT] + strength * prior
    head = torch.linalg.solve(gram, moment)
    return ((query @ head - targets[:, CONTEXT:]) ** 2).mean()


def fit(logs: dict, seed: int, width: int = 32, epochs: int = 40, batch: int = 128) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    episodes = rng.permutation(np.unique(logs["episode"]))
    held_out = max(1, len(episodes) // 8)
    train, check = windows(logs, episodes[held_out:]), windows(logs, episodes[:held_out])

    flat = train[0].reshape(-1, train[0].shape[-1]).numpy()
    scale = train[1].reshape(-1, train[1].shape[-1]).std(0)
    joints = train[1].shape[-1]
    model = ResidualModel(flat.mean(0), flat.std(0) + 1e-6, joints, width)
    prior = nn.Parameter(torch.zeros(width, joints))
    log_strength = nn.Parameter(torch.tensor(0.0))
    parameters = [*model.parameters(), prior, log_strength]
    optimiser = torch.optim.Adam(parameters, lr=1e-3)
    normalise = lambda data: (data[0], data[1] / scale)

    train, check = normalise(train), normalise(check)
    best, history = float("inf"), []
    for epoch in range(epochs):
        model.train()
        for index in torch.randperm(len(train[0])).split(batch):
            optimiser.zero_grad()
            loss = adapted_error(model, prior, log_strength, train[0][index], train[1][index])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            error = adapted_error(model, prior, log_strength, *check).item()
            fixed = ((model(check[0]) @ prior - check[1]) ** 2).mean().item()
        history.append({"epoch": epoch + 1, "adapted": error, "prior_only": fixed})
        if error < best:
            best = error
            state = {"model": {k: v.clone() for k, v in model.state_dict().items()},
                     "prior": prior.detach().clone(), "log_strength": log_strength.detach().clone()}
        print(f"epoch {epoch + 1:2d}  held-out error after adapting {error:.4f}  with the prior head only {fixed:.4f}")

    return {**state, "scale": scale, "width": width, "joints": joints, "history": history,
            "records": int(train[0].shape[0] * train[0].shape[1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    options = parser.parse_args()
    torch.set_num_threads(4)
    logs = dict(np.load(DATA / "logs" / f"seed-{options.seed}.npz"))
    result = fit(logs, options.seed, epochs=options.epochs)
    folder = RUNS / f"seed-{options.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(result, folder / "fit.pt")
    (folder / "history.json").write_text(json.dumps(result["history"], indent=1))


if __name__ == "__main__":
    main()
