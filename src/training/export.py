"""Export a trained residual model: the feature network as ONNX, the prior head
in torque units, and the settings recursive least squares starts from.

    python -m src.training.export --seed 0            # to runs/seed-0/
    python -m src.training.export --seed 0 --release  # also to models/
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from src.paths import MODELS, RUNS
from src.training.fit import ResidualModel


def export(result: dict, folder: Path) -> dict:
    joints, width = result["joints"], result["width"]
    mean, spread = result["model"]["mean"].numpy(), result["model"]["spread"].numpy()
    model = ResidualModel(mean, spread, joints, width)
    model.load_state_dict(result["model"])
    model.eval()

    folder.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 4 * joints)
    torch.onnx.export(model, example, folder / "dynamics.onnx", input_names=["state"], output_names=["features"],
                      dynamic_axes={"state": {0: "batch"}, "features": {0: "batch"}}, dynamo=False)

    # Training worked on targets divided by each joint's spread. Scaling the head's
    # columns by the same spread gives torques in N·m, and leaves least squares unchanged.
    head = (result["prior"] * result["scale"]).numpy().astype(np.float64)
    np.save(folder / "head.npy", head)

    probe = np.random.default_rng(0).standard_normal((64, 4 * joints)).astype(np.float32)
    session = onnxruntime.InferenceSession(str(folder / "dynamics.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        difference = np.abs(session.run(None, {"state": probe})[0] - model(torch.tensor(probe)).numpy()).max()
    return {"width": width, "prior": float(1 / result["log_strength"].exp()), "onnx_difference": float(difference),
            "records": result["records"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--release", action="store_true")
    options = parser.parse_args()
    folder = RUNS / f"seed-{options.seed}"
    summary = export(torch.load(folder / "fit.pt", weights_only=False), folder)
    (folder / "export.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    if options.release:
        MODELS.mkdir(exist_ok=True)
        for name in ("dynamics.onnx", "head.npy"):
            shutil.copy(folder / name, MODELS / name)


if __name__ == "__main__":
    main()
