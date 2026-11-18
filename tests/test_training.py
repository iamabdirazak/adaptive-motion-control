import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.system import OnnxFeatures
from src.training.export import export
from src.training.fit import fit


def synthetic_logs(episodes: int = 4, length: int = 400) -> dict:
    """Residuals that depend on the state and change between episodes, like a payload."""
    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((episodes * length, 24)).astype(np.float32)
    gain = np.repeat(rng.uniform(0.5, 2.0, episodes), length)[:, None]
    residuals = (gain * np.tanh(inputs[:, :6]) + 0.01 * rng.standard_normal((len(inputs), 6))).astype(np.float32)
    return {"inputs": inputs, "residuals": residuals, "episode": np.repeat(np.arange(episodes), length)}


def test_training_exports_a_model_that_onnx_runtime_reproduces(tmp_path):
    torch.set_num_threads(1)
    result = fit(synthetic_logs(), seed=0, epochs=2)
    summary = export(result, tmp_path)
    assert summary["onnx_difference"] < 1e-4 and summary["prior"] > 0
    features = OnnxFeatures(tmp_path / "dynamics.onnx")(np.zeros(24, dtype=np.float32))
    assert features.shape == (32,) and features[-1] == pytest.approx(1.0)
    assert np.load(tmp_path / "head.npy").shape == (32, 6)
