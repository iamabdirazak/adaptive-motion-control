"""What each episode is scored on."""
import numpy as np

from src.evaluation.rollout import Log


def transient(log: Log, window: float = 0.5) -> float:
    """RMS tool error (mm) in the first `window` seconds after each pick or place,
    when the payload has just changed and the model has had the least time to follow."""
    changes = np.flatnonzero(np.diff(log.holding.astype(int))) + 1
    steps = int(window / log.period)
    errors = np.concatenate([log.error[start:start + steps] for start in changes]) if len(changes) else log.error
    return float(np.sqrt(np.mean(errors**2)))


def score(log: Log) -> dict[str, float]:
    return {
        "rms_mm": float(np.sqrt(np.mean(log.error**2))),
        "peak_mm": float(np.max(log.error)),
        "transient_mm": transient(log),
        "energy": float(np.sum(log.torque**2) * log.period),  # N²·m²·s, ∫τ² dt over all joints
        "saturated": float(np.mean(log.saturated)),           # share of steps at a torque limit
        "compute_us": float(np.median(log.compute)),
    }
