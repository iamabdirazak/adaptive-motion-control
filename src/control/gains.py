"""Feedback gains from a natural frequency and damping ratio, the same for every joint
of the model-based controllers, so their differences come only from the model they use."""
import numpy as np


def second_order(joints: int, frequency: float, damping: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Stiffness and damping gains that make the tracking error decay like a mass-spring
    system with the given natural frequency (rad/s) and damping ratio."""
    return np.full(joints, frequency**2), np.full(joints, 2 * damping * frequency)
