"""Online adaptation of the residual's linear head by recursive least squares.

Each control step brings one new measurement of the residual torque. The head
is updated to fit it, with a forgetting factor that lets old measurements fade,
so the model follows the arm as its payload, friction and motors change.
"""
import numpy as np


class RecursiveLeastSquares:
    def __init__(self, head: np.ndarray, forgetting: float = 0.995, prior: float = 1.0, ceiling: float = 1e3):
        self.initial = head.copy()
        self.forgetting, self.prior, self.ceiling = forgetting, prior, ceiling
        self.reset()

    def reset(self) -> None:
        self.head = self.initial.copy()                        # features × joints
        self.covariance = self.prior * np.eye(len(self.head))  # uncertainty of the head

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features @ self.head

    def update(self, features: np.ndarray, residual: np.ndarray) -> None:
        spread = self.covariance @ features
        gain = spread / (self.forgetting + features @ spread)
        self.head += np.outer(gain, residual - features @ self.head)
        self.covariance = (self.covariance - np.outer(gain, spread)) / self.forgetting
        # When the arm moves little, forgetting would inflate the covariance without bound
        # and the next movement would throw the head off. A ceiling on its trace prevents that.
        trace = np.trace(self.covariance)
        if trace > self.ceiling:
            self.covariance *= self.ceiling / trace
