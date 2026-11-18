"""Confidence intervals that respect where the variation comes from.

Results vary between training seeds and between scenarios. A hierarchical
bootstrap resamples seeds first, then scenarios within each resampled seed,
so the interval covers both. Comparisons use paired differences on the same
scenarios, which removes the variation the controllers share.
"""
import numpy as np


def bootstrap(values: np.ndarray, resamples: int = 10_000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Mean and confidence interval of `values`, shaped (seeds, scenarios)."""
    values = np.atleast_2d(values)
    seeds, scenarios = values.shape
    rng = np.random.default_rng(seed)
    chosen_seeds = rng.integers(seeds, size=(resamples, seeds))
    chosen_scenarios = rng.integers(scenarios, size=(resamples, seeds, scenarios))
    means = values[chosen_seeds[..., None], chosen_scenarios].mean(axis=(1, 2))
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(values.mean()), float(low), float(high)


def paired(values: np.ndarray, baseline: np.ndarray, **options) -> tuple[float, float, float]:
    """Mean difference to a baseline on the same scenarios, with its interval.
    A deterministic baseline, shaped (1, scenarios), is compared against every seed."""
    return bootstrap(np.atleast_2d(values) - np.atleast_2d(baseline), **options)
