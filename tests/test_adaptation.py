import numpy as np

from src.dynamics.adaptation import RecursiveLeastSquares
from src.evaluation.statistics import bootstrap, paired


def test_rls_recovers_a_known_head_and_follows_a_change():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(8, 2))
    estimator = RecursiveLeastSquares(np.zeros((8, 2)), forgetting=0.98)
    for step in range(600):
        if step == 300:
            truth = truth + 1.0  # the arm changes mid-run
        features = rng.normal(size=8)
        estimator.update(features, features @ truth)
    # Data from before the change still weighs 0.98³⁰⁰ ≈ 0.2%, so the head sits that close to the new truth.
    assert np.allclose(estimator.head, truth, atol=0.01)


def test_bootstrap_interval_contains_the_mean_and_narrows_with_data():
    rng = np.random.default_rng(1)
    small = rng.normal(5, 1, size=(5, 10))
    large = rng.normal(5, 1, size=(5, 200))
    mean, low, high = bootstrap(small)
    assert low < mean < high
    assert bootstrap(large)[2] - bootstrap(large)[1] < high - low


def test_paired_difference_against_a_deterministic_baseline():
    baseline = np.arange(10.0)[None]
    mean, low, high = paired(baseline + 2.0 + np.zeros((5, 10)), baseline)
    assert np.isclose(mean, 2.0) and np.isclose(low, 2.0) and np.isclose(high, 2.0)
