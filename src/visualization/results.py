"""Result figures, drawn only from the saved reports.

    python -m src.visualization.results
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.statistics import bootstrap
from src.paths import FIGURES, REPORTS
from src.visualization import style

GREY, DARK_GREY, LIGHT_GREY = "#8E8E93", "#636366", "#AEAEB2"
RICH, MEDIUM, SOFT, LIGHT = (style.SHADES[k] for k in ("rich", "medium", "soft", "light"))

BASELINES = [("pid", "PID"), ("computed-torque", "Computed\ntorque"), ("slotine-li", "Slotine–Li"),
             ("residual-sac", "Residual\nSAC"), ("learned-adaptive", "Adaptive\nmodel")]
COLORS = {"pid": GREY, "computed-torque": GREY, "slotine-li": GREY, "residual-sac": GREY, "learned-adaptive": RICH}


def report(name: str) -> dict:
    return json.loads((REPORTS / "test" / f"{name}.json").read_text())


def errors(name: str, ranges: str | None = None, seed: int | None = None) -> np.ndarray:
    """RMS errors shaped (seeds, scenarios), for one range of conditions if asked."""
    data = report(name)
    values = np.array(data["metrics"]["rms_mm"])
    if seed is not None:
        values = values[[data["seeds"].index(seed)]]
    if ranges is not None:
        values = values[:, [r == ranges for _, r in data["scenarios"]]]
    return values


def interval_bars(axis, labels, values, colors) -> None:
    """Rounded bars with their 95% intervals and values above them."""
    means = [v[0] for v in values]
    positions = np.arange(len(labels))
    axis.set_xlim(-0.6, len(labels) - 0.4)
    axis.set_ylim(0, max(v[2] for v in values) * 1.22)
    axis.set_xticks(positions, labels)
    style.rounded_bars(axis, positions, means, colors)
    for x, (mean, low, high) in zip(positions, values):
        axis.plot([x, x], [low, high], color=style.NEUTRAL["ink"], linewidth=0.8, alpha=0.55, zorder=3)
        axis.text(x, high + axis.get_ylim()[1] * 0.025, f"{mean:.1f}", ha="center", va="bottom", fontsize=8,
                  color=style.NEUTRAL["ink"])


def comparison() -> None:
    """RMS tool error within and beyond the training range, with the oracle's floor."""
    style.apply()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
    for axis, ranges, title in zip(axes, ("training", "beyond"), ("Within the training range", "Beyond the training range")):
        values = [bootstrap(errors(name, ranges)) for name, _ in BASELINES]
        interval_bars(axis, [label for _, label in BASELINES], values, [COLORS[name] for name, _ in BASELINES])
        floor = errors("oracle", ranges).mean()
        axis.axhline(floor, color=style.NEUTRAL["secondary"], linestyle=(0, (2, 3)), linewidth=1, zorder=1)
        axis.set_title(title)
    axes[0].set_ylabel("RMS tool error (mm)")
    top = max(axis.get_ylim()[1] for axis in axes)
    for axis in axes:
        axis.set_ylim(0, top)
    style.legend(figure, [("swatch", GREY, "Baseline"), ("swatch", RICH, "Adaptive model"),
                          ("dash", style.NEUTRAL["secondary"], "Oracle")], anchor=(0.98, 1.0))
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    style.save(figure, FIGURES, "comparison")


def parts() -> None:
    """What each part adds, on the first training seed, over all 40 test scenarios."""
    style.apply()
    steps = [("computed-torque", None, "Datasheet\nmodel", GREY),
             ("learned-fixed", 0, "Learned\nresidual", LIGHT),
             ("learned-feedback", 0, "Adapting,\nin feedback", SOFT),
             ("learned-adaptive", 0, "Adapting,\non reference", RICH)]
    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    values = [bootstrap(errors(name, seed=seed)) for name, seed, _, _ in steps]
    interval_bars(axis, [label for *_, label, _ in steps], values, [color for *_, color in steps])
    axis.set_ylabel("RMS tool error (mm)")
    figure.tight_layout()
    style.save(figure, FIGURES, "parts")


def trace() -> None:
    """Tool error through one pick, on a beyond-range scenario."""
    data = json.loads((REPORTS / "trace.json").read_text())
    style.apply()
    figure, axis = plt.subplots(figsize=(9.2, 3.1))
    lines = [("computed-torque", GREY, "-", 1.1, "swatch", "Computed torque"),
             ("residual-sac", LIGHT_GREY, (0, (2, 3)), 1.1, "dash", "Residual SAC"),
             ("learned-adaptive", RICH, "-", 1.8, "swatch", "Adaptive model"),
             ("oracle", DARK_GREY, (0, (2, 3)), 1.1, "dash", "Oracle")]
    for name, color, dash, width, _, _ in lines:
        values = np.array(data["errors"][name])
        time = np.arange(len(values)) * data["period"] - data["pick_at"]
        axis.plot(time, values, color=color, linestyle=dash, linewidth=width, zorder=3 if name == "learned-adaptive" else 2)
    top = max(max(values) for values in (data["errors"][name] for name, *_ in lines)) * 1.2
    axis.set_ylim(0, top)
    axis.axvline(0, color=style.NEUTRAL["tertiary"], linewidth=1, linestyle=(0, (1, 2)), zorder=1)
    axis.text(0.03, top * 0.95, f"Picks {data['payload_kg']:.1f} kg", fontsize=8, va="top", color=style.NEUTRAL["secondary"])
    axis.set_xlabel("Time from the pick (s)")
    axis.set_ylabel("Tool error (mm)")
    axis.set_xlim(time[0], time[-1])
    style.legend(figure, [(kind, color, label) for _, color, _, _, kind, label in lines], anchor=(0.98, 1.0))
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    style.save(figure, FIGURES, "trace")


def memory() -> None:
    """Validation error against how long the adaptation remembers (1 / (1 − forgetting) steps at 1 kHz)."""
    trials = json.loads((REPORTS / "tuning.json").read_text())["trials"]["forgetting"]
    chosen = json.loads((REPORTS / "tuning.json").read_text())["chosen"]["forgetting"]
    factors = sorted(trials, key=float)
    labels = [f"{1 / (1 - float(f)) / 1000:g} s" for f in factors]
    style.apply()
    figure, axis = plt.subplots(figsize=(4.4, 3.0))
    positions = np.arange(len(factors))
    axis.set_xlim(-0.6, len(factors) - 0.4)
    axis.set_ylim(0, max(trials.values()) * 1.18)
    axis.set_xticks(positions, labels)
    heights = [trials[f] for f in factors]
    style.rounded_bars(axis, positions, heights, [RICH if float(f) == chosen else SOFT for f in factors])
    for x, h in zip(positions, heights):
        axis.text(x, h + axis.get_ylim()[1] * 0.02, f"{h:.1f}", ha="center", va="bottom", fontsize=8, color=style.NEUTRAL["ink"])
    axis.set_xlabel("Memory of the adaptation")
    axis.set_ylabel("Validation RMS error (mm)")
    figure.tight_layout()
    style.save(figure, FIGURES, "memory")


if __name__ == "__main__":
    comparison()
    trace()
    memory()
    parts()
