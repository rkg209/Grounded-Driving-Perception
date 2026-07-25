"""The accuracy-vs-latency curve — spec 05's literal deliverable.

One point per variant, mAP against p50 latency. This plot *is* the H8 guard given a shape: a
latency win with no accuracy point beside it, on the same axes, is a half-truth nobody can miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this runs on the laptop CLI, never a notebook kernel

import matplotlib.pyplot as plt  # noqa: E402 — must follow matplotlib.use("Agg")

from gdp.paths import resolve  # noqa: E402


@dataclass(frozen=True)
class VariantPoint:
    """One variant's position on the curve: accuracy (mAP@[0.50:0.95]) vs. p50 latency."""

    map: float
    p50: float


def plot_accuracy_vs_latency(variants: dict[str, VariantPoint], out_path: str | Path) -> Path:
    if not variants:
        raise ValueError("plot_accuracy_vs_latency requires at least one variant")

    out_path = resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for name, point in variants.items():
        ax.scatter(point.p50, point.map)
        ax.annotate(name, (point.p50, point.map))
    ax.set_xlabel("p50 latency (s)")
    ax.set_ylabel("mAP@[0.50:0.95]")
    ax.set_title("Accuracy vs. latency")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
