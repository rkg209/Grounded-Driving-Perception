"""`plot_accuracy_vs_latency` against fake variants — no model needed."""

from __future__ import annotations

import pytest

from gdp.deploy.curve import VariantPoint, plot_accuracy_vs_latency


def test_plot_writes_a_nonempty_png(tmp_path):
    variants = {
        "fp32-pt": VariantPoint(map=0.42, p50=0.30),
        "fp32-onnx": VariantPoint(map=0.42, p50=0.18),
        "int8-onnx": VariantPoint(map=0.39, p50=0.09),
    }
    out_path = tmp_path / "curve.png"

    result = plot_accuracy_vs_latency(variants, out_path)

    assert result == out_path
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_requires_at_least_one_variant(tmp_path):
    with pytest.raises(ValueError, match="at least one variant"):
        plot_accuracy_vs_latency({}, tmp_path / "curve.png")
