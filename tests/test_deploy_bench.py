"""Pure harness logic against fake, zero-cost callables — no real model (CLAUDE.md §4)."""

from __future__ import annotations

import time

import pytest

from gdp.deploy.bench import BenchResult, benchmark, run_interleaved


def test_benchmark_excludes_warmup_from_timings():
    """A slow warmup must not leak into the reported percentiles."""
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        if calls["n"] <= 5:
            time.sleep(0.02)

    result = benchmark(fn, variant="fp32-pt", warmup=5, iters=20)

    assert calls["n"] == 25
    assert result.p99 < 0.02
    assert result.warmup_iters == 5
    assert result.timed_iters == 20


def test_percentile_ordering_always_holds():
    result = benchmark(lambda: None, variant="fp32-pt", warmup=2, iters=50)
    assert result.p50 <= result.p95 <= result.p99


def test_benchmark_result_shape():
    result = benchmark(lambda: None, variant="fp32-onnx", warmup=1, iters=5)
    assert isinstance(result, BenchResult)
    assert result.variant == "fp32-onnx"
    assert result.fps > 0
    assert result.peak_rss_bytes > 0
    assert result.interleaved is False


def test_run_interleaved_requires_at_least_one_variant():
    with pytest.raises(ValueError, match="at least one variant"):
        run_interleaved({})


def test_run_interleaved_is_round_robin():
    order: list[str] = []
    variants = {
        "fp32-pt": lambda: order.append("fp32-pt"),
        "fp32-onnx": lambda: order.append("fp32-onnx"),
        "int8-onnx": lambda: order.append("int8-onnx"),
    }

    run_interleaved(variants, warmup=2, iters=3)

    # 2 warmup rounds + 3 timed rounds, each round calling all three variants in order.
    assert order == ["fp32-pt", "fp32-onnx", "int8-onnx"] * 5


def test_run_interleaved_excludes_warmup_and_returns_all_variants():
    calls = {"a": 0, "b": 0}

    def slow_a() -> None:
        calls["a"] += 1
        if calls["a"] <= 4:
            time.sleep(0.02)

    def fast_b() -> None:
        calls["b"] += 1

    results = run_interleaved({"a": slow_a, "b": fast_b}, warmup=4, iters=10)

    assert set(results) == {"a", "b"}
    assert results["a"].p99 < 0.02
    assert results["a"].interleaved is True
    assert results["b"].interleaved is True
    for result in results.values():
        assert result.p50 <= result.p95 <= result.p99
        assert result.timed_iters == 10
        assert result.warmup_iters == 4
