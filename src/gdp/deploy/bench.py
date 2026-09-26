"""Latency/memory benchmarking: warmup discarded, tail latency reported, hardware pinned.

A latency number without p95/p99 hides the tail that matters for a perception loop, and a latency
number without discarding warmup or recording hardware is not a benchmark (the onnx-export
skill). This module is pure timing/percentile logic against an arbitrary zero-arg callable — it
has no opinion on what is being timed, so it is provable against a stub
(`tests/test_deploy_bench.py`) before spec 05 task 3 wires it to a real PyTorch/ONNX Runtime call.
"""

from __future__ import annotations

import platform
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchResult:
    """p50/p95/p99/fps over the timed iterations only — warmup never enters this.

    `peak_rss_bytes_process_wide` is `ru_maxrss` for the whole benchmarking process, not this
    variant alone. `run_interleaved` benchmarks every variant inside one process on purpose (to
    cancel thermal bias — see its docstring), so every variant's `BenchResult` in a given run
    reports the *same* number. It is real memory pressure, just not a per-variant measurement;
    never quote it as one (spec 05 Action 4).
    """

    variant: str
    p50: float
    p95: float
    p99: float
    fps: float
    peak_rss_bytes_process_wide: int
    warmup_iters: int
    timed_iters: int
    interleaved: bool = False


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile over an already-sorted array — no scipy dependency needed."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    idx = min(len(sorted_values) - 1, round(pct * (len(sorted_values) - 1)))
    return sorted_values[idx]


def _peak_rss_bytes() -> int:
    """`ru_maxrss` units differ by platform: bytes on macOS (Darwin), KB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if platform.system() == "Darwin" else rss * 1024


def _summarize(
    variant: str,
    timings: list[float],
    *,
    warmup: int,
    iters: int,
    interleaved: bool,
) -> BenchResult:
    timings = sorted(timings)
    total_time = sum(timings)
    fps = iters / total_time if total_time > 0 else float("inf")
    return BenchResult(
        variant=variant,
        p50=_percentile(timings, 0.50),
        p95=_percentile(timings, 0.95),
        p99=_percentile(timings, 0.99),
        fps=fps,
        peak_rss_bytes_process_wide=_peak_rss_bytes(),
        warmup_iters=warmup,
        timed_iters=iters,
        interleaved=interleaved,
    )


def benchmark(
    fn: Callable[[], object],
    *,
    variant: str,
    warmup: int = 50,
    iters: int = 200,
) -> BenchResult:
    """Call `fn()` `warmup` times (discarded), then `iters` times (timed)."""
    for _ in range(warmup):
        fn()

    timings: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)

    return _summarize(variant, timings, warmup=warmup, iters=iters, interleaved=False)


def run_interleaved(
    variants: dict[str, Callable[[], object]],
    *,
    warmup: int = 50,
    iters: int = 200,
) -> dict[str, BenchResult]:
    """Round-robin every variant together, one call each per round, to cancel thermal bias.

    Benchmarking variants sequentially lets whichever runs last absorb any thermal throttling the
    earlier ones triggered on the laptop's CPU. Interleaving spreads that bias evenly across all
    variants instead of concentrating it on one. Warmup rounds are interleaved too, then
    discarded, so the timed rounds start from the same thermal state for every variant.
    """
    if not variants:
        raise ValueError("run_interleaved requires at least one variant")

    names = list(variants.keys())
    timings: dict[str, list[float]] = {name: [] for name in names}

    for round_idx in range(warmup + iters):
        for name in names:
            start = time.perf_counter()
            variants[name]()
            elapsed = time.perf_counter() - start
            if round_idx >= warmup:
                timings[name].append(elapsed)

    return {
        name: _summarize(name, timings[name], warmup=warmup, iters=iters, interleaved=True)
        for name in names
    }


def collect_hardware_info(
    *,
    batch_size: int,
    image_size: tuple[int, int],
    execution_target: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A latency number without its hardware is meaningless (the onnx-export skill's words) —
    written alongside every `latency.json` (`gdp.deploy.metrics.write_latency`).

    `execution_target` names, per variant, the actual torch device or onnxruntime execution
    provider used (e.g. `{"fp32-pt": "cpu", "fp32-onnx": "CPUExecutionProvider"}`). Without it,
    a multi-variant latency comparison can silently mix GPU and CPU execution and nothing in the
    artifact discloses it — the exact bug spec 05 Action 4 fixed (PyTorch resolved to MPS while
    both ONNX variants ran on CPU, so "PyTorch vs ONNX" was actually "MPS vs CPU").
    """
    import os

    import onnxruntime
    import torch

    return {
        "onnxruntime_version": onnxruntime.__version__,
        "torch_version": torch.__version__,
        "processor": platform.processor(),
        "thread_count": os.cpu_count() or 1,
        "batch_size": batch_size,
        "image_size": list(image_size),
        "execution_target": execution_target or {},
    }
