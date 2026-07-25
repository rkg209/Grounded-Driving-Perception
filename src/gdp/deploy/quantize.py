"""Dynamic INT8 quantization via ONNX Runtime — no calibration data needed.

Dynamic quantization is tried first because it needs nothing beyond the exported fp32 graph
(plan decision 5). Static INT8, calibrated from *train* (never val — the same discipline as
`detector.sweep_candidates`), is the documented fallback if dynamic disappoints on accuracy or
latency; it is not implemented here because nothing so far has shown it's needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from gdp.paths import resolve


@dataclass(frozen=True)
class QuantizeResult:
    """Before/after size on disk — quantization's memory-footprint claim, as data."""

    onnx_path: Path
    fp32_size_bytes: int
    int8_size_bytes: int


def quantize_dynamic_int8(onnx_path: str | Path, out_path: str | Path) -> QuantizeResult:
    onnx_path = resolve(onnx_path)
    out_path = resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fp32_size_bytes = onnx_path.stat().st_size
    quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
    int8_size_bytes = out_path.stat().st_size

    return QuantizeResult(
        onnx_path=out_path, fp32_size_bytes=fp32_size_bytes, int8_size_bytes=int8_size_bytes
    )
