"""Spec 03 design decision 9 — the open-vocabulary forgetting probe. A demo, labelled one (H9)."""

from gdp.probe.openvocab import (
    load_probe_images,
    load_probe_phrases,
    run_probe,
    write_probe_result,
)

__all__ = [
    "load_probe_images",
    "load_probe_phrases",
    "run_probe",
    "write_probe_result",
]
