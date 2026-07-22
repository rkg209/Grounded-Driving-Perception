"""Re-exports so `from gdp.data import load_dataset` keeps working after the package split."""

from __future__ import annotations

from gdp.data.core import Box, Dataset, Sample, load_dataset

__all__ = ["Box", "Dataset", "Sample", "load_dataset"]
