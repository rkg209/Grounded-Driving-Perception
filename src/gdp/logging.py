"""Project logging: one format, everywhere, so cluster logs and laptop logs read the same."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str = "gdp", level: int = logging.INFO) -> logging.Logger:
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root = logging.getLogger("gdp")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _configured = True
    return logging.getLogger(name)
