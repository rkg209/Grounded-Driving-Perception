"""Grounded Driving Perception (gdp).

Two-stage vision-language driving perception:
  Stage 1 — open-vocabulary detection (Grounding-DINO, fine-tuned on BDD100K)
  Stage 2 — driving-scene VQA (Qwen2.5-VL-3B, LoRA fine-tuned on DriveLM)

The two stages are separate models, evaluated separately (see CLAUDE.md, H6).
"""

__version__ = "0.1.0"

from gdp.config import Config, load_config
from gdp.seed import select_device, set_seed

__all__ = ["Config", "load_config", "select_device", "set_seed", "__version__"]
