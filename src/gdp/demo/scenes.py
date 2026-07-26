"""The scene registry: the images the demo's dropdown offers, each carrying its own honesty badge.

Only `tests/fixtures/mini_bdd/` ships in the repo today (plan fact 4) — real BDD100K images are not
checked in. A scene loaded from *any* path containing a `fixtures` directory component is treated
as synthetic and badged `SYNTHETIC_SCENE`; anything else (e.g. a Gradio file upload, which lands
outside the repo tree) gets a neutral, non-claiming badge instead — never labelled "real BDD100K",
since nothing here verifies that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gdp.demo.honesty import SYNTHETIC_SCENE
from gdp.probe.openvocab import IMAGE_SUFFIXES

UPLOADED_SCENE_BADGE = "uploaded image — provenance not verified"


@dataclass(frozen=True)
class Scene:
    name: str
    path: Path
    is_synthetic: bool


def _is_synthetic(path: Path) -> bool:
    return "fixtures" in path.parts


def load_scenes(scenes_dir: str | Path) -> list[Scene]:
    """Every image file directly under `scenes_dir`, sorted by name."""
    root = Path(scenes_dir)
    if not root.is_dir():
        return []
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [Scene(name=p.stem, path=p, is_synthetic=_is_synthetic(p)) for p in paths]


def scene_badge(scene: Scene) -> str:
    return SYNTHETIC_SCENE if scene.is_synthetic else UPLOADED_SCENE_BADGE
