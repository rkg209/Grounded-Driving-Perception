"""The official nuScenes `v1.0-trainval` scene split, and the token→name join to it.

Spec 06 preserves DriveLM's train/val partition by *resolving* it from nuScenes' own published
scene lists rather than choosing one ourselves (spec 06 design decision 1, H2/H9). The lists
themselves live in `nuscenes_splits.json`, vendored verbatim from `nuscenes-devkit` with a
provenance header — see that file and `tests/test_drivelm_split.py` for the drift check.

DriveLM keys scenes by 32-hex nuScenes *scene token*; the official split lists key by scene
*name* (`"scene-0061"`). `load_scene_meta` reads the join table out of nuScenes' `scene.json`
metadata file (design decision 2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gdp.paths import resolve

_SPLITS_PATH = Path(__file__).parent / "nuscenes_splits.json"


@dataclass(frozen=True)
class OfficialSplits:
    train: frozenset[str]
    val: frozenset[str]
    provenance: dict


def load_official_splits() -> OfficialSplits:
    """Load the vendored official nuScenes train/val scene-*name* lists."""
    data = json.loads(_SPLITS_PATH.read_text())
    train = frozenset(data["train"])
    val = frozenset(data["val"])
    overlap = train & val
    if overlap:
        raise ValueError(f"vendored nuscenes_splits.json has train/val overlap: {sorted(overlap)}")
    return OfficialSplits(train=train, val=val, provenance=data["provenance"])


def load_scene_meta(path: str | Path) -> dict[str, str]:
    """Read nuScenes `scene.json` and return a scene *token* → scene *name* map.

    `scene.json` is a JSON array of records with (at least) `"token"` and `"name"` keys — the
    format nuScenes' metadata blob ships in.
    """
    scenes = json.loads(resolve(path).read_text())
    if not isinstance(scenes, list):
        raise TypeError(f"scene.json must be a JSON array, got {type(scenes).__name__}")
    token_to_name: dict[str, str] = {}
    for record in scenes:
        token = record["token"]
        name = record["name"]
        if token in token_to_name and token_to_name[token] != name:
            raise ValueError(
                f"scene.json has conflicting names for token {token!r}: "
                f"{token_to_name[token]!r} vs {name!r}"
            )
        token_to_name[token] = name
    return token_to_name


def resolve_split(scene_token: str, *, scene_meta: dict[str, str], splits: OfficialSplits) -> str:
    """Resolve a DriveLM scene token to `"train"` or `"val"` via the official nuScenes partition.

    A scene token absent from `scene_meta` or from both official lists is a hard error (design
    decision 2): silently dropping scenes would shrink the split every later number is computed
    on, which is exactly the failure this function exists to prevent.
    """
    if scene_token not in scene_meta:
        raise KeyError(f"scene token {scene_token!r} not found in scene.json")
    scene_name = scene_meta[scene_token]
    if scene_name in splits.train:
        return "train"
    if scene_name in splits.val:
        return "val"
    raise KeyError(
        f"scene {scene_name!r} (token {scene_token!r}) is in neither the official nuScenes "
        "v1.0-trainval train nor val scene list"
    )
