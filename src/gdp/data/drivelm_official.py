"""DriveLM's official scorer-routing tags, from its own vendored `challenge/extract_data.py`.

DriveLM's answered train file carries **no** `tag` on any QA item (progress_report [SEQ-0143]).
The tags the vendored scorer routes on (`[0]` accuracy, `[1]` ChatGPT judge, `[2]` language,
`[3]` match) are assigned by DriveLM's `extract_data.py`, which also *selects* the handful of QA
per frame that its challenge evaluates. This module runs that file verbatim and reads its output
back. It assigns no tag itself: inventing routing rules would make our Stage-2 number the product
of a benchmark we wrote (H2/H9). See `third_party/drivelm/PROVENANCE.md`.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gdp.paths import repo_root

EXTRACT_DATA_PATH = repo_root() / "third_party" / "drivelm" / "extract_data.py"

# (scene_token, frame_token, category, question, answer) -> the tag list extract_data assigned.
TagKey = tuple[str, str, str, str, str]


def extract_data_sha256() -> str:
    """Stamped into the conversion stats, so a divergence from the pinned file is visible."""
    return hashlib.sha256(EXTRACT_DATA_PATH.read_bytes()).hexdigest()


def _load_extract_data() -> Callable[[str, str], None]:
    spec = importlib.util.spec_from_file_location("drivelm_extract_data", EXTRACT_DATA_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored {EXTRACT_DATA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_data


def official_eval_tags(raw: dict[str, Any]) -> tuple[dict[TagKey, list[int]], int]:
    """Run the vendored `extract_data(root_path, save_path)` on `raw` and map its selections back.

    Returns `(tags, n_entries)`. `n_entries` counts extract_data's output entries as written,
    including any QA it appended twice (e.g. a planning question matching two of its patterns).
    `tags` is keyed by content, because extract_data's output does not keep list indices; a QA
    whose (question, answer) repeats within one frame/category gets the tag on every copy.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "train.json"
        dst = Path(tmp) / "extracted.json"
        src.write_text(json.dumps(raw))
        # The vendored file print()s every frame's classes/locations (~8k lines on the real file);
        # captured here rather than edited out, since third_party/ is never modified.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _load_extract_data()(str(src), str(dst))
        except KeyError as exc:
            raise ValueError(
                f"DriveLM's extract_data.py needs every key frame to carry 'key_object_infos', "
                f"'image_paths' and all four QA categories; a frame is missing {exc}"
            ) from exc
        extracted = json.loads(dst.read_text())

    tags: dict[TagKey, list[int]] = {}
    n_entries = 0
    for scene_token, scene in extracted.items():
        for frame_token, frame in scene["key_frames"].items():
            for category, qa_list in frame["QA"].items():
                for qa in qa_list:
                    n_entries += 1
                    key = (scene_token, frame_token, category, qa["Q"], qa["A"])
                    tags[key] = list(qa["tag"])
    return tags, n_entries
