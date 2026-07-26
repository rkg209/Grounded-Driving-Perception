"""Where each README table's numbers come from — one declaration per stage (spec 10 decision 1).

Two things are deliberate here:

1. **The run directory is resolved once, from the primary file.** `cli.py`'s
   `_latest_predictions_path` and `_latest_deploy_run_dir` both take mtime-max over
   `runs/<spec>/*/`; this generalises that idiom, but resolves a *directory* and then reads the
   secondary files from inside it. Joining a `metrics.json` from one run to a `latency.json` from
   another would silently pair an accuracy number with a latency number that never ran together.
2. **Every copied field is named.** The snapshot is a verbatim subset, not a reshaping — a field
   this module does not list cannot reach the README, and a field it lists is copied unmodified.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gdp.paths import repo_root, resolve

# Spec 03's zero-shot -> fine-tuned mAP comparison (gdp.eval.compare.build_comparison).
_STAGE1_COMPARISON_FIELDS = (
    "dataset",
    "split",
    "num_images",
    "box_threshold",
    "zeroshot_model_id",
    "finetuned_model_id",
    "zeroshot_created",
    "finetuned_created",
    "map_zeroshot",
    "map_finetuned",
    "map_delta",
    "map50_zeroshot",
    "map50_finetuned",
    "map50_delta",
    "per_class_ap",
    "regressions",
)

# Spec 04's grounding comparison (gdp.ground.compare.build_grounding_comparison). `caveat` and
# `is_self_built_benchmark` are copied so H9's label travels with the number into the table.
_GROUNDING_COMPARISON_FIELDS = (
    "phrases_sha256",
    "num_phrases",
    "box_threshold",
    "zeroshot_model_id",
    "finetuned_model_id",
    "zeroshot_created",
    "finetuned_created",
    "accuracy_zeroshot",
    "accuracy_finetuned",
    "accuracy_delta",
    "positives_accuracy_zeroshot",
    "positives_accuracy_finetuned",
    "negatives_accuracy_zeroshot",
    "negatives_accuracy_finetuned",
    "per_type",
    "regressions",
    "is_self_built_benchmark",
    "caveat",
)

# Spec 05 (gdp.deploy.metrics). `config` is dropped — it is the whole resolved Config blob and
# nothing in the deployment table reads it.
_DEPLOY_METRICS_FIELDS = (
    "dataset",
    "num_images",
    "is_synthetic",
    "fixed_prompt",
    "quantization",
    "prompt",
    "variants",
    "git_sha",
    "created",
)
_DEPLOY_LATENCY_FIELDS = ("hardware", "variants", "git_sha", "created")

# Spec 08 (gdp.vqa.compare.build_vqa_comparison).
_VQA_COMPARISON_FIELDS = (
    "val_jsonl_sha256",
    "num_items",
    "generation",
    "scorer_sha256",
    "final_score_composition",
    "final_score",
    "base_created",
    "finetuned_created",
    "overall_accuracy",
    "per_category_accuracy",
    "regressions",
)

# Spec 08's per-model record (gdp.vqa.score.build_vqa_metrics). `overall` is kept for its
# `submetrics.omitted` list (decision 7 — the omitted sub-metrics are rendered, not dropped);
# `per_category` is not, because the comparison file already carries per-category accuracy beside
# its baseline and a bare fine-tuned per-category number would be an H8 hazard sitting in the
# snapshot waiting to be quoted.
_VQA_METRICS_FIELDS = (
    "model_role",
    "num_items",
    "is_synthetic",
    "split_provenance",
    "caveat",
    "comparable_to_leaderboard",
    "hallucination",
    "overall",
    "git_sha",
    "created",
)


def _pick(data: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Copy the named fields verbatim. A missing field is simply absent — no default is invented,
    because an invented default is a number nobody measured."""
    return {name: data[name] for name in fields if name in data}


def extract_vqa_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Spec 08's `metrics.json` has two shapes (`cli.py`'s `vqa score`): compare mode nests both
    models under `models`, single-model mode is flat. Normalise to the nested shape so the table
    renderer has exactly one thing to read."""
    models = raw["models"] if "models" in raw else {raw.get("model_role", "single"): raw}
    return {
        "models": {role: _pick(node, _VQA_METRICS_FIELDS) for role, node in models.items()},
    }


@dataclass(frozen=True)
class SourceFile:
    """One JSON artifact a stage's table reads."""

    key: str
    filename: str
    fields: tuple[str, ...] = ()
    extract: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def subset(self, raw: dict[str, Any]) -> dict[str, Any]:
        return self.extract(raw) if self.extract is not None else _pick(raw, self.fields)


@dataclass(frozen=True)
class StageSource:
    """One README table's provenance declaration.

    `blocker` is what has to happen before this stage can carry a number. It is rendered verbatim
    into the pending row: a named blocker reads as a plan, a blank cell reads as a zero.
    """

    title: str
    spec: str
    files: tuple[SourceFile, ...]
    blocker: str
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def primary(self) -> SourceFile:
        return self.files[0]


STAGES: dict[str, StageSource] = {
    "stage1_detection": StageSource(
        title="Stage 1 · detection mAP",
        spec="03-finetune",
        files=(SourceFile("comparison", "comparison.json", _STAGE1_COMPARISON_FIELDS),),
        blocker="spec 03 cluster fine-tuning run (`sbatch scripts/slurm/finetune_detector.slurm`), "
        "which itself needs spec 02's zero-shot baseline on real BDD100K",
    ),
    "grounding": StageSource(
        title="Stage 1 · grounding accuracy",
        spec="04-grounding",
        files=(
            SourceFile("comparison", "grounding_comparison.json", _GROUNDING_COMPARISON_FIELDS),
        ),
        blocker="spec 04 hand-authoring of the 150–300 real phrases, then "
        "`sbatch scripts/slurm/grounding_eval.slurm`",
    ),
    "deployment": StageSource(
        title="Deployment · accuracy vs latency",
        spec="05-deploy",
        files=(
            SourceFile("metrics", "metrics.json", _DEPLOY_METRICS_FIELDS),
            SourceFile("latency", "latency.json", _DEPLOY_LATENCY_FIELDS),
        ),
        blocker="spec 05 re-run on the real BDD100K val split (today's only run used "
        "`tests/fixtures/mini_bdd/`)",
    ),
    "stage2_vqa": StageSource(
        title="Stage 2 · DriveLM VQA",
        spec="08-vqa",
        files=(
            SourceFile("comparison", "comparison.json", _VQA_COMPARISON_FIELDS),
            SourceFile("metrics", "metrics.json", extract=extract_vqa_metrics),
        ),
        blocker="spec 08 cluster evaluation run (`sbatch scripts/slurm/vqa_eval.slurm`), which "
        "itself needs spec 06's real DriveLM conversion and spec 07's LoRA adapter",
    ),
}


def latest_run_dir(spec: str, filename: str, *, runs_root: Path | None = None) -> Path | None:
    """The most recently modified `runs/<spec>/<ts>/` that actually contains `filename`.

    Mtime of the *file*, not the directory: a run directory's mtime changes when anything is
    written into it, so the newest directory is not necessarily the one with the newest result.
    """
    root = runs_root if runs_root is not None else resolve("runs")
    spec_dir = Path(root) / spec
    if not spec_dir.is_dir():
        return None
    candidates = [d for d in spec_dir.iterdir() if d.is_dir() and (d / filename).is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / filename).stat().st_mtime)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_relative(path: str | Path) -> str:
    """Repo-relative POSIX string for the snapshot, so a source path means the same thing on the
    laptop and on the cluster. Falls back to the absolute path when outside the repo (a test's
    tmp_path), which is still honest — it just isn't portable."""
    p = Path(path)
    try:
        return p.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return p.as_posix()
