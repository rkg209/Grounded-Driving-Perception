"""The four README tables — pure functions of the snapshot (spec 10 decision 3).

Rules this module exists to enforce, mechanically rather than by remembering:

- **A missing stage renders a sentence with a named blocker, not a blank cell.** A blank reads as
  "zero" or "we didn't bother"; a named blocker reads as a plan.
- **`synthetic` renders exactly like `pending`.** A number computed on `tests/fixtures/` is not a
  result (H7) and there is no flag that turns it into one.
- **Nothing is dropped for looking bad.** Regressions get their own line, per-class rows include
  the classes that got worse, and the sub-metrics that could not be computed are listed with the
  reason they could not be (H7/H9).
- **Every present table ends with its provenance**: source path, sha256, split, item count, dates.
  A number without its split and its date is a rumour (CLAUDE.md §2).

The list-of-lines idiom mirrors `gdp.vqa.failures.write_failures_md`.
"""

from __future__ import annotations

from typing import Any

from gdp.config import DRIVELM_CATEGORIES
from gdp.deploy.metrics import REQUIRED_VARIANTS

BLOCK_NAMES: tuple[str, ...] = ("stage1-map", "grounding", "deployment", "stage2-vqa")

PENDING_TEMPLATE = (
    "> **Not yet measured.** Blocked on: **{blocker}**.\n"
    ">\n"
    "> This block is written by `uv run gdp report snapshot && uv run gdp report render` — "
    "never typed by hand."
)

NOT_SCORED = "not scored"
NOT_SCORED_NOTE = (
    f"`{NOT_SCORED}` = the official scorer returned no score for that subset "
    "(no items of that type in the split) — it is not a zero."
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or not isinstance(value, int | float) or isinstance(value, bool):
        return NOT_SCORED
    return f"{value:.{digits}f}"


def _delta(value: Any, digits: int = 3) -> str:
    if value is None or not isinstance(value, int | float) or isinstance(value, bool):
        return NOT_SCORED
    return f"{value:+.{digits}f}"


def _mib(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return NOT_SCORED
    return f"{value / (1024 * 1024):.0f}"


def _short_sha(value: Any, n: int = 12) -> str:
    return f"{str(value)[:n]}…" if value else "unknown"


def _pending_block(entry: dict[str, Any]) -> list[str]:
    # One list element per *line* — `inject`/`check` compare block bodies line by line, so an
    # element containing an embedded newline would never round-trip and `--check` would report
    # permanent drift.
    lines = PENDING_TEMPLATE.format(
        blocker=entry.get("blocker", "an unrecorded blocker")
    ).splitlines()
    if entry.get("status") == "synthetic":
        lines += [
            ">",
            f"> A run *does* exist (`{entry.get('source')}`), but {entry.get('reason')}. "
            "It is not rendered as a number, and no option makes it one.",
        ]
    return lines


def _regression_line(regressions: list[str] | None, *, unit: str) -> str:
    if not regressions:
        return f"**Regressions:** none — no {unit} lost ground under fine-tuning."
    listed = ", ".join(f"`{r}`" for r in regressions)
    one = len(regressions) == 1
    return (
        f"**Regressions:** {listed} — {'this' if one else 'these'} {unit}{'' if one else 's'} "
        f"got *worse* after fine-tuning, and {'stays' if one else 'stay'} in the table (H7)."
    )


def _provenance(entry: dict[str, Any], extras: list[str]) -> list[str]:
    parts = [f"Source: `{entry['source']}` (sha256 `{_short_sha(entry.get('sha256'))}`)", *extras]
    return ["", "_" + " · ".join(parts) + "._"]


def _table(
    header: list[str], rows: list[list[str]], *, left_cols: tuple[int, ...] = (0,)
) -> list[str]:
    """Numbers right-aligned, labels and prose left-aligned — a right-aligned reason column is
    unreadable, and the omitted-sub-metric reasons are a whole column of prose."""
    sep = ["---" if i in left_cols else "---:" for i in range(len(header))]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(sep) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


# --------------------------------------------------------------------------- Stage 1 · mAP


def render_stage1_map(entry: dict[str, Any]) -> list[str]:
    if entry.get("status") != "present":
        return _pending_block(entry)

    d = entry["data"]["comparison"]
    lines = _table(
        ["Metric", "Zero-shot (baseline)", "Fine-tuned", "Δ"],
        [
            [
                "mAP@[.50:.95]",
                _fmt(d.get("map_zeroshot")),
                _fmt(d.get("map_finetuned")),
                _delta(d.get("map_delta")),
            ],
            [
                "mAP@0.50",
                _fmt(d.get("map50_zeroshot")),
                _fmt(d.get("map50_finetuned")),
                _delta(d.get("map50_delta")),
            ],
        ],
    )

    per_class = d.get("per_class_ap") or {}
    if per_class:
        lines += ["", "Per class (AP@[.50:.95]), every class the split contains:", ""]
        lines += _table(
            ["Class", "Zero-shot", "Fine-tuned", "Δ"],
            [
                [
                    f"`{cls}`",
                    _fmt(v.get("before")),
                    _fmt(v.get("after")),
                    _delta(v.get("delta")),
                ]
                for cls, v in sorted(per_class.items())
            ],
        )

    lines += ["", _regression_line(d.get("regressions"), unit="class")]
    lines += _provenance(
        entry,
        [
            f"dataset `{d.get('dataset')}`, split `{d.get('split')}`",
            f"{d.get('num_images')} images",
            f"box_threshold {d.get('box_threshold')}",
            f"zero-shot run {d.get('zeroshot_created')}",
            f"fine-tuned run {d.get('finetuned_created')}",
        ],
    )
    return lines


# --------------------------------------------------------------------------- grounding


def render_grounding(entry: dict[str, Any]) -> list[str]:
    if entry.get("status") != "present":
        return _pending_block(entry)

    d = entry["data"]["comparison"]
    lines = _table(
        ["Grounding accuracy (IoU ≥ 0.5)", "Zero-shot (baseline)", "Fine-tuned", "Δ"],
        [
            [
                "Overall",
                _fmt(d.get("accuracy_zeroshot")),
                _fmt(d.get("accuracy_finetuned")),
                _delta(d.get("accuracy_delta")),
            ],
            [
                "Positive phrases",
                _fmt(d.get("positives_accuracy_zeroshot")),
                _fmt(d.get("positives_accuracy_finetuned")),
                _delta(
                    _sub(
                        d.get("positives_accuracy_finetuned"), d.get("positives_accuracy_zeroshot")
                    )
                ),
            ],
            [
                "Negative phrases (correct = no box)",
                _fmt(d.get("negatives_accuracy_zeroshot")),
                _fmt(d.get("negatives_accuracy_finetuned")),
                _delta(
                    _sub(
                        d.get("negatives_accuracy_finetuned"), d.get("negatives_accuracy_zeroshot")
                    )
                ),
            ],
        ],
    )

    per_type = d.get("per_type") or {}
    if per_type:
        lines += ["", "Per qualifier type:", ""]
        lines += _table(
            ["Qualifier type", "Zero-shot", "Fine-tuned", "Δ"],
            [
                [f"`{t}`", _fmt(v.get("before")), _fmt(v.get("after")), _delta(v.get("delta"))]
                for t, v in sorted(per_type.items())
            ],
        )

    lines += ["", _regression_line(d.get("regressions"), unit="qualifier type")]
    if d.get("is_self_built_benchmark"):
        # Verbatim from the artifact, not paraphrased — the label has to travel with the number.
        lines += ["", f"> **Self-built evaluation set.** {d.get('caveat')}"]
    lines += _provenance(
        entry,
        [
            f"{d.get('num_phrases')} frozen phrases "
            f"(sha256 `{_short_sha(d.get('phrases_sha256'))}`)",
            f"box_threshold {d.get('box_threshold')}",
            f"zero-shot run {d.get('zeroshot_created')}",
            f"fine-tuned run {d.get('finetuned_created')}",
        ],
    )
    return lines


def _sub(after: Any, before: Any) -> float | None:
    if isinstance(after, int | float) and isinstance(before, int | float):
        return after - before
    return None


# --------------------------------------------------------------------------- deployment


def render_deployment(entry: dict[str, Any]) -> list[str]:
    if entry.get("status") != "present":
        return _pending_block(entry)

    metrics = entry["data"]["metrics"]
    latency = entry["data"]["latency"]
    acc = metrics.get("variants") or {}
    lat = latency.get("variants") or {}

    # Spec 05's canonical order first (fp32-pt -> fp32-onnx -> int8-onnx), then anything else the
    # artifact happens to carry — an unexpected variant is shown, never silently dropped.
    ordered = [v for v in REQUIRED_VARIANTS if v in acc or v in lat]
    ordered += sorted((set(acc) | set(lat)) - set(ordered))

    rows = []
    for variant in ordered:
        a = acc.get(variant, {})
        latency_row = lat.get(variant, {})
        rows.append(
            [
                f"`{variant}`",
                _fmt(a.get("map")),
                _fmt(a.get("map50")),
                _fmt(latency_row.get("p50")),
                _fmt(latency_row.get("p95")),
                _fmt(latency_row.get("p99")),
                _fmt(latency_row.get("fps"), 2),
                _mib(latency_row.get("peak_rss_bytes")),
            ]
        )

    lines = _table(
        [
            "Variant",
            "mAP@[.50:.95]",
            "mAP@0.50",
            "p50 (s)",
            "p95 (s)",
            "p99 (s)",
            "FPS",
            "peak RSS (MiB)",
        ],
        rows,
    )

    hardware = latency.get("hardware") or {}
    lines += [
        "",
        "**Every latency row has its accuracy row beside it (H8)** — a speed win with no mAP "
        "number next to it is half a result.",
    ]
    if metrics.get("fixed_prompt"):
        lines += [
            "",
            "> **The exported graph is fixed-vocabulary, not open-vocabulary.** ONNX export "
            "freezes the text prompt, so the exported artifact answers only the frozen class "
            "list — recorded in the artifact as `fixed_prompt: true`, not discovered later. The "
            "open-vocabulary path stays in PyTorch.",
        ]
    lines += _provenance(
        entry,
        [
            f"latency from `{entry['sources'].get('latency', {}).get('path')}`",
            f"dataset `{metrics.get('dataset')}`, {metrics.get('num_images')} images",
            f"quantization `{metrics.get('quantization')}`",
            f"hardware: {_hardware(hardware)}",
            f"warmup discarded, {_timed_iters(lat)} timed iterations, interleaved",
            f"run {metrics.get('created')}",
        ],
    )
    return lines


def _hardware(hardware: dict[str, Any]) -> str:
    bits = [
        f"{hardware.get('processor', 'unknown')} CPU",
        f"{hardware.get('thread_count', '?')} threads",
        f"onnxruntime {hardware.get('onnxruntime_version', '?')}",
        f"torch {hardware.get('torch_version', '?')}",
        f"batch {hardware.get('batch_size', '?')}",
    ]
    size = hardware.get("image_size")
    if isinstance(size, list) and len(size) == 2:
        bits.append(f"input {size[0]}×{size[1]}")
    return ", ".join(bits)


def _timed_iters(lat: dict[str, Any]) -> Any:
    for row in lat.values():
        if "timed_iters" in row:
            return row["timed_iters"]
    return "?"


# --------------------------------------------------------------------------- Stage 2 · VQA


def render_stage2_vqa(entry: dict[str, Any]) -> list[str]:
    if entry.get("status") != "present":
        return _pending_block(entry)

    d = entry["data"]["comparison"]
    models = (entry["data"].get("metrics") or {}).get("models") or {}
    base = models.get("base", {})
    finetuned = models.get("finetuned", {})

    overall = d.get("overall_accuracy") or {}
    lines = _table(
        ["DriveLM `accuracy` sub-metric", "Base VLM (baseline)", "LoRA fine-tuned", "Δ"],
        [
            [
                "Overall",
                _fmt(overall.get("before")),
                _fmt(overall.get("after")),
                _delta(overall.get("delta")),
            ]
        ],
    )

    per_category = d.get("per_category_accuracy") or {}
    if per_category:
        categories = [c for c in DRIVELM_CATEGORIES if c in per_category]
        categories += [c for c in sorted(per_category) if c not in categories]
        lines += [
            "",
            "Per official DriveLM category — `planning` and `behavior` **are** the risk-reasoning "
            "result; there is no separate risk engine and no separate risk metric (H9):",
            "",
        ]
        lines += _table(
            ["Category", "Base VLM", "LoRA fine-tuned", "Δ"],
            [
                [
                    f"`{c}`",
                    _fmt(per_category[c].get("before")),
                    _fmt(per_category[c].get("after")),
                    _delta(per_category[c].get("delta")),
                ]
                for c in categories
            ],
        )

    lines += ["", _regression_line(d.get("regressions"), unit="category")]
    lines += ["", NOT_SCORED_NOTE]

    lines += [
        "",
        "**Official composite `final_score`: not computed — it is `null`, deliberately.**",
    ]
    lines += [
        "",
        f"Composition, verbatim from the vendored scorer: `{d.get('final_score_composition')}`",
        "",
    ]
    lines += _table(
        ["Official sub-metric", "Items", "Status"],
        _submetric_rows(finetuned or base),
        left_cols=(0, 2),
    )
    lines += [
        "",
        "Two of the four weighted components need a paid, non-deterministic OpenAI judge this "
        "repo never calls, so the composite cannot be computed honestly. It is reported as "
        "missing, with the reason, rather than silently replaced by the two components that do "
        'run — see `third_party/drivelm/PROVENANCE.md`\'s "Known gap".',
    ]

    lines += _hallucination_lines(base, finetuned)

    lines += ["", f"> **Not comparable to the DriveLM leaderboard.** {base.get('caveat', '')}"]
    lines += _provenance(
        entry,
        [
            f"{d.get('num_items')} val items",
            f"val split sha256 `{_short_sha(d.get('val_jsonl_sha256'))}`",
            f"split provenance: {base.get('split_provenance')}",
            f"scorer sha256 `{_short_sha(d.get('scorer_sha256'))}`",
            f"decoding: {_decoding(d.get('generation'))}",
            f"base run {d.get('base_created')}",
            f"fine-tuned run {d.get('finetuned_created')}",
        ],
    )
    return lines


def _decoding(generation: Any) -> str:
    if not isinstance(generation, dict):
        return "unrecorded"
    return ", ".join(f"{k}={v}" for k, v in generation.items())


def _submetric_rows(node: dict[str, Any]) -> list[list[str]]:
    """Computed *and* omitted, in that order. Dropping the uncomputable rows is precisely the
    reflex spec 10's honesty contract names (decision 7)."""
    submetrics = (node.get("overall") or {}).get("submetrics") or {}
    rows: list[list[str]] = []
    for item in submetrics.get("computed", []):
        score = item.get("score")
        status = (
            f"computed — {_fmt(score)}"
            if isinstance(score, int | float) and not isinstance(score, bool)
            else "computed (corpus-level, see the artifact for the per-metric breakdown)"
            if isinstance(score, dict)
            else f"computed — {NOT_SCORED}"
        )
        rows.append([f"`{item.get('name')}`", str(item.get("n_items")), status])
    for item in submetrics.get("omitted", []):
        rows.append(
            [
                f"`{item.get('name')}`",
                str(item.get("n_items")),
                f"**omitted** — {item.get('reason')}",
            ]
        )
    return rows


def _hallucination_lines(base: dict[str, Any], finetuned: dict[str, Any]) -> list[str]:
    base_h = (base.get("hallucination") or {}).get("overall")
    ft_h = (finetuned.get("hallucination") or {}).get("overall")
    if not base_h and not ft_h:
        return []
    base_h = base_h or {}
    ft_h = ft_h or {}
    lines = ["", "Hallucination diagnostic (base vs fine-tuned, same split):", ""]
    lines += _table(
        ["Diagnostic", "Base VLM", "LoRA fine-tuned", "Δ"],
        [
            [
                "Ungrounded object-tag rate",
                _fmt(base_h.get("ungrounded_tag_rate")),
                _fmt(ft_h.get("ungrounded_tag_rate")),
                _delta(_sub(ft_h.get("ungrounded_tag_rate"), base_h.get("ungrounded_tag_rate"))),
            ],
            [
                "Answers with ≥1 ungrounded tag",
                _fmt(base_h.get("items_with_hallucination_rate")),
                _fmt(ft_h.get("items_with_hallucination_rate")),
                _delta(
                    _sub(
                        ft_h.get("items_with_hallucination_rate"),
                        base_h.get("items_with_hallucination_rate"),
                    )
                ),
            ],
        ],
    )
    caveat = (finetuned.get("hallucination") or base.get("hallucination") or {}).get("caveat")
    if caveat:
        lines += ["", f"> **Not an official DriveLM metric.** {caveat} (H9)."]
    return lines


RENDERERS = {
    "stage1-map": ("stage1_detection", render_stage1_map),
    "grounding": ("grounding", render_grounding),
    "deployment": ("deployment", render_deployment),
    "stage2-vqa": ("stage2_vqa", render_stage2_vqa),
}


def render_all(snapshot: dict[str, Any]) -> dict[str, list[str]]:
    """Marker name -> the lines that belong between that marker pair."""
    stages = snapshot.get("stages") or {}
    out: dict[str, list[str]] = {}
    for block, (stage_name, renderer) in RENDERERS.items():
        entry = stages.get(stage_name)
        if entry is None:
            raise KeyError(
                f"snapshot has no stage {stage_name!r} for block {block!r} — "
                "re-run `uv run gdp report snapshot`"
            )
        out[block] = renderer(entry)
    return out
