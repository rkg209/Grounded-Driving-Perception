"""Adapter around the vendored DriveLM challenge scorer (`third_party/drivelm/evaluation.py`).

H2/H9 (CLAUDE.md §2): this module may reshape our predictions into the scorer's expected input and
read its output back out — it computes zero scores itself. See `third_party/drivelm/PROVENANCE.md`
for where the vendored file came from, how it routes/weights sub-metrics, and why
`run_official_scorer` below never calls the vendored `evaluation_suit.evaluation()` directly.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any

from gdp.paths import repo_root

VENDOR_DIR = repo_root() / "third_party" / "drivelm"
EVALUATION_PY = VENDOR_DIR / "evaluation.py"

# Read out of third_party/drivelm/evaluation.py's `__main__` block (lines ~194-223 of the pinned
# commit) and third_party/drivelm/PROVENANCE.md — verbatim, not re-derived. `gdp.vqa.score` writes
# this string into every metrics.json's `final_score_composition` field.
SCORER_COMPOSITION = (
    "final_score = 0.4 * (chatgpt / 100) "
    "+ 0.2 * (Bleu_1/4/3 + Bleu_2/4/3 + Bleu_3/4/3 + Bleu_4/4/3 + ROUGE_L/3 + CIDEr/10/3) "
    "+ 0.2 * (match / 100) "
    "+ 0.2 * accuracy "
    "[weights: chatgpt=0.4, language=0.2, match=0.2, accuracy=0.2] "
    "(third_party/drivelm/evaluation.py __main__ block, "
    "commit 266b570f1c746a4cad7f8f0317eb3c2f99141512)"
)

# Both require the vendored evaluation_suit's paid, non-deterministic OpenAI judge to compute —
# `match` isn't just "the F1 part", `eval_match` fuses F1 with `eval_chatGPT` internally (confirmed
# by running the vendored scorer, third_party/drivelm/PROVENANCE.md's "Known gap" section). This
# repo never calls that API (H2/H9), so both are always omitted, never computed here.
UNSCORABLE_TAGS = {1: "chatgpt", 3: "match"}


class OfficialScorerUnavailable(RuntimeError):
    """The vendored scorer's Python dependencies aren't installed in this environment."""


def scorer_sha256() -> str:
    """SHA-256 of the vendored `evaluation.py`, stamped into every `metrics.json` (decision 1)."""
    return hashlib.sha256(EVALUATION_PY.read_bytes()).hexdigest()


def scorer_composition() -> str:
    """The verbatim final-score weighting read out of the vendored scorer (design decision 6/7)."""
    return SCORER_COMPOSITION


def _load_evaluation_suit() -> type:
    """Import `evaluation_suit` from the vendored module without ever editing it.

    The vendored file does `sys.path.append(".")` then `from gpt_eval import GPTEvaluation`,
    i.e. it expects to run from inside `third_party/drivelm/`. We satisfy that by adding the
    vendor directory to `sys.path` here, in the adapter — not by touching `third_party/`.
    """
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    try:
        from evaluation import evaluation_suit  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OfficialScorerUnavailable(
            "The vendored DriveLM scorer needs `language_evaluation` and `openai` installed. "
            "`language_evaluation` isn't on PyPI — install per its own README:\n"
            '  uv pip install "git+https://github.com/bckim92/language-evaluation.git"\n'
            '  uv run python -c "import language_evaluation; '
            "language_evaluation.download('coco')\"\n"
            "Then: uv sync --extra drivelm-eval"
        ) from exc
    return evaluation_suit


def to_official_format(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape our `predictions.jsonl` rows into `(tag, answer, GT)` triples per item.

    `predictions` rows carry `qa_id`, `tag`, `gt_answer`, `prediction` per the plan's per-line
    schema (design decision 4; `tag` added by spec 08 task 3a once DriveLM's own scorer-routing
    field was found missing from the pipeline). The vendored scorer routes items by this integer
    `tag` list (design decision 6), so every row must carry one — this function raises if any
    row lacks it, rather than silently dropping or guessing a route (H9).
    """
    items = []
    for row in predictions:
        if not row.get("tag"):
            raise KeyError(
                f"prediction row {row.get('qa_id')!r} has no 'tag' field — the vendored scorer "
                "routes items by tag (design decision 6); see gdp.vqa.official.to_official_format."
            )
        items.append(
            {
                "qa_id": row["qa_id"],
                "tag": row["tag"],
                "answer": row["prediction"],
                "GT": row["gt_answer"],
            }
        )
    return items


def run_official_scorer(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Feed reshaped items through the vendored `evaluation_suit` and return only the sub-metrics
    this repo can honestly compute without a live paid API call.

    Mirrors `evaluation.py`'s `__main__` loop's `forward` calls (skipping the graph-based
    question-chaining `eval_graph` gate upstream's `__main__` uses to walk a raw scene/frame/QA
    tree — we score one flat list of already-selected items). Deliberately calls `eval_acc()` (tag
    0) and `eval_language()` (tag 2) directly instead of `evaluation_suit.evaluation()`, because
    `evaluation()` unconditionally calls `eval_chatGPT` (tag 1) — which places a real, paid,
    non-deterministic OpenAI API call, and even crashes with `ZeroDivisionError` when no chatgpt-
    tagged items exist — and `eval_match` (tag 3) is itself fused with that same judge
    internally (see `third_party/drivelm/PROVENANCE.md`'s "Known gap" section). Neither is ever
    computed here; `gdp.vqa.score` reports both as omitted using the `n_*_items` counts below.

    Returns `{"accuracy": float | None, "language": dict | None, "n_accuracy_items": int,
    "n_language_items": int, "n_chatgpt_items": int, "n_match_items": int}` — `None` (not `0.0`)
    when a computable bucket has zero items (design decision 7).
    """
    evaluation_suit = _load_evaluation_suit()
    suit = evaluation_suit()
    for item in items:
        suit.forward(item["tag"], item["answer"], item["GT"])

    accuracy = suit.eval_acc() if suit.accuracy["answer"] else None
    language = suit.eval_language() if suit.language["answer"] else None
    return {
        "accuracy": accuracy,
        "language": language,
        "n_accuracy_items": len(suit.accuracy["answer"]),
        "n_language_items": len(suit.language["answer"]),
        "n_chatgpt_items": len(suit.GPT),
        "n_match_items": len(suit.match["match"]["answer"]),
    }
