"""The class-index -> prompt-token-span mapping is spec 01's real risk (specs/01-data-bdd100k.md
§3): a silently wrong span produces a detector that trains and runs without error but detects
nothing. Every test here targets one of the construction rules that guards against that.

Tests need the `bert-base-uncased` tokenizer, either cached or downloadable. If neither is
available, they skip loudly rather than passing on a code path that never ran.
"""

from __future__ import annotations

import json

import pytest

from gdp.config import BDD100K_CLASSES
from gdp.data.prompt import PromptSpans, get_tokenizer

GOLDEN_PATH = "tests/fixtures/prompt_spans_golden.json"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return get_tokenizer()
    except Exception as exc:  # noqa: BLE001 — any failure to obtain the tokenizer is a skip
        pytest.skip(f"bert-base-uncased tokenizer unavailable (no cache, no network): {exc}")


@pytest.fixture(scope="module")
def spans(tokenizer):
    return PromptSpans.from_classes(BDD100K_CLASSES, tokenizer=tokenizer)


def test_prompt_joins_classes_with_periods(spans):
    expected = (
        "pedestrian. rider. car. truck. bus. train. motorcycle. bicycle. "
        "traffic light. traffic sign."
    )
    assert spans.prompt == expected


@pytest.mark.parametrize("class_idx", range(len(BDD100K_CLASSES)))
def test_span_round_trips(spans, class_idx):
    """Acceptance 3: decoding a class's token span returns its name — the two multi-word
    classes (traffic light / traffic sign) are the ones that actually matter."""
    assert spans.decode_span(class_idx) == BDD100K_CLASSES[class_idx]


def test_char_spans_do_not_confuse_shared_prefixes(spans):
    """'traffic light' and 'traffic sign' share the prefix 'traffic' — a str.find implementation
    would risk matching the wrong occurrence. The constructive cursor must not."""
    light_start, light_end = spans.char_spans[BDD100K_CLASSES.index("traffic light")]
    sign_start, sign_end = spans.char_spans[BDD100K_CLASSES.index("traffic sign")]
    assert spans.prompt[light_start:light_end] == "traffic light"
    assert spans.prompt[sign_start:sign_end] == "traffic sign"
    assert light_end <= sign_start


def test_token_spans_are_disjoint_and_increasing(spans):
    prev_end = -1
    for start, end in spans.token_spans:
        assert end > start
        assert start >= prev_end
        prev_end = end


def test_special_tokens_own_no_class(spans):
    ids = spans.token_class_ids()
    assert ids[0] == -1  # [CLS]
    assert ids[-1] == -1  # [SEP]


def test_positive_map_rows_are_l1_normalized(spans):
    pmap = spans.positive_map(max_text_len=64)
    assert pmap.shape == (len(BDD100K_CLASSES), 64)
    row_sums = pmap.sum(axis=1)
    for s in row_sums:
        assert abs(s - 1.0) < 1e-6


def test_positive_map_rejects_prompt_longer_than_max_text_len(spans):
    with pytest.raises(ValueError, match="max_text_len"):
        spans.positive_map(max_text_len=4)


def test_empty_token_span_raises(tokenizer):
    """A construction that produces a zero-length span must raise, not silently proceed — a
    silent empty span is exactly the 'trains fine, detects nothing' failure mode."""
    broken = PromptSpans(
        prompt="car.",
        classes=("car",),
        input_ids=tuple(tokenizer("car.", add_special_tokens=True)["input_ids"]),
        char_spans=((0, 3),),
        token_spans=((1, 1),),  # empty on purpose
    )
    with pytest.raises(ValueError, match="empty token span"):
        broken._assert_invariants()


def test_overlapping_token_spans_raise(tokenizer):
    broken = PromptSpans(
        prompt="car. bus.",
        classes=("car", "bus"),
        input_ids=tuple(tokenizer("car. bus.", add_special_tokens=True)["input_ids"]),
        char_spans=((0, 3), (5, 8)),
        token_spans=((1, 3), (2, 4)),  # deliberately overlapping
    )
    with pytest.raises(ValueError, match="overlaps"):
        broken._assert_invariants()


def test_matches_committed_golden_file(spans):
    """A tokenizer-version bump that silently shifts a span would otherwise change every
    training label in spec 03 without a single test going red. This is the guard."""
    golden = json.loads(open(GOLDEN_PATH).read())
    assert spans.to_json() == golden
