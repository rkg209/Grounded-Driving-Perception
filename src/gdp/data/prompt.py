"""Class-index -> prompt-token-span mapping for Grounding-DINO's contrastive text head.

Grounding-DINO has no class head: it scores each predicted box against the *text tokens* of a
prompt like "pedestrian. rider. car. ...". A class is therefore a token span, not a label. Getting
this mapping wrong produces a model that trains and runs without error, but detects nothing (or a
plausible-but-wrong mAP) — so every construction rule below guards a specific, real failure mode
(see specs/01-data-bdd100k.md §3 for the rationale each rule encodes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

TOKENIZER_NAME = "bert-base-uncased"

_tokenizer_cache: dict[str, PreTrainedTokenizerBase] = {}


def get_tokenizer(name: str = TOKENIZER_NAME) -> PreTrainedTokenizerBase:
    """Load (and cache) the tokenizer Grounding-DINO's text backbone uses."""
    if name not in _tokenizer_cache:
        _tokenizer_cache[name] = AutoTokenizer.from_pretrained(name)
    return _tokenizer_cache[name]


@dataclass(frozen=True)
class PromptSpans:
    """The prompt, its tokenization, and the char/token span owned by each class.

    Construction is the whole point: `from_classes` builds every span with a cursor rather than
    `str.find`, gets token spans from the tokenizer's own offset mapping rather than by counting
    words, and asserts every invariant a bad implementation could silently violate.
    """

    prompt: str
    classes: tuple[str, ...]
    input_ids: tuple[int, ...]
    char_spans: tuple[tuple[int, int], ...]
    token_spans: tuple[tuple[int, int], ...]

    @classmethod
    def from_classes(
        cls, classes: tuple[str, ...], tokenizer: PreTrainedTokenizerBase | None = None
    ) -> PromptSpans:
        tokenizer = tokenizer or get_tokenizer()

        # Rule 1: build char offsets constructively with a cursor while joining the prompt.
        # Never str.find(class_name) — "traffic light" and "traffic sign" share the prefix
        # "traffic", and a substring search is a coin flip waiting to happen.
        prompt_parts: list[str] = []
        char_spans: list[tuple[int, int]] = []
        cursor = 0
        for name in classes:
            start = cursor
            end = start + len(name)
            char_spans.append((start, end))
            prompt_parts.append(f"{name}.")
            cursor = end + 2  # "<name>." + " " before the next class

        prompt = " ".join(prompt_parts)

        # Rule 2: token spans come from the tokenizer's own offset mapping, never from counting
        # words — "traffic light" is two words and several word-pieces.
        encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        input_ids = tuple(encoding["input_ids"])
        offsets = encoding["offset_mapping"]

        token_spans: list[tuple[int, int]] = []
        for char_start, char_end in char_spans:
            owned = [
                tok_idx
                for tok_idx, (o_start, o_end) in enumerate(offsets)
                if (o_start, o_end) != (0, 0) and o_start < char_end and o_end > char_start
            ]
            if not owned:
                raise ValueError(
                    f"no token overlaps char span ({char_start}, {char_end}) in prompt {prompt!r}"
                )
            token_spans.append((owned[0], owned[-1] + 1))

        spans = cls(
            prompt=prompt,
            classes=tuple(classes),
            input_ids=input_ids,
            char_spans=tuple(char_spans),
            token_spans=tuple(token_spans),
        )
        spans._assert_invariants()
        return spans

    def _assert_invariants(self) -> None:
        # Rule 3: every class owns >= 1 token; spans are disjoint and strictly increasing. A
        # silent empty span is exactly the "trains fine, detects nothing" failure.
        prev_end = -1
        for i, (start, end) in enumerate(self.token_spans):
            if end <= start:
                raise ValueError(
                    f"class {self.classes[i]!r} has an empty token span ({start}, {end})"
                )
            if start < prev_end:
                raise ValueError(
                    f"class {self.classes[i]!r} token span {(start, end)} overlaps the "
                    "previous span"
                )
            prev_end = end

    def token_class_ids(self) -> np.ndarray:
        """len(input_ids); -1 where the token belongs to no class."""
        ids = np.full(len(self.input_ids), -1, dtype=np.int64)
        for class_idx, (start, end) in enumerate(self.token_spans):
            ids[start:end] = class_idx
        return ids

    def positive_map(self, max_text_len: int = 256) -> np.ndarray:
        """(num_classes, max_text_len), L1-normalized rows — the MDETR/Grounding-DINO convention.

        max_text_len must match what the processor pads the prompt to.
        """
        if len(self.input_ids) > max_text_len:
            raise ValueError(
                f"prompt has {len(self.input_ids)} tokens, exceeds max_text_len={max_text_len}"
            )
        pmap = np.zeros((len(self.classes), max_text_len), dtype=np.float32)
        for class_idx, (start, end) in enumerate(self.token_spans):
            pmap[class_idx, start:end] = 1.0
        row_sums = pmap.sum(axis=1, keepdims=True)
        return pmap / row_sums

    def decode_span(self, class_idx: int, tokenizer: PreTrainedTokenizerBase | None = None) -> str:
        """Decode the tokens owned by a class back to text — the round-trip check."""
        tokenizer = tokenizer or get_tokenizer()
        start, end = self.token_spans[class_idx]
        return tokenizer.decode(self.input_ids[start:end]).strip()

    def to_json(self) -> dict:
        return {
            "prompt": self.prompt,
            "classes": list(self.classes),
            "input_ids": list(self.input_ids),
            "char_spans": [list(s) for s in self.char_spans],
            "token_spans": [list(s) for s in self.token_spans],
        }
