"""Lazy-encoding behaviour of `DriveLMVQADataset` with a stubbed encoder — no model, laptop-safe."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gdp.vqa.dataset import AnswerBoundaryUnresolvable, DriveLMVQADataset, VQAExample


def _example(qa_id: str, seq_len: int = 3) -> VQAExample:
    t = torch.zeros(seq_len, dtype=torch.long)
    return VQAExample(qa_id, t, t, t, t, torch.zeros(1, 1), torch.zeros(1, 3, dtype=torch.long))


class _Stub(DriveLMVQADataset):
    """Skips the processor: `bad` maps qa_id -> "seq" | "boundary"; everything else encodes."""

    def __init__(self, ids, bad, lazy):
        self.bad, self.encoded = bad, []
        super().__init__(
            [SimpleNamespace(qa_id=i) for i in ids], None, ".", max_seq_len=4, lazy=lazy
        )

    def _encode(self, record):
        self.encoded.append(record.qa_id)
        kind = self.bad.get(record.qa_id)
        if kind == "boundary":
            raise AnswerBoundaryUnresolvable(record.qa_id)
        return _example(record.qa_id, seq_len=9 if kind == "seq" else 3)


def test_lazy_encodes_nothing_up_front():
    ds = _Stub(["a", "b", "c"], {}, lazy=True)
    assert ds.encoded == [] and len(ds) == 3
    assert ds[1].qa_id == "b" and ds.encoded == ["b"]


def test_lazy_replaces_a_dropped_record_with_the_next_and_counts_it():
    ds = _Stub(["a", "b", "c"], {"b": "seq", "c": "boundary"}, lazy=True)
    assert ds[1].qa_id == "a"  # b (seq too long) -> c (boundary) -> wraps to a
    assert ds.drop_stats == {"seq_too_long": 1, "answer_boundary_unresolvable": 1}


def test_lazy_raises_when_every_record_is_dropped():
    ds = _Stub(["a", "b"], {"a": "seq", "b": "seq"}, lazy=True)
    with pytest.raises(RuntimeError, match="no usable QA pair"):
        ds[0]


def test_eager_still_filters_at_construction():
    ds = _Stub(["a", "b", "c"], {"b": "seq"}, lazy=False)
    assert len(ds) == 2 and ds.drop_stats["seq_too_long"] == 1
