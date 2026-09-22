"""`weighted_loss`: HF's Grounding-DINO total, with the encoder class term reweighted ([SEQ-0132]).

No model is loaded here; the model_heavy equivalence against HF's own `outputs.loss` lives in
tests/test_train_loop.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gdp.train.trainer import HF_CLASS_LOSS_WEIGHT, weighted_loss

# Grounding-DINO-tiny's values: two-stage on, auxiliary decoder losses off.
TINY = SimpleNamespace(
    bbox_loss_coefficient=5.0,
    giou_loss_coefficient=2.0,
    two_stage=True,
    auxiliary_loss=False,
    decoder_layers=6,
)

# The magnitudes the gate's forward pass produced on mini_bdd.
LOSS_DICT = {
    "loss_ce": torch.tensor(0.8),
    "loss_bbox": torch.tensor(0.01),
    "loss_giou": torch.tensor(0.07),
    "loss_ce_enc": torch.tensor(65000.0),
    "loss_bbox_enc": torch.tensor(0.2),
    "loss_giou_enc": torch.tensor(0.5),
    "cardinality_error": torch.tensor(2.5),
    "cardinality_error_enc": torch.tensor(891.0),
}


def test_zero_weight_drops_only_the_encoder_class_term():
    loss = weighted_loss(LOSS_DICT, TINY, enc_class_loss_weight=0.0)
    expected = 2.0 * 0.8 + 5.0 * 0.01 + 2.0 * 0.07 + 5.0 * 0.2 + 2.0 * 0.5
    assert float(loss) == pytest.approx(expected)


def test_hf_weight_reproduces_hfs_formula():
    """Cardinality errors are logged by HF but never weighted into its total."""
    loss = weighted_loss(LOSS_DICT, TINY, enc_class_loss_weight=HF_CLASS_LOSS_WEIGHT)
    expected = 2.0 * 0.8 + 5.0 * 0.01 + 2.0 * 0.07 + 2.0 * 65000.0 + 5.0 * 0.2 + 2.0 * 0.5
    assert float(loss) == pytest.approx(expected)


def test_auxiliary_decoder_terms_are_weighted_when_enabled():
    config = SimpleNamespace(**{**vars(TINY), "two_stage": False, "auxiliary_loss": True})
    loss_dict = {"loss_ce": torch.tensor(1.0), "loss_ce_0": torch.tensor(1.0)}
    assert float(weighted_loss(loss_dict, config, 0.0)) == pytest.approx(4.0)
