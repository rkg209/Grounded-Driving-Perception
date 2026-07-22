from __future__ import annotations

from pathlib import Path

from gdp.eval.operating_point import GtBox, PredBox
from gdp.eval.qualitative import find_miss_to_hit_pairs, save_pair_crops


def test_finds_a_miss_that_becomes_a_hit():
    gt = [GtBox(image_id=0, class_id=2, xyxy=(10, 10, 50, 50))]
    zeroshot_preds: list[PredBox] = []  # zero-shot found nothing
    finetuned_preds = [PredBox(image_id=0, class_id=2, score=0.9, xyxy=(11, 11, 49, 49))]

    pairs = find_miss_to_hit_pairs(gt, zeroshot_preds, finetuned_preds)

    assert len(pairs) == 1
    assert pairs[0].image_id == 0
    assert pairs[0].gt_box == gt[0]
    assert pairs[0].finetuned_box == finetuned_preds[0]


def test_a_zeroshot_hit_is_never_a_pair():
    gt = [GtBox(image_id=0, class_id=2, xyxy=(10, 10, 50, 50))]
    zeroshot_preds = [PredBox(image_id=0, class_id=2, score=0.8, xyxy=(11, 11, 49, 49))]
    finetuned_preds = [PredBox(image_id=0, class_id=2, score=0.9, xyxy=(11, 11, 49, 49))]

    pairs = find_miss_to_hit_pairs(gt, zeroshot_preds, finetuned_preds)

    assert pairs == []


def test_a_finetuned_miss_is_never_a_pair():
    """Zero-shot missed it, but so did fine-tuned — nothing "got better" here."""
    gt = [GtBox(image_id=0, class_id=2, xyxy=(10, 10, 50, 50))]
    pairs = find_miss_to_hit_pairs(gt, [], [])
    assert pairs == []


def test_wrong_class_prediction_does_not_count_as_a_hit():
    gt = [GtBox(image_id=0, class_id=2, xyxy=(10, 10, 50, 50))]
    finetuned_preds = [PredBox(image_id=0, class_id=5, score=0.9, xyxy=(11, 11, 49, 49))]
    pairs = find_miss_to_hit_pairs(gt, [], finetuned_preds)
    assert pairs == []


def test_limit_caps_the_number_of_pairs():
    gt = [GtBox(image_id=i, class_id=2, xyxy=(10, 10, 50, 50)) for i in range(5)]
    finetuned_preds = [
        PredBox(image_id=i, class_id=2, score=0.9, xyxy=(11, 11, 49, 49)) for i in range(5)
    ]
    pairs = find_miss_to_hit_pairs(gt, [], finetuned_preds, limit=2)
    assert len(pairs) == 2


def test_save_pair_crops_writes_one_file_per_pair(tmp_path):
    image_path = Path("tests/fixtures/mini_bdd/scene_000.jpg")
    gt = [GtBox(image_id=0, class_id=0, xyxy=(12, 203, 108, 240))]
    finetuned_preds = [PredBox(image_id=0, class_id=0, score=0.9, xyxy=(12, 203, 108, 240))]
    pairs = find_miss_to_hit_pairs(gt, [], finetuned_preds)
    assert len(pairs) == 1

    out_dir = tmp_path / "pairs"
    saved = save_pair_crops(
        pairs,
        {0: image_path},
        ["pedestrian", "rider", "car"],
        out_dir=out_dir,
    )

    assert len(saved) == 1
    assert (tmp_path / saved[0]).is_file()
    assert "pedestrian" in saved[0]
