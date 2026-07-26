"""Attribution invariants + the H5 command guard (spec 09 task 2). No model is loaded by any test
in this file, so it is intentionally NOT `model_heavy` (CLAUDE.md §4)."""

from __future__ import annotations

import pytest

from gdp.demo.honesty import REPORT_CAVEAT
from gdp.demo.report import (
    Attribution,
    ReportLine,
    ReportSection,
    compose_report,
    render_report,
    summarize_detections,
)
from gdp.detect.predictions import Detection

CLASSES = ("pedestrian", "car", "bus")


def _det(class_id: int, score: float = 0.9) -> Detection:
    return Detection(image_id=0, class_id=class_id, score=score, xyxy=(0.0, 0.0, 10.0, 10.0))


def test_report_line_requires_source_explicitly():
    with pytest.raises(TypeError):
        ReportLine("3 cars detected.")  # type: ignore[call-arg]


def test_report_line_rejects_empty_text():
    with pytest.raises(ValueError):
        ReportLine("   ", Attribution.DETECTOR)


def test_report_line_rejects_smuggled_tag():
    with pytest.raises(ValueError):
        ReportLine("3 cars detected. [VLM — Stage 2]", Attribution.DETECTOR)


def test_report_line_rejects_imperative_command_as_first_token():
    with pytest.raises(ValueError):
        ReportLine("Brake now.", Attribution.VLM)


@pytest.mark.parametrize(
    "command", ["merge", "stop", "accelerate", "steer", "turn", "yield", "overtake"]
)
def test_report_line_rejects_every_imperative_command(command):
    with pytest.raises(ValueError):
        ReportLine(f"{command.capitalize()} immediately.", Attribution.VLM)


def test_report_line_accepts_descriptive_text():
    line = ReportLine("A cyclist is ahead on the right.", Attribution.VLM)
    assert line.text == "A cyclist is ahead on the right."


def test_render_report_tags_every_line():
    section = ReportSection(
        "Scene summary",
        (
            ReportLine("2 cars detected.", Attribution.DETECTOR),
            ReportLine("A pedestrian is crossing.", Attribution.VLM),
        ),
    )
    rendered = render_report([section])
    assert "[detector — Stage 1]" in rendered
    assert "[VLM — Stage 2]" in rendered
    assert "## Scene summary" in rendered


def test_summarize_detections_counts_by_class():
    dets = [_det(1), _det(1), _det(0)]
    line = summarize_detections(dets, CLASSES)
    assert line.source is Attribution.DETECTOR
    assert "2 cars" in line.text
    assert "1 pedestrian" in line.text
    assert line.text.endswith("detected.")


def test_summarize_detections_empty_is_still_attributed():
    line = summarize_detections([], CLASSES)
    assert line.source is Attribution.DETECTOR
    assert "Nothing detected" in line.text


def test_compose_report_ends_with_caveat_and_tags_every_line():
    det_lines = [ReportLine("2 cars detected.", Attribution.DETECTOR)]
    vlm_lines = [ReportLine("A cyclist is ahead.", Attribution.VLM)]
    heuristic_lines = [ReportLine("1 car is ahead of the ego vehicle.", Attribution.HEURISTIC)]
    report = compose_report(det_lines, vlm_lines, heuristic_lines)
    assert report.strip().endswith(REPORT_CAVEAT)
    for line in report.splitlines():
        if line.startswith("##") or not line.strip():
            continue
        if line.strip() == REPORT_CAVEAT:
            continue
        assert line.rstrip().endswith("]")
