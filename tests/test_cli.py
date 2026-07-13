"""The CLI is an honest roadmap: implemented commands work, unimplemented ones name their spec."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from gdp.cli import app

runner = CliRunner()

# command → the spec that will implement it
PENDING = {
    "data": "01-data-bdd100k",
    "detect": "02-zeroshot-baseline",
    "evaluate": "02-zeroshot-baseline",
    "deploy": "05-deploy-onnx-edge",
    "vqa": "07-finetune-vlm",
    "demo": "09-integrated-demo",
}


def test_help_lists_the_whole_command_surface():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["info", *PENDING]:
        assert command in result.stdout


def test_info_is_implemented_and_reports_the_environment():
    result = runner.invoke(app, ["info", "-c", "configs/default.yaml"])
    assert result.exit_code == 0, result.stdout

    payload = json.loads(result.stdout)
    assert payload["seed"] == 42
    assert payload["num_classes"] == 10
    assert payload["device"] in ("cuda", "mps", "cpu")
    assert payload["detector"] == "IDEA-Research/grounding-dino-tiny"


@pytest.mark.parametrize(("command", "spec"), sorted(PENDING.items()))
def test_unimplemented_commands_fail_loudly_and_name_their_spec(command, spec):
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    output = result.stdout + str(result.stderr or "")
    assert "Not implemented" in output
    assert spec in output


def test_bad_config_path_is_an_error():
    result = runner.invoke(app, ["info", "-c", "configs/nope.yaml"])
    assert result.exit_code != 0
