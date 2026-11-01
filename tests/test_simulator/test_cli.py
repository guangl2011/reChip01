"""Tests for the simulator demo CLI (python -m rechip01.simulator)."""

import json
from pathlib import Path

from rechip01.schemas import Provenance, ScanResult
from rechip01.simulator.__main__ import main, parse_args


def test_parse_args_defaults() -> None:
    args = parse_args(["--out", "/tmp/demo-fleet"])
    assert args.seed == 0
    assert args.n_units == 100
    assert args.horizon_days == 365


def test_main_writes_a_valid_demo_batch(tmp_path: Path) -> None:
    exit_code = main(["--seed=1234", "--n-units=25", "--horizon-days=180", f"--out={tmp_path}"])
    assert exit_code == 0

    scans_path = tmp_path / "scans.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    config_path = tmp_path / "config.json"
    assert scans_path.exists() and labels_path.exists() and config_path.exists()

    scans = [
        ScanResult.model_validate(json.loads(line)) for line in scans_path.read_text().splitlines()
    ]
    assert len(scans) == 25
    assert all(scan.provenance == Provenance.SIMULATED for scan in scans)

    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    assert [label["asset_id"] for label in labels] == [scan.asset_id for scan in scans]

    config = json.loads(config_path.read_text())
    assert config["seed"] == 1234
    assert config["n_units"] == 25
