from __future__ import annotations

from pathlib import Path

from kinesis.experiments.exp001.model_assets import ensure_pose_landmarker_model


def test_ensure_pose_landmarker_model_downloads_missing_file(tmp_path: Path) -> None:
    source_model = tmp_path / "source.task"
    source_model.write_bytes(b"model-bytes")
    destination_model = tmp_path / "models" / "pose_landmarker_full.task"

    result = ensure_pose_landmarker_model(
        destination_model,
        model_url=source_model.as_uri(),
    )

    assert result == destination_model
    assert destination_model.read_bytes() == b"model-bytes"


def test_ensure_pose_landmarker_model_reuses_existing_file(tmp_path: Path) -> None:
    destination_model = tmp_path / "pose_landmarker_full.task"
    destination_model.write_bytes(b"existing-model")

    result = ensure_pose_landmarker_model(
        destination_model,
        model_url="file:///missing-source.task",
    )

    assert result == destination_model
    assert destination_model.read_bytes() == b"existing-model"
