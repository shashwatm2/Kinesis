from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kinesis.experiments.exp001.config import PoseEstimationConfig
from kinesis.experiments.exp002.matching import MatchConfig
from kinesis.experiments.exp002.pipeline import compare_movement_videos


def test_compare_movement_videos_processes_both_videos_then_compares_csvs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[tuple[str, Path, Path]] = []
    reference_csv = tmp_path / "reference.csv"
    practice_csv = tmp_path / "practice.csv"
    match_summary = SimpleNamespace(overall_score=91.0)

    def fake_process_video(*, input_path: Path, output_dir: Path, config: Any) -> Any:
        calls.append(("process", input_path, output_dir))
        csv_path = reference_csv if input_path.name == "reference.mp4" else practice_csv
        return SimpleNamespace(analysis_csv_path=csv_path)

    def fake_compare_movement_csvs(
        *,
        reference_csv_path: Path,
        practice_csv_path: Path,
        output_dir: Path,
        config: Any,
    ) -> Any:
        calls.append(("compare", reference_csv_path, practice_csv_path))
        assert output_dir == tmp_path / "run" / "match"
        assert isinstance(config, MatchConfig)
        return match_summary

    monkeypatch.setattr(
        "kinesis.experiments.exp002.pipeline.process_video",
        fake_process_video,
    )
    monkeypatch.setattr(
        "kinesis.experiments.exp002.pipeline.compare_movement_csvs",
        fake_compare_movement_csvs,
    )

    summary = compare_movement_videos(
        reference_video_path=tmp_path / "reference.mp4",
        practice_video_path=tmp_path / "practice.mp4",
        output_dir=tmp_path / "run",
        pose_config=PoseEstimationConfig(model_path=tmp_path),
        match_config=MatchConfig(),
    )

    assert summary.movement_match is match_summary
    assert calls == [
        ("process", tmp_path / "reference.mp4", tmp_path / "run" / "reference"),
        ("process", tmp_path / "practice.mp4", tmp_path / "run" / "practice"),
        ("compare", reference_csv, practice_csv),
    ]
