from __future__ import annotations

import csv
from pathlib import Path

from kinesis.experiments.exp002.matching import MatchConfig, compare_movement_csvs

METRIC_COLUMNS = (
    "smoothed_torso_lean_angle_degrees",
    "smoothed_left_knee_angle_degrees",
    "smoothed_right_knee_angle_degrees",
    "smoothed_left_elbow_angle_degrees",
    "smoothed_right_elbow_angle_degrees",
    "smoothed_center_of_body_x",
)


def test_compare_movement_csvs_scores_identical_series_as_full_match(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    practice_csv = tmp_path / "practice.csv"
    rows = [
        _row(0.0, torso=0, left_knee=170, right_knee=170, left_elbow=160, right_elbow=160),
        _row(0.5, torso=10, left_knee=120, right_knee=125, left_elbow=110, right_elbow=115),
        _row(1.0, torso=0, left_knee=175, right_knee=175, left_elbow=165, right_elbow=165),
    ]
    _write_csv(reference_csv, rows)
    _write_csv(practice_csv, rows)

    summary = compare_movement_csvs(
        reference_csv_path=reference_csv,
        practice_csv_path=practice_csv,
        output_dir=tmp_path / "match",
        config=MatchConfig(max_offset_seconds=0.5, min_compared_frames=3),
    )

    assert summary.overall_score == 100.0
    assert summary.best_time_offset_seconds == 0.0
    assert summary.compared_frame_count == 3
    assert summary.report_csv_path.exists()
    assert summary.summary_json_path.exists()


def test_compare_movement_csvs_finds_delayed_practice_timeline(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    practice_csv = tmp_path / "practice.csv"
    reference_rows = [
        _row(0.0, torso=0, left_knee=170, right_knee=170, left_elbow=160, right_elbow=160),
        _row(0.5, torso=12, left_knee=120, right_knee=125, left_elbow=110, right_elbow=115),
        _row(1.0, torso=-5, left_knee=175, right_knee=175, left_elbow=165, right_elbow=165),
    ]
    practice_rows = [
        {**row, "timestamp_seconds": f"{float(row['timestamp_seconds']) + 0.5:.1f}"}
        for row in reference_rows
    ]
    _write_csv(reference_csv, reference_rows)
    _write_csv(practice_csv, practice_rows)

    summary = compare_movement_csvs(
        reference_csv_path=reference_csv,
        practice_csv_path=practice_csv,
        output_dir=tmp_path / "match",
        config=MatchConfig(
            max_offset_seconds=0.5,
            offset_step_seconds=0.5,
            timestamp_tolerance_seconds=0.05,
            min_compared_frames=3,
        ),
    )

    assert summary.overall_score == 100.0
    assert summary.best_time_offset_seconds == 0.5


def test_compare_movement_csvs_reports_largest_difference_moments(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    practice_csv = tmp_path / "practice.csv"
    reference_rows = [
        _row(0.0, torso=0, left_knee=170, right_knee=170, left_elbow=160, right_elbow=160),
        _row(0.5, torso=0, left_knee=170, right_knee=170, left_elbow=160, right_elbow=160),
        _row(1.0, torso=0, left_knee=170, right_knee=170, left_elbow=160, right_elbow=160),
    ]
    practice_rows = [
        _row(0.0, torso=0, left_knee=170, right_knee=170, left_elbow=160, right_elbow=160),
        _row(0.5, torso=0, left_knee=90, right_knee=170, left_elbow=160, right_elbow=160),
        _row(1.0, torso=0, left_knee=170, right_knee=170, left_elbow=160, right_elbow=160),
    ]
    _write_csv(reference_csv, reference_rows)
    _write_csv(practice_csv, practice_rows)

    summary = compare_movement_csvs(
        reference_csv_path=reference_csv,
        practice_csv_path=practice_csv,
        output_dir=tmp_path / "match",
        config=MatchConfig(max_offset_seconds=0.0, min_compared_frames=3),
    )

    assert summary.overall_score < 100.0
    assert summary.largest_difference_moments[0].reference_timestamp_seconds == 0.5
    assert summary.largest_difference_moments[0].largest_difference_metric == (
        "left_knee_angle_degrees"
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["timestamp_seconds", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def _row(
    timestamp_seconds: float,
    *,
    torso: float,
    left_knee: float,
    right_knee: float,
    left_elbow: float,
    right_elbow: float,
) -> dict[str, str]:
    return {
        "timestamp_seconds": str(timestamp_seconds),
        "smoothed_torso_lean_angle_degrees": str(torso),
        "smoothed_left_knee_angle_degrees": str(left_knee),
        "smoothed_right_knee_angle_degrees": str(right_knee),
        "smoothed_left_elbow_angle_degrees": str(left_elbow),
        "smoothed_right_elbow_angle_degrees": str(right_elbow),
        "smoothed_center_of_body_x": "0.5",
    }
