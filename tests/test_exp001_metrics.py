from __future__ import annotations

from math import isclose

from kinesis.experiments.exp001.csv_export import (
    analysis_csv_fieldnames,
    build_analysis_csv_row,
)
from kinesis.experiments.exp001.metrics import calculate_frame_metrics
from kinesis.experiments.exp001.types import Landmark


def test_calculate_frame_metrics_from_visible_landmarks() -> None:
    landmarks = _empty_landmarks()
    landmarks[11] = Landmark(x=0.40, y=0.20, visibility=1.0, presence=1.0)
    landmarks[12] = Landmark(x=0.60, y=0.30, visibility=1.0, presence=1.0)
    landmarks[13] = Landmark(x=0.30, y=0.40, visibility=1.0, presence=1.0)
    landmarks[14] = Landmark(x=0.70, y=0.50, visibility=1.0, presence=1.0)
    landmarks[15] = Landmark(x=0.20, y=0.60, visibility=1.0, presence=1.0)
    landmarks[16] = Landmark(x=0.80, y=0.70, visibility=1.0, presence=1.0)
    landmarks[23] = Landmark(x=0.40, y=0.60, visibility=1.0, presence=1.0)
    landmarks[24] = Landmark(x=0.60, y=0.60, visibility=1.0, presence=1.0)
    landmarks[25] = Landmark(x=0.40, y=0.80, visibility=1.0, presence=1.0)
    landmarks[26] = Landmark(x=0.60, y=0.80, visibility=1.0, presence=1.0)
    landmarks[27] = Landmark(x=0.40, y=1.00, visibility=1.0, presence=1.0)
    landmarks[28] = Landmark(x=0.60, y=1.00, visibility=1.0, presence=1.0)

    metrics = calculate_frame_metrics(landmarks, visibility_threshold=0.5)

    assert isclose(metrics.shoulder_height_asymmetry, 0.10)
    assert isclose(metrics.hip_height_asymmetry, 0.00)
    assert isclose(metrics.torso_lean_angle_degrees, 0.00)
    assert isclose(metrics.left_knee_angle_degrees, 180.00)
    assert isclose(metrics.right_knee_angle_degrees, 180.00)
    assert isclose(metrics.left_elbow_angle_degrees, 180.00)
    assert isclose(metrics.right_elbow_angle_degrees, 180.00)
    assert isclose(metrics.center_of_body_x, 0.50)
    assert isclose(metrics.average_landmark_visibility, 1.00)


def test_calculate_frame_metrics_returns_none_for_missing_visible_points() -> None:
    metrics = calculate_frame_metrics([], visibility_threshold=0.5)

    assert metrics.shoulder_height_asymmetry is None
    assert metrics.left_knee_angle_degrees is None
    assert metrics.center_of_body_x is None
    assert metrics.average_landmark_visibility is None


def test_build_analysis_csv_row_contains_timestamps_landmarks_and_metrics() -> None:
    landmarks = [Landmark(x=0.1, y=0.2, z=-0.3, visibility=0.9, presence=0.8)]
    metrics = calculate_frame_metrics(landmarks, visibility_threshold=0.5)

    row = build_analysis_csv_row(
        frame_index=12,
        timestamp_ms=400,
        pose_landmarks=landmarks,
        metrics=metrics,
    )

    assert "landmark_00_nose_x" in analysis_csv_fieldnames()
    assert "shoulder_height_asymmetry" in analysis_csv_fieldnames()
    assert row["frame_index"] == 12
    assert row["timestamp_ms"] == 400
    assert row["timestamp_seconds"] == 0.4
    assert row["pose_detected"] is True
    assert row["landmark_00_nose_x"] == 0.1
    assert row["landmark_00_nose_visibility"] == 0.9
    assert row["shoulder_height_asymmetry"] is None


def _empty_landmarks() -> list[Landmark]:
    return [Landmark(x=0.5, y=0.5, visibility=1.0, presence=1.0) for _ in range(33)]
