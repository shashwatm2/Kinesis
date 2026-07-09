from __future__ import annotations

from math import isclose

from kinesis.experiments.exp001.metrics import calculate_frame_metrics
from kinesis.experiments.exp001.quality import MetricQualityEvaluator
from kinesis.experiments.exp001.smoothing import MetricSmoother
from kinesis.experiments.exp001.types import Landmark


def test_quality_marks_visible_plausible_metric_as_usable() -> None:
    landmarks = _body_landmarks()
    metrics = calculate_frame_metrics(landmarks, visibility_threshold=0.5)

    quality = MetricQualityEvaluator(
        visibility_threshold=0.5,
        min_average_visibility=0.5,
    ).evaluate(pose_landmarks=landmarks, metrics=metrics)

    assert quality["torso_lean_angle_degrees"].usable is True
    assert quality["torso_lean_angle_degrees"].reason == "ok"
    assert quality["left_knee_angle_degrees"].usable is True


def test_quality_rejects_low_required_landmark_visibility() -> None:
    landmarks = _body_landmarks()
    landmarks[25] = Landmark(x=0.40, y=0.80, visibility=0.1, presence=1.0)
    metrics = calculate_frame_metrics(landmarks, visibility_threshold=0.5)

    quality = MetricQualityEvaluator(
        visibility_threshold=0.5,
        min_average_visibility=0.5,
    ).evaluate(pose_landmarks=landmarks, metrics=metrics)

    assert quality["left_knee_angle_degrees"].usable is False
    assert quality["left_knee_angle_degrees"].reason == "missing_metric"


def test_quality_rejects_large_frame_delta() -> None:
    evaluator = MetricQualityEvaluator(
        visibility_threshold=0.5,
        min_average_visibility=0.5,
    )

    first_landmarks = _body_landmarks(left_elbow_y=0.40, left_wrist_y=0.60)
    first_metrics = calculate_frame_metrics(first_landmarks, visibility_threshold=0.5)
    first_quality = evaluator.evaluate(
        pose_landmarks=first_landmarks,
        metrics=first_metrics,
    )

    second_landmarks = _body_landmarks(left_elbow_y=0.80, left_wrist_y=0.60)
    second_metrics = calculate_frame_metrics(second_landmarks, visibility_threshold=0.5)
    second_quality = evaluator.evaluate(
        pose_landmarks=second_landmarks,
        metrics=second_metrics,
    )

    assert first_quality["left_elbow_angle_degrees"].usable is True
    assert second_quality["left_elbow_angle_degrees"].usable is False
    assert second_quality["left_elbow_angle_degrees"].reason == "large_frame_delta"


def test_smoother_averages_only_usable_values() -> None:
    landmarks = _body_landmarks()
    metrics = calculate_frame_metrics(landmarks, visibility_threshold=0.5)
    quality = MetricQualityEvaluator(
        visibility_threshold=0.5,
        min_average_visibility=0.5,
    ).evaluate(pose_landmarks=landmarks, metrics=metrics)
    smoother = MetricSmoother(window_size=2)

    first = smoother.smooth(metrics=metrics, quality=quality)
    second = smoother.smooth(metrics=metrics, quality=quality)

    assert isclose(first.left_knee_angle_degrees, metrics.left_knee_angle_degrees)
    assert isclose(second.left_knee_angle_degrees, metrics.left_knee_angle_degrees)


def _body_landmarks(
    *,
    left_elbow_y: float = 0.40,
    left_wrist_y: float = 0.60,
) -> list[Landmark]:
    landmarks = [Landmark(x=0.5, y=0.5, visibility=1.0, presence=1.0) for _ in range(33)]
    landmarks[11] = Landmark(x=0.40, y=0.20, visibility=1.0, presence=1.0)
    landmarks[12] = Landmark(x=0.60, y=0.20, visibility=1.0, presence=1.0)
    landmarks[13] = Landmark(x=0.30, y=left_elbow_y, visibility=1.0, presence=1.0)
    landmarks[14] = Landmark(x=0.70, y=0.40, visibility=1.0, presence=1.0)
    landmarks[15] = Landmark(x=0.20, y=left_wrist_y, visibility=1.0, presence=1.0)
    landmarks[16] = Landmark(x=0.80, y=0.60, visibility=1.0, presence=1.0)
    landmarks[23] = Landmark(x=0.40, y=0.60, visibility=1.0, presence=1.0)
    landmarks[24] = Landmark(x=0.60, y=0.60, visibility=1.0, presence=1.0)
    landmarks[25] = Landmark(x=0.40, y=0.80, visibility=1.0, presence=1.0)
    landmarks[26] = Landmark(x=0.60, y=0.80, visibility=1.0, presence=1.0)
    landmarks[27] = Landmark(x=0.40, y=1.00, visibility=1.0, presence=1.0)
    landmarks[28] = Landmark(x=0.60, y=1.00, visibility=1.0, presence=1.0)
    return landmarks
