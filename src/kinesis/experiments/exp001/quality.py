from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from kinesis.experiments.exp001.landmarks import landmark_has_visible_position
from kinesis.experiments.exp001.metrics import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    METRIC_COLUMNS,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    FrameMetrics,
)

QUALITY_COLUMNS: tuple[str, ...] = tuple(
    column
    for metric_name in METRIC_COLUMNS
    for column in (f"{metric_name}_quality", f"{metric_name}_quality_reason")
) + ("quality_usable_metric_count",)


@dataclass(frozen=True)
class MetricQualityRule:
    required_landmarks: tuple[int, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    max_frame_delta: float | None = None


@dataclass(frozen=True)
class MetricQualityResult:
    usable: bool
    reason: str


METRIC_QUALITY_RULES: dict[str, MetricQualityRule] = {
    "shoulder_height_asymmetry": MetricQualityRule(
        required_landmarks=(LEFT_SHOULDER, RIGHT_SHOULDER),
        min_value=0.0,
        max_value=0.50,
        max_frame_delta=0.12,
    ),
    "hip_height_asymmetry": MetricQualityRule(
        required_landmarks=(LEFT_HIP, RIGHT_HIP),
        min_value=0.0,
        max_value=0.50,
        max_frame_delta=0.12,
    ),
    "torso_lean_angle_degrees": MetricQualityRule(
        required_landmarks=(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP),
        min_value=-75.0,
        max_value=75.0,
        max_frame_delta=35.0,
    ),
    "left_knee_angle_degrees": MetricQualityRule(
        required_landmarks=(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        min_value=10.0,
        max_value=180.0,
        max_frame_delta=60.0,
    ),
    "right_knee_angle_degrees": MetricQualityRule(
        required_landmarks=(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        min_value=10.0,
        max_value=180.0,
        max_frame_delta=60.0,
    ),
    "left_elbow_angle_degrees": MetricQualityRule(
        required_landmarks=(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        min_value=10.0,
        max_value=180.0,
        max_frame_delta=70.0,
    ),
    "right_elbow_angle_degrees": MetricQualityRule(
        required_landmarks=(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        min_value=10.0,
        max_value=180.0,
        max_frame_delta=70.0,
    ),
    "center_of_body_x": MetricQualityRule(
        required_landmarks=(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP),
        min_value=0.0,
        max_value=1.0,
        max_frame_delta=0.20,
    ),
    "average_landmark_visibility": MetricQualityRule(min_value=0.0, max_value=1.0),
    "average_landmark_presence": MetricQualityRule(min_value=0.0, max_value=1.0),
}


class MetricQualityEvaluator:
    def __init__(
        self,
        *,
        visibility_threshold: float,
        min_average_visibility: float,
    ) -> None:
        self._visibility_threshold = visibility_threshold
        self._min_average_visibility = min_average_visibility
        self._previous_usable_values: dict[str, float] = {}

    def evaluate(
        self,
        *,
        pose_landmarks: Sequence[Any],
        metrics: FrameMetrics,
    ) -> dict[str, MetricQualityResult]:
        results: dict[str, MetricQualityResult] = {}

        for metric_name in METRIC_COLUMNS:
            result = self._evaluate_metric(
                metric_name=metric_name,
                pose_landmarks=pose_landmarks,
                metrics=metrics,
            )
            results[metric_name] = result

            value = getattr(metrics, metric_name)
            if result.usable and value is not None:
                self._previous_usable_values[metric_name] = value

        return results

    def _evaluate_metric(
        self,
        *,
        metric_name: str,
        pose_landmarks: Sequence[Any],
        metrics: FrameMetrics,
    ) -> MetricQualityResult:
        value = getattr(metrics, metric_name)
        if value is None:
            return MetricQualityResult(usable=False, reason="missing_metric")

        rule = METRIC_QUALITY_RULES[metric_name]
        if not _required_landmarks_visible(
            pose_landmarks,
            required_landmarks=rule.required_landmarks,
            visibility_threshold=self._visibility_threshold,
        ):
            return MetricQualityResult(usable=False, reason="low_required_landmark_visibility")

        average_visibility = metrics.average_landmark_visibility
        if (
            metric_name != "average_landmark_visibility"
            and average_visibility is not None
            and average_visibility < self._min_average_visibility
        ):
            return MetricQualityResult(usable=False, reason="low_average_visibility")

        if rule.min_value is not None and value < rule.min_value:
            return MetricQualityResult(usable=False, reason="outside_plausible_range")
        if rule.max_value is not None and value > rule.max_value:
            return MetricQualityResult(usable=False, reason="outside_plausible_range")

        previous_value = self._previous_usable_values.get(metric_name)
        if (
            previous_value is not None
            and rule.max_frame_delta is not None
            and abs(value - previous_value) > rule.max_frame_delta
        ):
            return MetricQualityResult(usable=False, reason="large_frame_delta")

        return MetricQualityResult(usable=True, reason="ok")


def quality_csv_values(
    quality: dict[str, MetricQualityResult],
) -> dict[str, bool | int | str]:
    values: dict[str, bool | int | str] = {}
    usable_count = 0

    for metric_name in METRIC_COLUMNS:
        result = quality.get(metric_name, MetricQualityResult(False, "missing_quality"))
        values[f"{metric_name}_quality"] = result.usable
        values[f"{metric_name}_quality_reason"] = result.reason
        if result.usable:
            usable_count += 1

    values["quality_usable_metric_count"] = usable_count
    return values


def _required_landmarks_visible(
    pose_landmarks: Sequence[Any],
    *,
    required_landmarks: tuple[int, ...],
    visibility_threshold: float,
) -> bool:
    for index in required_landmarks:
        if index >= len(pose_landmarks):
            return False
        if not landmark_has_visible_position(pose_landmarks[index], visibility_threshold):
            return False
    return True

