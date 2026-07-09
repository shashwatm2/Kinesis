from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kinesis.experiments.exp001.frame_analysis import FrameAnalysis
from kinesis.experiments.exp001.metrics import METRIC_COLUMNS


@dataclass(frozen=True)
class MetricRange:
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class MetricUsage:
    usable_frames: int
    excluded_frames: int
    total_frames: int


@dataclass(frozen=True)
class MovementSummary:
    processed_frames: int
    frames_with_pose: int
    pose_detection_rate: float | None
    average_shoulder_height_asymmetry: float | None
    average_hip_height_asymmetry: float | None
    average_landmark_visibility: float | None
    torso_lean_angle_degrees: MetricRange
    left_knee_angle_degrees: MetricRange
    right_knee_angle_degrees: MetricRange
    left_elbow_angle_degrees: MetricRange
    right_elbow_angle_degrees: MetricRange
    center_of_body_x: MetricRange
    metric_usage: dict[str, MetricUsage]


def summarize_movement(
    frame_analyses: Sequence[FrameAnalysis],
    *,
    processed_frames: int,
    frames_with_pose: int,
) -> MovementSummary:
    pose_detection_rate = None
    if processed_frames > 0:
        pose_detection_rate = frames_with_pose / processed_frames

    return MovementSummary(
        processed_frames=processed_frames,
        frames_with_pose=frames_with_pose,
        pose_detection_rate=pose_detection_rate,
        average_shoulder_height_asymmetry=_average(
            frame_analyses,
            "shoulder_height_asymmetry",
        ),
        average_hip_height_asymmetry=_average(frame_analyses, "hip_height_asymmetry"),
        average_landmark_visibility=_average_raw(
            frame_analyses,
            "average_landmark_visibility",
        ),
        torso_lean_angle_degrees=_range(frame_analyses, "torso_lean_angle_degrees"),
        left_knee_angle_degrees=_range(frame_analyses, "left_knee_angle_degrees"),
        right_knee_angle_degrees=_range(frame_analyses, "right_knee_angle_degrees"),
        left_elbow_angle_degrees=_range(frame_analyses, "left_elbow_angle_degrees"),
        right_elbow_angle_degrees=_range(frame_analyses, "right_elbow_angle_degrees"),
        center_of_body_x=_range(frame_analyses, "center_of_body_x"),
        metric_usage={
            metric_name: _usage(frame_analyses, metric_name) for metric_name in METRIC_COLUMNS
        },
    )


def movement_summary_lines(summary: MovementSummary) -> list[str]:
    lines = [
        (
            "These are descriptive 2D measurements from the uploaded video, "
            "not a score, diagnosis, or medical assessment."
        ),
        (
            f"Pose was detected in {summary.frames_with_pose} of "
            f"{summary.processed_frames} processed frames "
            f"({_format_percent(summary.pose_detection_rate)})."
        ),
    ]

    if summary.average_landmark_visibility is not None:
        lines.append(
            f"Average landmark visibility/confidence was {summary.average_landmark_visibility:.2f}."
        )

    lines.append("Movement values below use quality-filtered, smoothed frame measurements.")

    if (
        summary.average_shoulder_height_asymmetry is not None
        or summary.average_hip_height_asymmetry is not None
    ):
        lines.append(
            "Average height difference was "
            f"{_format_normalized_percent(summary.average_shoulder_height_asymmetry)} "
            "of frame height at the shoulders and "
            f"{_format_normalized_percent(summary.average_hip_height_asymmetry)} "
            "at the hips."
        )

    if _range_available(summary.torso_lean_angle_degrees):
        lines.append(
            "Torso lean ranged from "
            f"{summary.torso_lean_angle_degrees.minimum:.1f} to "
            f"{summary.torso_lean_angle_degrees.maximum:.1f} degrees relative to vertical. "
            "Positive values mean the shoulder center is to image-right of the hip center."
        )

    if _range_available(summary.center_of_body_x):
        lines.append(
            "Center-of-body x-position ranged from "
            f"{_format_normalized_percent(summary.center_of_body_x.minimum)} to "
            f"{_format_normalized_percent(summary.center_of_body_x.maximum)} "
            "of frame width."
        )

    lines.append(f"Quality-filtered frames used: {_usage_summary_text(summary)}.")

    joint_lines = _joint_range_lines(summary)
    if joint_lines:
        lines.extend(joint_lines)
    else:
        lines.append("Not enough visible joint landmarks were available for limb angle ranges.")

    return lines


def _average(frame_analyses: Sequence[FrameAnalysis], field_name: str) -> float | None:
    values = [
        value
        for analysis in frame_analyses
        if (value := _quality_smoothed_value(analysis, field_name)) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _average_raw(frame_analyses: Sequence[FrameAnalysis], field_name: str) -> float | None:
    values = [
        value
        for analysis in frame_analyses
        if (value := getattr(analysis.raw_metrics, field_name)) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _range(frame_analyses: Sequence[FrameAnalysis], field_name: str) -> MetricRange:
    values = [
        value
        for analysis in frame_analyses
        if (value := _quality_smoothed_value(analysis, field_name)) is not None
    ]
    if not values:
        return MetricRange(minimum=None, maximum=None)
    return MetricRange(minimum=min(values), maximum=max(values))


def _usage(frame_analyses: Sequence[FrameAnalysis], field_name: str) -> MetricUsage:
    total_frames = len(frame_analyses)
    usable_frames = sum(
        1
        for analysis in frame_analyses
        if analysis.metric_quality.get(field_name) is not None
        and analysis.metric_quality[field_name].usable
    )
    return MetricUsage(
        usable_frames=usable_frames,
        excluded_frames=total_frames - usable_frames,
        total_frames=total_frames,
    )


def _quality_smoothed_value(analysis: FrameAnalysis, field_name: str) -> float | None:
    quality = analysis.metric_quality.get(field_name)
    if quality is None or not quality.usable:
        return None
    return getattr(analysis.smoothed_metrics, field_name)


def _joint_range_lines(summary: MovementSummary) -> list[str]:
    lines: list[str] = []
    knee_text = _paired_range_text(
        left=summary.left_knee_angle_degrees,
        right=summary.right_knee_angle_degrees,
    )
    if knee_text is not None:
        lines.append(f"Knee angle ranges: {knee_text}.")

    elbow_text = _paired_range_text(
        left=summary.left_elbow_angle_degrees,
        right=summary.right_elbow_angle_degrees,
    )
    if elbow_text is not None:
        lines.append(f"Elbow angle ranges: {elbow_text}.")

    return lines


def _paired_range_text(left: MetricRange, right: MetricRange) -> str | None:
    parts: list[str] = []
    if _range_available(left):
        parts.append(f"left {_format_degree_range(left)}")
    if _range_available(right):
        parts.append(f"right {_format_degree_range(right)}")
    if not parts:
        return None
    return ", ".join(parts)


def _usage_summary_text(summary: MovementSummary) -> str:
    usage_items = (
        ("torso lean", "torso_lean_angle_degrees"),
        ("center x", "center_of_body_x"),
        ("left knee", "left_knee_angle_degrees"),
        ("right knee", "right_knee_angle_degrees"),
        ("left elbow", "left_elbow_angle_degrees"),
        ("right elbow", "right_elbow_angle_degrees"),
    )
    parts = []
    for label, metric_name in usage_items:
        usage = summary.metric_usage.get(metric_name)
        if usage is None:
            continue
        parts.append(f"{label} {usage.usable_frames}/{usage.total_frames}")
    return ", ".join(parts) if parts else "not available"


def _range_available(metric_range: MetricRange) -> bool:
    return metric_range.minimum is not None and metric_range.maximum is not None


def _format_degree_range(metric_range: MetricRange) -> str:
    return f"{metric_range.minimum:.1f} to {metric_range.maximum:.1f} degrees"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value * 100:.1f}%"


def _format_normalized_percent(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value * 100:.1f}%"
