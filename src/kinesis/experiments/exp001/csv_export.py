from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kinesis.experiments.exp001.landmarks import LANDMARK_NAMES, landmark_value
from kinesis.experiments.exp001.metrics import METRIC_COLUMNS, FrameMetrics

FRAME_COLUMNS: tuple[str, ...] = (
    "frame_index",
    "timestamp_ms",
    "timestamp_seconds",
    "pose_detected",
)
LANDMARK_VALUE_COLUMNS: tuple[str, ...] = ("x", "y", "z", "visibility", "presence")


def analysis_csv_fieldnames() -> list[str]:
    landmark_columns = [
        _landmark_column(index, name, value_name)
        for index, name in enumerate(LANDMARK_NAMES)
        for value_name in LANDMARK_VALUE_COLUMNS
    ]
    return [*FRAME_COLUMNS, *landmark_columns, *METRIC_COLUMNS]


def build_analysis_csv_row(
    *,
    frame_index: int,
    timestamp_ms: int,
    pose_landmarks: Sequence[Any],
    metrics: FrameMetrics,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "timestamp_seconds": timestamp_ms / 1000,
        "pose_detected": bool(pose_landmarks),
    }

    for index, name in enumerate(LANDMARK_NAMES):
        landmark = pose_landmarks[index] if index < len(pose_landmarks) else None
        for value_name in LANDMARK_VALUE_COLUMNS:
            row[_landmark_column(index, name, value_name)] = landmark_value(
                landmark,
                value_name,
            )

    row.update(metrics.as_dict())
    return row


def _landmark_column(index: int, name: str, value_name: str) -> str:
    return f"landmark_{index:02d}_{name}_{value_name}"

