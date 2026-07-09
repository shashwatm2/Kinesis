from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


def landmark_value(landmark: Any, key: str) -> float | None:
    if landmark is None:
        return None

    if isinstance(landmark, Mapping):
        value = landmark.get(key)
    else:
        value = getattr(landmark, key, None)

    if value is None:
        return None
    return float(value)


def landmark_has_visible_position(landmark: Any, visibility_threshold: float) -> bool:
    x = landmark_value(landmark, "x")
    y = landmark_value(landmark, "y")
    if x is None or y is None:
        return False
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        return False

    visibility = landmark_value(landmark, "visibility")
    presence = landmark_value(landmark, "presence")

    if visibility is not None and visibility < visibility_threshold:
        return False
    if presence is not None and presence < visibility_threshold:
        return False
    return True
