from __future__ import annotations

import numpy as np

from kinesis.experiments.exp001.drawing import draw_pose_landmarks, extract_keyframe_indices
from kinesis.experiments.exp001.types import Landmark


def test_draw_pose_landmarks_returns_annotated_copy() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = _empty_landmarks()
    landmarks[11] = Landmark(x=0.25, y=0.50, visibility=1.0, presence=1.0)
    landmarks[12] = Landmark(x=0.75, y=0.50, visibility=1.0, presence=1.0)

    annotated = draw_pose_landmarks(frame, landmarks)

    assert annotated is not frame
    assert frame.sum() == 0
    assert annotated.sum() > 0


def test_draw_pose_landmarks_skips_low_visibility_points() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = _empty_landmarks()
    landmarks[11] = Landmark(x=0.25, y=0.50, visibility=0.1, presence=1.0)
    landmarks[12] = Landmark(x=0.75, y=0.50, visibility=0.1, presence=1.0)

    annotated = draw_pose_landmarks(frame, landmarks, visibility_threshold=0.5)

    assert annotated.sum() == 0


def test_extract_keyframe_indices_spreads_frames_across_video() -> None:
    assert extract_keyframe_indices(total_frames=10, max_keyframes=4) == [0, 3, 6, 9]


def test_extract_keyframe_indices_supports_more_than_default_keyframes() -> None:
    assert extract_keyframe_indices(total_frames=20, max_keyframes=10) == [
        0,
        2,
        4,
        6,
        8,
        11,
        13,
        15,
        17,
        19,
    ]


def test_extract_keyframe_indices_can_be_disabled() -> None:
    assert extract_keyframe_indices(total_frames=10, max_keyframes=0) == []


def test_extract_keyframe_indices_handles_empty_video() -> None:
    assert extract_keyframe_indices(total_frames=0, max_keyframes=4) == []


def _empty_landmarks() -> list[Landmark]:
    return [Landmark(x=-1.0, y=-1.0, visibility=0.0, presence=0.0) for _ in range(33)]
