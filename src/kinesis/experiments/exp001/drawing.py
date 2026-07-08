from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from kinesis.experiments.exp001.landmarks import landmark_has_visible_position, landmark_value

Color = tuple[int, int, int]
Point = tuple[int, int]

# MediaPipe Pose Landmarker returns 33 landmarks. These are the standard body
# connections used for a readable BlazePose-style skeleton overlay.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


def draw_pose_landmarks(
    frame: np.ndarray,
    pose_landmarks: Sequence[Any],
    *,
    visibility_threshold: float = 0.5,
    connection_color: Color = (49, 204, 132),
    joint_color: Color = (255, 255, 255),
    connection_thickness: int = 2,
    joint_radius: int = 4,
) -> np.ndarray:
    """Return a copy of frame with one pose skeleton drawn on top.

    The input frame is expected to be a height x width x 3 uint8 image. Colors
    are BGR because OpenCV reads and writes video in BGR order.
    """

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3).")

    annotated = frame.copy()
    if not pose_landmarks:
        return annotated

    height, width = annotated.shape[:2]
    visible_points: dict[int, Point] = {}

    for index, landmark in enumerate(pose_landmarks):
        if landmark_has_visible_position(landmark, visibility_threshold):
            visible_points[index] = _to_pixel_point(landmark, width=width, height=height)

    for start_index, end_index in POSE_CONNECTIONS:
        start = visible_points.get(start_index)
        end = visible_points.get(end_index)
        if start is not None and end is not None:
            _draw_line(
                annotated,
                start=start,
                end=end,
                color=connection_color,
                thickness=connection_thickness,
            )

    for point in visible_points.values():
        _draw_disk(annotated, center=point, radius=joint_radius, color=joint_color)

    return annotated


def extract_keyframe_indices(total_frames: int, max_keyframes: int) -> list[int]:
    if total_frames <= 0 or max_keyframes <= 0:
        return []
    if total_frames <= max_keyframes:
        return list(range(total_frames))
    if max_keyframes == 1:
        return [total_frames // 2]

    indices = {
        round(position * (total_frames - 1) / (max_keyframes - 1))
        for position in range(max_keyframes)
    }
    return sorted(indices)


def _to_pixel_point(landmark: Any, *, width: int, height: int) -> Point:
    x = landmark_value(landmark, "x")
    y = landmark_value(landmark, "y")
    if x is None or y is None:
        raise ValueError("landmark must include x and y values.")

    pixel_x = min(width - 1, max(0, round(x * (width - 1))))
    pixel_y = min(height - 1, max(0, round(y * (height - 1))))
    return pixel_x, pixel_y


def _draw_line(
    image: np.ndarray,
    *,
    start: Point,
    end: Point,
    color: Color,
    thickness: int,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    radius = max(1, thickness)

    xs = np.linspace(x0, x1, steps + 1)
    ys = np.linspace(y0, y1, steps + 1)
    for x, y in zip(xs, ys, strict=True):
        _draw_disk(image, center=(round(float(x)), round(float(y))), radius=radius, color=color)


def _draw_disk(image: np.ndarray, *, center: Point, radius: int, color: Color) -> None:
    height, width = image.shape[:2]
    center_x, center_y = center
    radius = max(1, radius)

    min_x = max(0, center_x - radius)
    max_x = min(width, center_x + radius + 1)
    min_y = max(0, center_y - radius)
    max_y = min(height, center_y + radius + 1)

    if min_x >= max_x or min_y >= max_y:
        return

    y_grid, x_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x_grid * x_grid + y_grid * y_grid <= radius * radius

    mask_min_y = min_y - (center_y - radius)
    mask_max_y = mask_min_y + (max_y - min_y)
    mask_min_x = min_x - (center_x - radius)
    mask_max_x = mask_min_x + (max_x - min_x)

    region = image[min_y:max_y, min_x:max_x]
    region[mask[mask_min_y:mask_max_y, mask_min_x:mask_max_x]] = color
