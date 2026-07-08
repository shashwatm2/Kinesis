from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import acos, atan2, degrees, sqrt
from typing import Any

from kinesis.experiments.exp001.landmarks import landmark_has_visible_position, landmark_value

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

CORE_BODY_INDICES = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
METRIC_COLUMNS: tuple[str, ...] = (
    "shoulder_height_asymmetry",
    "hip_height_asymmetry",
    "torso_lean_angle_degrees",
    "left_knee_angle_degrees",
    "right_knee_angle_degrees",
    "left_elbow_angle_degrees",
    "right_elbow_angle_degrees",
    "center_of_body_x",
    "average_landmark_visibility",
    "average_landmark_presence",
)


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class FrameMetrics:
    shoulder_height_asymmetry: float | None = None
    hip_height_asymmetry: float | None = None
    torso_lean_angle_degrees: float | None = None
    left_knee_angle_degrees: float | None = None
    right_knee_angle_degrees: float | None = None
    left_elbow_angle_degrees: float | None = None
    right_elbow_angle_degrees: float | None = None
    center_of_body_x: float | None = None
    average_landmark_visibility: float | None = None
    average_landmark_presence: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def calculate_frame_metrics(
    pose_landmarks: Sequence[Any],
    *,
    visibility_threshold: float,
) -> FrameMetrics:
    """Calculate descriptive 2D movement metrics for one detected pose.

    All distances are normalized image-space values. Angles are calculated from
    normalized x/y coordinates, not 3D world coordinates.
    """

    left_shoulder = _visible_point(pose_landmarks, LEFT_SHOULDER, visibility_threshold)
    right_shoulder = _visible_point(pose_landmarks, RIGHT_SHOULDER, visibility_threshold)
    left_hip = _visible_point(pose_landmarks, LEFT_HIP, visibility_threshold)
    right_hip = _visible_point(pose_landmarks, RIGHT_HIP, visibility_threshold)

    shoulder_center = _midpoint(left_shoulder, right_shoulder)
    hip_center = _midpoint(left_hip, right_hip)

    return FrameMetrics(
        shoulder_height_asymmetry=_height_asymmetry(left_shoulder, right_shoulder),
        hip_height_asymmetry=_height_asymmetry(left_hip, right_hip),
        torso_lean_angle_degrees=_torso_lean_angle(shoulder_center, hip_center),
        left_knee_angle_degrees=_joint_angle(
            _visible_point(pose_landmarks, LEFT_HIP, visibility_threshold),
            _visible_point(pose_landmarks, LEFT_KNEE, visibility_threshold),
            _visible_point(pose_landmarks, LEFT_ANKLE, visibility_threshold),
        ),
        right_knee_angle_degrees=_joint_angle(
            _visible_point(pose_landmarks, RIGHT_HIP, visibility_threshold),
            _visible_point(pose_landmarks, RIGHT_KNEE, visibility_threshold),
            _visible_point(pose_landmarks, RIGHT_ANKLE, visibility_threshold),
        ),
        left_elbow_angle_degrees=_joint_angle(
            _visible_point(pose_landmarks, LEFT_SHOULDER, visibility_threshold),
            _visible_point(pose_landmarks, LEFT_ELBOW, visibility_threshold),
            _visible_point(pose_landmarks, LEFT_WRIST, visibility_threshold),
        ),
        right_elbow_angle_degrees=_joint_angle(
            _visible_point(pose_landmarks, RIGHT_SHOULDER, visibility_threshold),
            _visible_point(pose_landmarks, RIGHT_ELBOW, visibility_threshold),
            _visible_point(pose_landmarks, RIGHT_WRIST, visibility_threshold),
        ),
        center_of_body_x=_center_of_body_x(pose_landmarks, visibility_threshold),
        average_landmark_visibility=_average_landmark_score(pose_landmarks, "visibility"),
        average_landmark_presence=_average_landmark_score(pose_landmarks, "presence"),
    )


def _visible_point(
    pose_landmarks: Sequence[Any],
    index: int,
    visibility_threshold: float,
) -> Point2D | None:
    if index >= len(pose_landmarks):
        return None

    landmark = pose_landmarks[index]
    if not landmark_has_visible_position(landmark, visibility_threshold):
        return None

    x = landmark_value(landmark, "x")
    y = landmark_value(landmark, "y")
    if x is None or y is None:
        return None
    return Point2D(x=x, y=y)


def _height_asymmetry(left: Point2D | None, right: Point2D | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(left.y - right.y)


def _midpoint(a: Point2D | None, b: Point2D | None) -> Point2D | None:
    if a is None or b is None:
        return None
    return Point2D(x=(a.x + b.x) / 2, y=(a.y + b.y) / 2)


def _torso_lean_angle(
    shoulder_center: Point2D | None,
    hip_center: Point2D | None,
) -> float | None:
    if shoulder_center is None or hip_center is None:
        return None

    dx = shoulder_center.x - hip_center.x
    dy = hip_center.y - shoulder_center.y
    if dx == 0 and dy == 0:
        return None
    return degrees(atan2(dx, dy))


def _joint_angle(a: Point2D | None, b: Point2D | None, c: Point2D | None) -> float | None:
    if a is None or b is None or c is None:
        return None

    ba_x = a.x - b.x
    ba_y = a.y - b.y
    bc_x = c.x - b.x
    bc_y = c.y - b.y

    ba_length = sqrt(ba_x * ba_x + ba_y * ba_y)
    bc_length = sqrt(bc_x * bc_x + bc_y * bc_y)
    if ba_length == 0 or bc_length == 0:
        return None

    cosine = (ba_x * bc_x + ba_y * bc_y) / (ba_length * bc_length)
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))


def _center_of_body_x(
    pose_landmarks: Sequence[Any],
    visibility_threshold: float,
) -> float | None:
    points = [
        point.x
        for index in CORE_BODY_INDICES
        if (point := _visible_point(pose_landmarks, index, visibility_threshold)) is not None
    ]
    if not points:
        return None
    return sum(points) / len(points)


def _average_landmark_score(pose_landmarks: Sequence[Any], key: str) -> float | None:
    values = [
        value
        for landmark in pose_landmarks
        if (value := landmark_value(landmark, key)) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)

