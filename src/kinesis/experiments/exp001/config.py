from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PoseEstimationConfig:
    model_path: Path
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    landmark_visibility_threshold: float = 0.5
    metric_min_average_visibility: float = 0.5
    smoothing_window_frames: int = 5
    max_keyframes: int = 6
    frame_stride: int = 1
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None

    def validate(self) -> None:
        if self.num_poses < 1:
            raise ValueError("num_poses must be at least 1.")
        if self.max_keyframes < 0:
            raise ValueError("max_keyframes must be 0 or greater.")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1.")
        if self.smoothing_window_frames < 1:
            raise ValueError("smoothing_window_frames must be at least 1.")
        if self.start_time_seconds is not None and self.start_time_seconds < 0:
            raise ValueError("start_time_seconds must be 0 or greater.")
        if self.end_time_seconds is not None and self.end_time_seconds <= 0:
            raise ValueError("end_time_seconds must be greater than 0.")
        if (
            self.start_time_seconds is not None
            and self.end_time_seconds is not None
            and self.end_time_seconds <= self.start_time_seconds
        ):
            raise ValueError("end_time_seconds must be greater than start_time_seconds.")

        confidence_values = {
            "min_pose_detection_confidence": self.min_pose_detection_confidence,
            "min_pose_presence_confidence": self.min_pose_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "landmark_visibility_threshold": self.landmark_visibility_threshold,
            "metric_min_average_visibility": self.metric_min_average_visibility,
        }
        for name, value in confidence_values.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self.model_path}. "
                "Download pose_landmarker_full.task into models/ first."
            )
