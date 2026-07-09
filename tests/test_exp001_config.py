from __future__ import annotations

from pathlib import Path

import pytest

from kinesis.experiments.exp001.config import PoseEstimationConfig


def test_pose_estimation_config_accepts_valid_trim_window(tmp_path: Path) -> None:
    config = PoseEstimationConfig(
        model_path=tmp_path,
        start_time_seconds=1.0,
        end_time_seconds=3.0,
    )

    config.validate()


def test_pose_estimation_config_rejects_empty_trim_window(tmp_path: Path) -> None:
    config = PoseEstimationConfig(
        model_path=tmp_path,
        start_time_seconds=3.0,
        end_time_seconds=3.0,
    )

    with pytest.raises(ValueError, match="end_time_seconds"):
        config.validate()
