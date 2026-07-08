from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kinesis.experiments.exp001.config import PoseEstimationConfig
from kinesis.experiments.exp001.movement_summary import MovementSummary


def write_run_manifest(
    *,
    manifest_path: Path,
    input_path: Path,
    output_video_path: Path,
    analysis_csv_path: Path,
    keyframe_paths: list[Path],
    plot_paths: list[Path],
    config: PoseEstimationConfig,
    movement_summary: MovementSummary,
    total_frames_reported: int,
    processed_frames: int,
    frames_with_pose: int,
    fps: float,
    width: int,
    height: int,
) -> Path:
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "EXP-001",
        "input": {
            "path": str(input_path),
            "name": input_path.name,
        },
        "outputs": {
            "video": str(output_video_path),
            "movement_csv": str(analysis_csv_path),
            "keyframes": [str(path) for path in keyframe_paths],
            "plots": [str(path) for path in plot_paths],
            "manifest": str(manifest_path),
        },
        "video_metadata": {
            "total_frames_reported": total_frames_reported,
            "processed_frames": processed_frames,
            "frames_with_pose": frames_with_pose,
            "fps": fps,
            "width": width,
            "height": height,
        },
        "config": _config_dict(config),
        "movement_summary": asdict(movement_summary),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _config_dict(config: PoseEstimationConfig) -> dict[str, Any]:
    values = asdict(config)
    values["model_path"] = str(config.model_path)
    return values

