from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kinesis.experiments.exp001.config import PoseEstimationConfig
from kinesis.experiments.exp001.processor import ProcessingSummary, process_video
from kinesis.experiments.exp002.matching import (
    MatchConfig,
    MovementMatchSummary,
    compare_movement_csvs,
)


@dataclass(frozen=True)
class VideoMatchSummary:
    reference_processing: ProcessingSummary
    practice_processing: ProcessingSummary
    movement_match: MovementMatchSummary


def compare_movement_videos(
    *,
    reference_video_path: Path,
    practice_video_path: Path,
    output_dir: Path,
    pose_config: PoseEstimationConfig,
    match_config: MatchConfig | None = None,
) -> VideoMatchSummary:
    """Process two videos with EXP-001, then compare their movement CSVs with EXP-002."""

    output_dir.mkdir(parents=True, exist_ok=True)

    reference_processing = process_video(
        input_path=reference_video_path,
        output_dir=output_dir / "reference",
        config=pose_config,
    )
    practice_processing = process_video(
        input_path=practice_video_path,
        output_dir=output_dir / "practice",
        config=pose_config,
    )
    movement_match = compare_movement_csvs(
        reference_csv_path=reference_processing.analysis_csv_path,
        practice_csv_path=practice_processing.analysis_csv_path,
        output_dir=output_dir / "match",
        config=match_config,
    )

    return VideoMatchSummary(
        reference_processing=reference_processing,
        practice_processing=practice_processing,
        movement_match=movement_match,
    )
