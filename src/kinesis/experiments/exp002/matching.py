from __future__ import annotations

import csv
import json
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

MOVEMENT_METRICS: tuple[str, ...] = (
    "shoulder_height_asymmetry",
    "hip_height_asymmetry",
    "torso_lean_angle_degrees",
    "left_knee_angle_degrees",
    "right_knee_angle_degrees",
    "left_elbow_angle_degrees",
    "right_elbow_angle_degrees",
    "center_of_body_x",
)


@dataclass(frozen=True)
class MetricComparisonRule:
    scale: float
    weight: float = 1.0


METRIC_COMPARISON_RULES: dict[str, MetricComparisonRule] = {
    "shoulder_height_asymmetry": MetricComparisonRule(scale=0.12, weight=0.75),
    "hip_height_asymmetry": MetricComparisonRule(scale=0.12, weight=0.75),
    "torso_lean_angle_degrees": MetricComparisonRule(scale=35.0, weight=1.0),
    "left_knee_angle_degrees": MetricComparisonRule(scale=45.0, weight=1.25),
    "right_knee_angle_degrees": MetricComparisonRule(scale=45.0, weight=1.25),
    "left_elbow_angle_degrees": MetricComparisonRule(scale=50.0, weight=1.0),
    "right_elbow_angle_degrees": MetricComparisonRule(scale=50.0, weight=1.0),
    "center_of_body_x": MetricComparisonRule(scale=0.25, weight=0.75),
}


@dataclass(frozen=True)
class MatchConfig:
    max_offset_seconds: float = 1.0
    offset_step_seconds: float | None = None
    timestamp_tolerance_seconds: float | None = None
    min_metrics_per_frame: int = 3
    min_compared_frames: int = 5
    largest_moment_count: int = 5

    def validate(self) -> None:
        if self.max_offset_seconds < 0:
            raise ValueError("max_offset_seconds must be 0 or greater.")
        if self.offset_step_seconds is not None and self.offset_step_seconds <= 0:
            raise ValueError("offset_step_seconds must be greater than 0.")
        if self.timestamp_tolerance_seconds is not None and self.timestamp_tolerance_seconds <= 0:
            raise ValueError("timestamp_tolerance_seconds must be greater than 0.")
        if self.min_metrics_per_frame < 1:
            raise ValueError("min_metrics_per_frame must be at least 1.")
        if self.min_compared_frames < 1:
            raise ValueError("min_compared_frames must be at least 1.")
        if self.largest_moment_count < 0:
            raise ValueError("largest_moment_count must be 0 or greater.")


@dataclass(frozen=True)
class MovementFrame:
    timestamp_seconds: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class FrameMatch:
    reference_timestamp_seconds: float
    practice_timestamp_seconds: float
    match_score: float
    compared_metric_count: int
    metric_differences: dict[str, float]
    largest_difference_metric: str | None
    largest_normalized_difference: float | None


@dataclass(frozen=True)
class MovementMatchSummary:
    overall_score: float
    best_time_offset_seconds: float
    compared_frame_count: int
    average_normalized_error: float
    report_csv_path: Path
    summary_json_path: Path
    largest_difference_moments: list[FrameMatch]


def compare_movement_csvs(
    *,
    reference_csv_path: Path,
    practice_csv_path: Path,
    output_dir: Path,
    config: MatchConfig | None = None,
) -> MovementMatchSummary:
    config = config or MatchConfig()
    config.validate()

    reference_frames = load_movement_frames(reference_csv_path)
    practice_frames = load_movement_frames(practice_csv_path)
    if not reference_frames:
        raise ValueError(f"No comparable movement frames found in {reference_csv_path}")
    if not practice_frames:
        raise ValueError(f"No comparable movement frames found in {practice_csv_path}")

    frame_step = _estimate_frame_step(reference_frames, practice_frames)
    offset_step_seconds = config.offset_step_seconds or frame_step
    timestamp_tolerance_seconds = config.timestamp_tolerance_seconds or frame_step * 0.6

    best_matches: list[FrameMatch] = []
    best_offset = 0.0
    best_error = float("inf")

    for offset_seconds in _candidate_offsets(
        max_offset_seconds=config.max_offset_seconds,
        step_seconds=offset_step_seconds,
    ):
        matches = _compare_at_offset(
            reference_frames=reference_frames,
            practice_frames=practice_frames,
            offset_seconds=offset_seconds,
            timestamp_tolerance_seconds=timestamp_tolerance_seconds,
            min_metrics_per_frame=config.min_metrics_per_frame,
        )
        if len(matches) < config.min_compared_frames:
            continue

        average_error = _average_frame_error(matches)
        if average_error < best_error:
            best_error = average_error
            best_offset = offset_seconds
            best_matches = matches

    if not best_matches:
        raise ValueError(
            "Could not compare enough frames. Try a larger max offset, lower the minimum "
            "compared frames, or confirm both CSVs came from similar video clips."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_csv_path = output_dir / "movement_match_report.csv"
    summary_json_path = output_dir / "movement_match_summary.json"
    largest_moments = sorted(best_matches, key=lambda match: match.match_score)[
        : config.largest_moment_count
    ]
    overall_score = _score_from_error(best_error)

    _write_report_csv(report_csv_path, best_matches)
    summary = MovementMatchSummary(
        overall_score=overall_score,
        best_time_offset_seconds=best_offset,
        compared_frame_count=len(best_matches),
        average_normalized_error=best_error,
        report_csv_path=report_csv_path,
        summary_json_path=summary_json_path,
        largest_difference_moments=largest_moments,
    )
    _write_summary_json(
        summary_json_path=summary_json_path,
        summary=summary,
        reference_csv_path=reference_csv_path,
        practice_csv_path=practice_csv_path,
        config=config,
    )
    return summary


def load_movement_frames(csv_path: Path) -> list[MovementFrame]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return []
        has_smoothed_columns = any(
            f"smoothed_{metric_name}" in reader.fieldnames for metric_name in MOVEMENT_METRICS
        )
        frames = [
            frame
            for row in reader
            if (frame := _movement_frame_from_row(row, has_smoothed_columns)) is not None
        ]
    return sorted(frames, key=lambda frame: frame.timestamp_seconds)


def _movement_frame_from_row(
    row: dict[str, str],
    has_smoothed_columns: bool,
) -> MovementFrame | None:
    timestamp_seconds = _parse_float(row.get("timestamp_seconds"))
    if timestamp_seconds is None:
        return None

    metrics: dict[str, float] = {}
    for metric_name in MOVEMENT_METRICS:
        column_name = f"smoothed_{metric_name}" if has_smoothed_columns else metric_name
        value = _parse_float(row.get(column_name))
        if value is None:
            continue
        if not has_smoothed_columns and not _metric_quality_is_usable(row, metric_name):
            continue
        metrics[metric_name] = value

    if not metrics:
        return None
    return MovementFrame(timestamp_seconds=timestamp_seconds, metrics=metrics)


def _metric_quality_is_usable(row: dict[str, str], metric_name: str) -> bool:
    value = row.get(f"{metric_name}_quality")
    if value is None or value == "":
        return True
    return value.lower() in {"true", "1", "yes"}


def _compare_at_offset(
    *,
    reference_frames: list[MovementFrame],
    practice_frames: list[MovementFrame],
    offset_seconds: float,
    timestamp_tolerance_seconds: float,
    min_metrics_per_frame: int,
) -> list[FrameMatch]:
    practice_timestamps = [frame.timestamp_seconds for frame in practice_frames]
    matches: list[FrameMatch] = []

    for reference_frame in reference_frames:
        target_timestamp = reference_frame.timestamp_seconds + offset_seconds
        practice_frame = _nearest_frame(
            frames=practice_frames,
            timestamps=practice_timestamps,
            target_timestamp=target_timestamp,
            tolerance_seconds=timestamp_tolerance_seconds,
        )
        if practice_frame is None:
            continue

        frame_match = _compare_frame_pair(
            reference_frame,
            practice_frame,
            min_metrics_per_frame=min_metrics_per_frame,
        )
        if frame_match is not None:
            matches.append(frame_match)

    return matches


def _compare_frame_pair(
    reference_frame: MovementFrame,
    practice_frame: MovementFrame,
    *,
    min_metrics_per_frame: int,
) -> FrameMatch | None:
    metric_differences: dict[str, float] = {}
    weighted_error_total = 0.0
    weight_total = 0.0
    largest_metric: str | None = None
    largest_normalized_difference: float | None = None

    for metric_name, rule in METRIC_COMPARISON_RULES.items():
        reference_value = reference_frame.metrics.get(metric_name)
        practice_value = practice_frame.metrics.get(metric_name)
        if reference_value is None or practice_value is None:
            continue

        difference = abs(practice_value - reference_value)
        normalized_difference = difference / rule.scale
        metric_differences[metric_name] = difference
        weighted_error_total += min(normalized_difference, 1.0) * rule.weight
        weight_total += rule.weight

        if (
            largest_normalized_difference is None
            or normalized_difference > largest_normalized_difference
        ):
            largest_normalized_difference = normalized_difference
            largest_metric = metric_name

    if len(metric_differences) < min_metrics_per_frame or weight_total == 0:
        return None

    normalized_error = weighted_error_total / weight_total
    return FrameMatch(
        reference_timestamp_seconds=reference_frame.timestamp_seconds,
        practice_timestamp_seconds=practice_frame.timestamp_seconds,
        match_score=_score_from_error(normalized_error),
        compared_metric_count=len(metric_differences),
        metric_differences=metric_differences,
        largest_difference_metric=largest_metric,
        largest_normalized_difference=largest_normalized_difference,
    )


def _nearest_frame(
    *,
    frames: list[MovementFrame],
    timestamps: list[float],
    target_timestamp: float,
    tolerance_seconds: float,
) -> MovementFrame | None:
    insertion_index = bisect_left(timestamps, target_timestamp)
    candidates = []
    if insertion_index < len(frames):
        candidates.append(frames[insertion_index])
    if insertion_index > 0:
        candidates.append(frames[insertion_index - 1])
    if not candidates:
        return None

    nearest = min(
        candidates,
        key=lambda frame: abs(frame.timestamp_seconds - target_timestamp),
    )
    if abs(nearest.timestamp_seconds - target_timestamp) > tolerance_seconds:
        return None
    return nearest


def _candidate_offsets(*, max_offset_seconds: float, step_seconds: float) -> list[float]:
    offset_count = max(0, round(max_offset_seconds / step_seconds))
    offsets = [round(index * step_seconds, 6) for index in range(-offset_count, offset_count + 1)]
    if 0.0 not in offsets:
        offsets.append(0.0)
    return sorted(set(offsets), key=lambda value: (abs(value), value))


def _estimate_frame_step(
    reference_frames: list[MovementFrame],
    practice_frames: list[MovementFrame],
) -> float:
    deltas = [
        later.timestamp_seconds - earlier.timestamp_seconds
        for frames in (reference_frames, practice_frames)
        for earlier, later in zip(frames, frames[1:], strict=False)
        if later.timestamp_seconds > earlier.timestamp_seconds
    ]
    if not deltas:
        return 1 / 30
    return max(median(deltas), 1 / 120)


def _average_frame_error(matches: list[FrameMatch]) -> float:
    if not matches:
        return float("inf")
    return sum(1 - match.match_score / 100 for match in matches) / len(matches)


def _score_from_error(normalized_error: float) -> float:
    return round(max(0.0, min(100.0, 100 * (1 - normalized_error))), 1)


def _write_report_csv(report_csv_path: Path, matches: list[FrameMatch]) -> None:
    fieldnames = [
        "reference_timestamp_seconds",
        "practice_timestamp_seconds",
        "match_score",
        "compared_metric_count",
        "largest_difference_metric",
        "largest_normalized_difference",
        *(f"{metric_name}_absolute_difference" for metric_name in MOVEMENT_METRICS),
    ]
    with report_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            row: dict[str, float | int | str | None] = {
                "reference_timestamp_seconds": match.reference_timestamp_seconds,
                "practice_timestamp_seconds": match.practice_timestamp_seconds,
                "match_score": match.match_score,
                "compared_metric_count": match.compared_metric_count,
                "largest_difference_metric": match.largest_difference_metric,
                "largest_normalized_difference": match.largest_normalized_difference,
            }
            for metric_name in MOVEMENT_METRICS:
                row[f"{metric_name}_absolute_difference"] = match.metric_differences.get(
                    metric_name
                )
            writer.writerow(row)


def _write_summary_json(
    *,
    summary_json_path: Path,
    summary: MovementMatchSummary,
    reference_csv_path: Path,
    practice_csv_path: Path,
    config: MatchConfig,
) -> None:
    payload = {
        "reference_csv": str(reference_csv_path),
        "practice_csv": str(practice_csv_path),
        "overall_score": summary.overall_score,
        "best_time_offset_seconds": summary.best_time_offset_seconds,
        "compared_frame_count": summary.compared_frame_count,
        "average_normalized_error": summary.average_normalized_error,
        "report_csv": str(summary.report_csv_path),
        "config": asdict(config),
        "largest_difference_moments": [
            {
                "reference_timestamp_seconds": moment.reference_timestamp_seconds,
                "practice_timestamp_seconds": moment.practice_timestamp_seconds,
                "match_score": moment.match_score,
                "compared_metric_count": moment.compared_metric_count,
                "largest_difference_metric": moment.largest_difference_metric,
                "largest_normalized_difference": moment.largest_normalized_difference,
                "metric_differences": moment.metric_differences,
            }
            for moment in summary.largest_difference_moments
        ],
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
