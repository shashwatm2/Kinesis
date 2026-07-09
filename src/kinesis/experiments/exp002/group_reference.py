from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinesis.experiments.exp001.config import PoseEstimationConfig
from kinesis.experiments.exp001.csv_export import (
    analysis_csv_fieldnames,
    build_analysis_csv_row,
)
from kinesis.experiments.exp001.drawing import draw_pose_landmarks, extract_keyframe_indices
from kinesis.experiments.exp001.landmarks import (
    landmark_has_visible_position,
    landmark_value,
)
from kinesis.experiments.exp001.metrics import (
    CORE_BODY_INDICES,
    FrameMetrics,
    calculate_frame_metrics,
)
from kinesis.experiments.exp001.processor import (
    _create_pose_landmarker,
    _frame_window,
    _import_cv2,
    _import_mediapipe,
)
from kinesis.experiments.exp001.quality import MetricQualityEvaluator, MetricQualityResult
from kinesis.experiments.exp001.smoothing import MetricSmoother
from kinesis.experiments.exp002.matching import (
    MatchConfig,
    MovementMatchSummary,
    compare_movement_csvs,
)

TRACK_COLORS: tuple[tuple[int, int, int], ...] = (
    (49, 204, 132),
    (255, 184, 77),
    (80, 180, 255),
    (214, 112, 255),
    (120, 220, 220),
    (255, 120, 120),
)
TRACK_SIGNATURE_LANDMARKS: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


@dataclass(frozen=True)
class BodyCenter:
    x: float
    y: float


@dataclass(frozen=True)
class PoseDetection:
    pose_landmarks: tuple[Any, ...]
    center: BodyCenter
    pose_signature: dict[int, tuple[float, float]]


@dataclass(frozen=True)
class TrackAssignment:
    track_id: int
    pose_landmarks: tuple[Any, ...]
    center: BodyCenter


@dataclass(frozen=True)
class TrackFrameAnalysis:
    frame_index: int
    timestamp_ms: int
    pose_landmarks: tuple[Any, ...]
    metrics: FrameMetrics
    metric_quality: dict[str, MetricQualityResult]
    smoothed_metrics: FrameMetrics


@dataclass(frozen=True)
class TrackSummary:
    track_id: int
    movement_csv_path: Path
    frame_count: int
    first_timestamp_seconds: float
    last_timestamp_seconds: float


@dataclass(frozen=True)
class TrackComparisonSummary:
    track_id: int
    match_summary: MovementMatchSummary | None
    error: str | None = None


@dataclass(frozen=True)
class GroupReferenceSummary:
    input_path: Path
    output_video_path: Path
    manifest_path: Path
    keyframe_paths: list[Path]
    reference_track_id: int
    track_summaries: list[TrackSummary]
    comparisons: list[TrackComparisonSummary]
    total_frames: int
    processed_frames: int
    frames_with_pose: int
    frames_with_multiple_poses: int
    fps: float
    width: int
    height: int


@dataclass
class _TrackState:
    track_id: int
    center: BodyCenter
    pose_signature: dict[int, tuple[float, float]]
    last_frame_index: int
    velocity_x_per_frame: float = 0.0
    velocity_y_per_frame: float = 0.0


@dataclass
class _TrackAnalysisState:
    quality_evaluator: MetricQualityEvaluator
    smoother: MetricSmoother
    frames: list[TrackFrameAnalysis]


class PoseTrackAssigner:
    def __init__(
        self,
        *,
        max_assignment_cost: float = 0.35,
        max_missing_frames: int = 15,
    ) -> None:
        self._max_assignment_cost = max_assignment_cost
        self._max_missing_frames = max_missing_frames
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def assign(
        self,
        detections: list[PoseDetection],
        *,
        frame_index: int,
    ) -> list[TrackAssignment]:
        if not detections:
            return []

        active_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame_index <= self._max_missing_frames
        ]
        candidate_pairs = sorted(
            (
                (
                    _assignment_cost(
                        track=self._tracks[track_id],
                        detection=detection,
                        frame_index=frame_index,
                    ),
                    track_id,
                    index,
                )
                for track_id in active_track_ids
                for index, detection in enumerate(detections)
            ),
            key=lambda item: item[0],
        )

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        assignments: list[TrackAssignment] = []

        for distance, track_id, detection_index in candidate_pairs:
            if distance > self._max_assignment_cost:
                continue
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue

            detection = detections[detection_index]
            track = self._tracks[track_id]
            frames_elapsed = max(1, frame_index - track.last_frame_index)
            track.velocity_x_per_frame = (detection.center.x - track.center.x) / frames_elapsed
            track.velocity_y_per_frame = (detection.center.y - track.center.y) / frames_elapsed
            track.center = detection.center
            track.pose_signature = detection.pose_signature
            track.last_frame_index = frame_index
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            assignments.append(
                TrackAssignment(
                    track_id=track_id,
                    pose_landmarks=detection.pose_landmarks,
                    center=detection.center,
                )
            )

        unassigned_detection_indices = [
            index for index in range(len(detections)) if index not in assigned_detections
        ]
        for detection_index in sorted(
            unassigned_detection_indices,
            key=lambda index: detections[index].center.x,
        ):
            detection = detections[detection_index]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                center=detection.center,
                pose_signature=detection.pose_signature,
                last_frame_index=frame_index,
            )
            assignments.append(
                TrackAssignment(
                    track_id=track_id,
                    pose_landmarks=detection.pose_landmarks,
                    center=detection.center,
                )
            )

        return sorted(assignments, key=lambda assignment: assignment.track_id)


def compare_group_video_to_reference(
    *,
    input_path: Path,
    output_dir: Path,
    pose_config: PoseEstimationConfig,
    reference_track_id: int,
    match_config: MatchConfig | None = None,
) -> GroupReferenceSummary:
    pose_config.validate()
    if reference_track_id < 1:
        raise ValueError("reference_track_id must be at least 1.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cv2 = _import_cv2()
    mp = _import_mediapipe()

    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = output_dir / "tracks"
    matches_dir = output_dir / "matches"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    matches_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = output_dir / f"{input_path.stem}_group_reference_overlay.mp4"
    manifest_path = output_dir / "group_reference_summary.json"

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_fps = fps / pose_config.frame_stride
    start_frame_index, end_frame_index = _frame_window(
        fps=fps,
        total_frames=total_frames,
        start_time_seconds=pose_config.start_time_seconds,
        end_time_seconds=pose_config.end_time_seconds,
    )

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    selected_frame_count = max(0, end_frame_index - start_frame_index)
    keyframe_indices = set(
        extract_keyframe_indices(selected_frame_count, pose_config.max_keyframes)
    )
    keyframe_paths: list[Path] = []
    assigner = PoseTrackAssigner()
    track_states: dict[int, _TrackAnalysisState] = {}
    processed_frames = 0
    frames_with_pose = 0
    frames_with_multiple_poses = 0
    source_frame_index = 0

    try:
        with _create_pose_landmarker(mp, pose_config) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if source_frame_index >= end_frame_index:
                    break
                if source_frame_index < start_frame_index:
                    source_frame_index += 1
                    continue

                if source_frame_index % pose_config.frame_stride != 0:
                    source_frame_index += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(source_frame_index * 1000 / fps)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                detections = _pose_detections(
                    result.pose_landmarks,
                    visibility_threshold=pose_config.landmark_visibility_threshold,
                )
                assignments = assigner.assign(
                    detections,
                    frame_index=source_frame_index,
                )
                if assignments:
                    frames_with_pose += 1
                if len(assignments) > 1:
                    frames_with_multiple_poses += 1

                annotated_frame = frame
                for assignment in assignments:
                    track_state = _track_state_for_assignment(
                        track_states,
                        track_id=assignment.track_id,
                        pose_config=pose_config,
                    )
                    metrics = calculate_frame_metrics(
                        assignment.pose_landmarks,
                        visibility_threshold=pose_config.landmark_visibility_threshold,
                    )
                    metric_quality = track_state.quality_evaluator.evaluate(
                        pose_landmarks=assignment.pose_landmarks,
                        metrics=metrics,
                    )
                    smoothed_metrics = track_state.smoother.smooth(
                        metrics=metrics,
                        quality=metric_quality,
                    )
                    track_state.frames.append(
                        TrackFrameAnalysis(
                            frame_index=source_frame_index,
                            timestamp_ms=timestamp_ms,
                            pose_landmarks=assignment.pose_landmarks,
                            metrics=metrics,
                            metric_quality=metric_quality,
                            smoothed_metrics=smoothed_metrics,
                        )
                    )

                    color = _track_color(assignment.track_id)
                    annotated_frame = draw_pose_landmarks(
                        annotated_frame,
                        assignment.pose_landmarks,
                        visibility_threshold=pose_config.landmark_visibility_threshold,
                        connection_color=color,
                        joint_color=color,
                    )
                    _draw_track_label(
                        cv2,
                        annotated_frame,
                        track_id=assignment.track_id,
                        center=assignment.center,
                        color=color,
                    )

                writer.write(annotated_frame)
                processed_frames += 1

                if (source_frame_index - start_frame_index) in keyframe_indices and len(
                    keyframe_paths
                ) < pose_config.max_keyframes:
                    keyframe_path = output_dir / f"group_keyframe_{len(keyframe_paths) + 1:03d}.jpg"
                    cv2.imwrite(str(keyframe_path), annotated_frame)
                    keyframe_paths.append(keyframe_path)

                source_frame_index += 1
    finally:
        writer.release()
        capture.release()

    track_summaries = _write_track_csvs(
        tracks_dir=tracks_dir,
        track_states=track_states,
    )
    reference_track = next(
        (track for track in track_summaries if track.track_id == reference_track_id),
        None,
    )
    if reference_track is None:
        raise ValueError(
            f"Reference track {reference_track_id} was not detected. "
            f"Detected tracks: {[track.track_id for track in track_summaries]}"
        )

    comparisons = _compare_tracks_to_reference(
        reference_track=reference_track,
        track_summaries=track_summaries,
        matches_dir=matches_dir,
        match_config=match_config,
    )
    summary = GroupReferenceSummary(
        input_path=input_path,
        output_video_path=output_video_path,
        manifest_path=manifest_path,
        keyframe_paths=keyframe_paths,
        reference_track_id=reference_track_id,
        track_summaries=track_summaries,
        comparisons=comparisons,
        total_frames=total_frames,
        processed_frames=processed_frames,
        frames_with_pose=frames_with_pose,
        frames_with_multiple_poses=frames_with_multiple_poses,
        fps=fps,
        width=width,
        height=height,
    )
    _write_group_manifest(summary)
    return summary


def _pose_detections(
    pose_landmarks_list: list[Any],
    *,
    visibility_threshold: float,
) -> list[PoseDetection]:
    detections: list[PoseDetection] = []
    for pose_landmarks in pose_landmarks_list:
        pose_tuple = tuple(pose_landmarks)
        center = _body_center(
            pose_tuple,
            visibility_threshold=visibility_threshold,
        )
        if center is None:
            continue
        detections.append(
            PoseDetection(
                pose_landmarks=pose_tuple,
                center=center,
                pose_signature=_pose_signature(
                    pose_tuple,
                    center=center,
                    visibility_threshold=visibility_threshold,
                ),
            )
        )
    return detections


def _body_center(
    pose_landmarks: tuple[Any, ...],
    *,
    visibility_threshold: float,
) -> BodyCenter | None:
    xs: list[float] = []
    ys: list[float] = []
    for index in CORE_BODY_INDICES:
        if index >= len(pose_landmarks):
            continue
        landmark = pose_landmarks[index]
        if not landmark_has_visible_position(landmark, visibility_threshold):
            continue
        x = landmark_value(landmark, "x")
        y = landmark_value(landmark, "y")
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None
    return BodyCenter(x=sum(xs) / len(xs), y=sum(ys) / len(ys))


def _pose_signature(
    pose_landmarks: tuple[Any, ...],
    *,
    center: BodyCenter,
    visibility_threshold: float,
) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    xs: list[float] = []
    ys: list[float] = []

    for index in TRACK_SIGNATURE_LANDMARKS:
        if index >= len(pose_landmarks):
            continue
        landmark = pose_landmarks[index]
        if not landmark_has_visible_position(landmark, visibility_threshold):
            continue
        x = landmark_value(landmark, "x")
        y = landmark_value(landmark, "y")
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        points[index] = (x, y)

    if not points:
        return {}

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 0.05)
    return {
        index: ((x - center.x) / scale, (y - center.y) / scale) for index, (x, y) in points.items()
    }


def _track_state_for_assignment(
    track_states: dict[int, _TrackAnalysisState],
    *,
    track_id: int,
    pose_config: PoseEstimationConfig,
) -> _TrackAnalysisState:
    if track_id not in track_states:
        track_states[track_id] = _TrackAnalysisState(
            quality_evaluator=MetricQualityEvaluator(
                visibility_threshold=pose_config.landmark_visibility_threshold,
                min_average_visibility=pose_config.metric_min_average_visibility,
            ),
            smoother=MetricSmoother(window_size=pose_config.smoothing_window_frames),
            frames=[],
        )
    return track_states[track_id]


def _write_track_csvs(
    *,
    tracks_dir: Path,
    track_states: dict[int, _TrackAnalysisState],
) -> list[TrackSummary]:
    summaries: list[TrackSummary] = []
    for track_id, track_state in sorted(track_states.items()):
        if not track_state.frames:
            continue

        movement_csv_path = tracks_dir / f"track_{track_id:02d}_movement_analysis.csv"
        with movement_csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=analysis_csv_fieldnames())
            writer.writeheader()
            for frame in track_state.frames:
                writer.writerow(
                    build_analysis_csv_row(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        pose_landmarks=frame.pose_landmarks,
                        metrics=frame.metrics,
                        metric_quality=frame.metric_quality,
                        smoothed_metrics=frame.smoothed_metrics,
                    )
                )

        summaries.append(
            TrackSummary(
                track_id=track_id,
                movement_csv_path=movement_csv_path,
                frame_count=len(track_state.frames),
                first_timestamp_seconds=track_state.frames[0].timestamp_ms / 1000,
                last_timestamp_seconds=track_state.frames[-1].timestamp_ms / 1000,
            )
        )
    return summaries


def _compare_tracks_to_reference(
    *,
    reference_track: TrackSummary,
    track_summaries: list[TrackSummary],
    matches_dir: Path,
    match_config: MatchConfig | None,
) -> list[TrackComparisonSummary]:
    comparisons: list[TrackComparisonSummary] = []
    for track in track_summaries:
        if track.track_id == reference_track.track_id:
            continue

        try:
            match_summary = compare_movement_csvs(
                reference_csv_path=reference_track.movement_csv_path,
                practice_csv_path=track.movement_csv_path,
                output_dir=matches_dir
                / f"reference_track_{reference_track.track_id:02d}_vs_track_{track.track_id:02d}",
                config=match_config,
            )
            comparisons.append(
                TrackComparisonSummary(
                    track_id=track.track_id,
                    match_summary=match_summary,
                )
            )
        except ValueError as exc:
            comparisons.append(
                TrackComparisonSummary(
                    track_id=track.track_id,
                    match_summary=None,
                    error=str(exc),
                )
            )
    return comparisons


def _write_group_manifest(summary: GroupReferenceSummary) -> None:
    payload = {
        "input": str(summary.input_path),
        "output_video": str(summary.output_video_path),
        "reference_track_id": summary.reference_track_id,
        "keyframes": [str(path) for path in summary.keyframe_paths],
        "tracks": [
            {
                "track_id": track.track_id,
                "movement_csv": str(track.movement_csv_path),
                "frame_count": track.frame_count,
                "first_timestamp_seconds": track.first_timestamp_seconds,
                "last_timestamp_seconds": track.last_timestamp_seconds,
            }
            for track in summary.track_summaries
        ],
        "comparisons": [
            {
                "track_id": comparison.track_id,
                "error": comparison.error,
                "overall_score": (
                    comparison.match_summary.overall_score
                    if comparison.match_summary is not None
                    else None
                ),
                "best_time_offset_seconds": (
                    comparison.match_summary.best_time_offset_seconds
                    if comparison.match_summary is not None
                    else None
                ),
                "compared_frame_count": (
                    comparison.match_summary.compared_frame_count
                    if comparison.match_summary is not None
                    else None
                ),
                "report_csv": (
                    str(comparison.match_summary.report_csv_path)
                    if comparison.match_summary is not None
                    else None
                ),
                "summary_json": (
                    str(comparison.match_summary.summary_json_path)
                    if comparison.match_summary is not None
                    else None
                ),
            }
            for comparison in summary.comparisons
        ],
        "video": {
            "total_frames_reported": summary.total_frames,
            "processed_frames": summary.processed_frames,
            "frames_with_pose": summary.frames_with_pose,
            "frames_with_multiple_poses": summary.frames_with_multiple_poses,
            "fps": summary.fps,
            "width": summary.width,
            "height": summary.height,
        },
    }
    summary.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _track_color(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]


def _draw_track_label(
    cv2: Any,
    frame: Any,
    *,
    track_id: int,
    center: BodyCenter,
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    position = (
        min(width - 1, max(0, round(center.x * (width - 1)))),
        min(height - 1, max(0, round(center.y * (height - 1)))),
    )
    cv2.putText(
        frame,
        f"ID {track_id}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def _center_distance(a: BodyCenter, b: BodyCenter) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def _assignment_cost(
    *,
    track: _TrackState,
    detection: PoseDetection,
    frame_index: int,
) -> float:
    frames_elapsed = max(1, frame_index - track.last_frame_index)
    predicted_center = BodyCenter(
        x=track.center.x + track.velocity_x_per_frame * frames_elapsed,
        y=track.center.y + track.velocity_y_per_frame * frames_elapsed,
    )
    motion_cost = _center_distance(predicted_center, detection.center)
    pose_cost = _pose_signature_distance(track.pose_signature, detection.pose_signature)
    if pose_cost is None:
        pose_cost = 0.25

    return motion_cost * 0.70 + min(pose_cost, 1.0) * 0.30


def _pose_signature_distance(
    a: dict[int, tuple[float, float]],
    b: dict[int, tuple[float, float]],
) -> float | None:
    common_indices = sorted(set(a) & set(b))
    if len(common_indices) < 4:
        return None

    distances = []
    for index in common_indices:
        ax, ay = a[index]
        bx, by = b[index]
        distances.append(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    return sum(distances) / len(distances)
