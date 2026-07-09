from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinesis.experiments.exp001.config import PoseEstimationConfig
from kinesis.experiments.exp001.csv_export import (
    analysis_csv_fieldnames,
    build_analysis_csv_row,
)
from kinesis.experiments.exp001.drawing import draw_pose_landmarks, extract_keyframe_indices
from kinesis.experiments.exp001.frame_analysis import FrameAnalysis
from kinesis.experiments.exp001.manifest import write_run_manifest
from kinesis.experiments.exp001.metrics import calculate_frame_metrics
from kinesis.experiments.exp001.movement_summary import MovementSummary, summarize_movement
from kinesis.experiments.exp001.plots import create_metric_plots
from kinesis.experiments.exp001.quality import MetricQualityEvaluator
from kinesis.experiments.exp001.smoothing import MetricSmoother


@dataclass(frozen=True)
class ProcessingSummary:
    input_path: Path
    output_video_path: Path
    analysis_csv_path: Path
    manifest_path: Path
    plot_paths: list[Path]
    keyframe_paths: list[Path]
    movement_summary: MovementSummary
    total_frames: int
    processed_frames: int
    frames_with_pose: int
    fps: float
    width: int
    height: int


def process_video(
    *,
    input_path: Path,
    output_dir: Path,
    config: PoseEstimationConfig,
    output_name: str | None = None,
) -> ProcessingSummary:
    config.validate()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cv2 = _import_cv2()
    mp = _import_mediapipe()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / (output_name or f"{input_path.stem}_pose_overlay.mp4")
    analysis_csv_path = output_dir / f"{input_path.stem}_movement_analysis.csv"
    manifest_path = output_dir / "run_manifest.json"

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_fps = fps / config.frame_stride
    start_frame_index, end_frame_index = _frame_window(
        fps=fps,
        total_frames=total_frames,
        start_time_seconds=config.start_time_seconds,
        end_time_seconds=config.end_time_seconds,
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

    keyframe_indices = set(
        extract_keyframe_indices(
            max(0, end_frame_index - start_frame_index),
            config.max_keyframes,
        )
    )
    keyframe_paths: list[Path] = []
    frame_analyses: list[FrameAnalysis] = []
    quality_evaluator = MetricQualityEvaluator(
        visibility_threshold=config.landmark_visibility_threshold,
        min_average_visibility=config.metric_min_average_visibility,
    )
    smoother = MetricSmoother(window_size=config.smoothing_window_frames)
    processed_frames = 0
    frames_with_pose = 0
    source_frame_index = 0

    try:
        with (
            analysis_csv_path.open("w", newline="") as analysis_csv_file,
            _create_pose_landmarker(mp, config) as landmarker,
        ):
            csv_writer = csv.DictWriter(
                analysis_csv_file,
                fieldnames=analysis_csv_fieldnames(),
            )
            csv_writer.writeheader()

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if source_frame_index >= end_frame_index:
                    break
                if source_frame_index < start_frame_index:
                    source_frame_index += 1
                    continue

                if source_frame_index % config.frame_stride != 0:
                    source_frame_index += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(source_frame_index * 1000 / fps)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                annotated_frame = frame
                primary_pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else ()
                metrics = calculate_frame_metrics(
                    primary_pose_landmarks,
                    visibility_threshold=config.landmark_visibility_threshold,
                )
                metric_quality = quality_evaluator.evaluate(
                    pose_landmarks=primary_pose_landmarks,
                    metrics=metrics,
                )
                smoothed_metrics = smoother.smooth(
                    metrics=metrics,
                    quality=metric_quality,
                )
                frame_analysis = FrameAnalysis(
                    frame_index=source_frame_index,
                    timestamp_ms=timestamp_ms,
                    raw_metrics=metrics,
                    smoothed_metrics=smoothed_metrics,
                    metric_quality=metric_quality,
                )
                frame_analyses.append(frame_analysis)
                csv_writer.writerow(
                    build_analysis_csv_row(
                        frame_index=source_frame_index,
                        timestamp_ms=timestamp_ms,
                        pose_landmarks=primary_pose_landmarks,
                        metrics=metrics,
                        metric_quality=metric_quality,
                        smoothed_metrics=smoothed_metrics,
                    )
                )

                if result.pose_landmarks:
                    frames_with_pose += 1
                    for pose_landmarks in result.pose_landmarks:
                        annotated_frame = draw_pose_landmarks(
                            annotated_frame,
                            pose_landmarks,
                            visibility_threshold=config.landmark_visibility_threshold,
                        )

                writer.write(annotated_frame)
                processed_frames += 1

                if _should_save_keyframe(
                    source_frame_index=source_frame_index,
                    start_frame_index=start_frame_index,
                    selected_frame_count=max(0, end_frame_index - start_frame_index),
                    keyframe_indices=keyframe_indices,
                    keyframe_count=len(keyframe_paths),
                    max_keyframes=config.max_keyframes,
                ):
                    keyframe_path = output_dir / f"keyframe_{len(keyframe_paths) + 1:03d}.jpg"
                    cv2.imwrite(str(keyframe_path), annotated_frame)
                    keyframe_paths.append(keyframe_path)

                source_frame_index += 1
    finally:
        writer.release()
        capture.release()

    movement_summary = summarize_movement(
        frame_analyses,
        processed_frames=processed_frames,
        frames_with_pose=frames_with_pose,
    )
    plot_paths = create_metric_plots(frame_analyses, output_dir=output_dir)
    write_run_manifest(
        manifest_path=manifest_path,
        input_path=input_path,
        output_video_path=output_video_path,
        analysis_csv_path=analysis_csv_path,
        keyframe_paths=keyframe_paths,
        plot_paths=plot_paths,
        config=config,
        movement_summary=movement_summary,
        total_frames_reported=total_frames,
        processed_frames=processed_frames,
        frames_with_pose=frames_with_pose,
        fps=fps,
        width=width,
        height=height,
    )

    return ProcessingSummary(
        input_path=input_path,
        output_video_path=output_video_path,
        analysis_csv_path=analysis_csv_path,
        manifest_path=manifest_path,
        plot_paths=plot_paths,
        keyframe_paths=keyframe_paths,
        movement_summary=movement_summary,
        total_frames=total_frames,
        processed_frames=processed_frames,
        frames_with_pose=frames_with_pose,
        fps=fps,
        width=width,
        height=height,
    )


def _create_pose_landmarker(mp: Any, config: PoseEstimationConfig) -> Any:
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(config.model_path),
        delegate=mp.tasks.BaseOptions.Delegate.CPU,
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=config.num_poses,
        min_pose_detection_confidence=config.min_pose_detection_confidence,
        min_pose_presence_confidence=config.min_pose_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _should_save_keyframe(
    *,
    source_frame_index: int,
    start_frame_index: int,
    selected_frame_count: int,
    keyframe_indices: set[int],
    keyframe_count: int,
    max_keyframes: int,
) -> bool:
    if max_keyframes <= 0:
        return False
    if selected_frame_count > 0:
        return (source_frame_index - start_frame_index) in keyframe_indices
    return keyframe_count < max_keyframes


def _frame_window(
    *,
    fps: float,
    total_frames: int,
    start_time_seconds: float | None,
    end_time_seconds: float | None,
) -> tuple[int, int]:
    start_frame_index = 0
    if start_time_seconds is not None:
        start_frame_index = max(0, int(start_time_seconds * fps))

    if total_frames > 0:
        end_frame_index = total_frames
    else:
        end_frame_index = 2**63 - 1

    if end_time_seconds is not None:
        end_frame_index = min(end_frame_index, max(0, int(end_time_seconds * fps)))

    if end_frame_index <= start_frame_index:
        raise ValueError("Selected video window contains no frames.")

    return start_frame_index, end_frame_index


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video IO. Install project dependencies with `make setup`."
        ) from exc
    return cv2


def _import_mediapipe() -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for pose detection. "
            "Install project dependencies with `make setup`."
        ) from exc
    return mp
