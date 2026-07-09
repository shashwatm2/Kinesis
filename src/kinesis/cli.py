from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kinesis.experiments.exp001.config import PoseEstimationConfig
from kinesis.experiments.exp001.movement_summary import movement_summary_lines
from kinesis.experiments.exp001.processor import process_video
from kinesis.experiments.exp002.group_reference import compare_group_video_to_reference
from kinesis.experiments.exp002.matching import (
    MatchConfig,
    MovementMatchSummary,
    compare_movement_csvs,
)
from kinesis.experiments.exp002.pipeline import compare_movement_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinesis",
        description="Kinesis experiment tooling.",
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    exp001_parser = subparsers.add_parser("exp001", help="Dance pose estimation.")
    exp001_subparsers = exp001_parser.add_subparsers(dest="command", required=True)

    process_parser = exp001_subparsers.add_parser("process", help="Process a dance video.")
    process_parser.add_argument("--input", required=True, type=Path, help="Input video path.")
    process_parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="MediaPipe Pose Landmarker .task model path.",
    )
    process_parser.add_argument(
        "--output-dir",
        default=Path("outputs/exp001"),
        type=Path,
        help="Directory for processed video and key frames.",
    )
    process_parser.add_argument(
        "--max-keyframes",
        default=6,
        type=int,
        help="Maximum number of annotated key frames to save.",
    )
    process_parser.add_argument(
        "--frame-stride",
        default=1,
        type=int,
        help="Process every Nth frame. Keep this at 1 for normal output video.",
    )
    process_parser.add_argument(
        "--min-detection-confidence",
        default=0.5,
        type=float,
        help="Minimum pose detection confidence.",
    )
    process_parser.add_argument(
        "--min-presence-confidence",
        default=0.5,
        type=float,
        help="Minimum pose presence confidence.",
    )
    process_parser.add_argument(
        "--min-tracking-confidence",
        default=0.5,
        type=float,
        help="Minimum pose tracking confidence.",
    )
    process_parser.add_argument(
        "--metric-min-average-visibility",
        default=0.5,
        type=float,
        help="Minimum average landmark visibility for metric quality checks.",
    )
    process_parser.add_argument(
        "--smoothing-window",
        default=5,
        type=int,
        help="Moving-average window size for smoothed metric values.",
    )
    process_parser.add_argument(
        "--start-time-seconds",
        default=None,
        type=float,
        help="Start processing at this timestamp. Defaults to the beginning.",
    )
    process_parser.add_argument(
        "--end-time-seconds",
        default=None,
        type=float,
        help="Stop processing at this timestamp. Defaults to the end.",
    )

    exp002_parser = subparsers.add_parser("exp002", help="Reference movement matching.")
    exp002_subparsers = exp002_parser.add_subparsers(dest="command", required=True)

    compare_parser = exp002_subparsers.add_parser(
        "compare",
        help="Compare a practice movement CSV against a reference movement CSV.",
    )
    compare_parser.add_argument(
        "--reference-csv",
        required=True,
        type=Path,
        help="EXP-001 movement CSV for the reference movement.",
    )
    compare_parser.add_argument(
        "--practice-csv",
        required=True,
        type=Path,
        help="EXP-001 movement CSV for the practice movement.",
    )
    compare_parser.add_argument(
        "--output-dir",
        default=Path("outputs/exp002"),
        type=Path,
        help="Directory for match report CSV and summary JSON.",
    )
    compare_parser.add_argument(
        "--max-offset-seconds",
        default=1.0,
        type=float,
        help="Maximum early/late timing offset to search in either direction.",
    )
    compare_parser.add_argument(
        "--offset-step-seconds",
        default=None,
        type=float,
        help="Timing offset search step. Defaults to the estimated frame step.",
    )
    compare_parser.add_argument(
        "--timestamp-tolerance-seconds",
        default=None,
        type=float,
        help="Nearest-frame timestamp tolerance. Defaults to the estimated frame step.",
    )
    compare_parser.add_argument(
        "--min-compared-frames",
        default=5,
        type=int,
        help="Minimum number of matched frames required to produce a score.",
    )
    compare_parser.add_argument(
        "--largest-moment-count",
        default=5,
        type=int,
        help="Number of largest difference moments to include in the summary.",
    )

    compare_videos_parser = exp002_subparsers.add_parser(
        "compare-videos",
        help="Process and compare a practice video against a reference video.",
    )
    compare_videos_parser.add_argument(
        "--reference-video",
        required=True,
        type=Path,
        help="Reference movement video.",
    )
    compare_videos_parser.add_argument(
        "--practice-video",
        required=True,
        type=Path,
        help="Practice movement video.",
    )
    compare_videos_parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="MediaPipe Pose Landmarker .task model path.",
    )
    compare_videos_parser.add_argument(
        "--output-dir",
        default=Path("outputs/exp002"),
        type=Path,
        help="Directory for processed videos, movement CSVs, and match reports.",
    )
    compare_videos_parser.add_argument(
        "--max-keyframes",
        default=6,
        type=int,
        help="Maximum number of annotated key frames to save per video.",
    )
    compare_videos_parser.add_argument(
        "--metric-min-average-visibility",
        default=0.5,
        type=float,
        help="Minimum average landmark visibility for metric quality checks.",
    )
    compare_videos_parser.add_argument(
        "--smoothing-window",
        default=5,
        type=int,
        help="Moving-average window size for smoothed metric values.",
    )
    compare_videos_parser.add_argument(
        "--start-time-seconds",
        default=None,
        type=float,
        help="Start processing both videos at this timestamp. Defaults to the beginning.",
    )
    compare_videos_parser.add_argument(
        "--end-time-seconds",
        default=None,
        type=float,
        help="Stop processing both videos at this timestamp. Defaults to the end.",
    )
    compare_videos_parser.add_argument(
        "--max-offset-seconds",
        default=1.0,
        type=float,
        help="Maximum early/late timing offset to search in either direction.",
    )
    compare_videos_parser.add_argument(
        "--offset-step-seconds",
        default=None,
        type=float,
        help="Timing offset search step. Defaults to the estimated frame step.",
    )
    compare_videos_parser.add_argument(
        "--timestamp-tolerance-seconds",
        default=None,
        type=float,
        help="Nearest-frame timestamp tolerance. Defaults to the estimated frame step.",
    )
    compare_videos_parser.add_argument(
        "--min-compared-frames",
        default=5,
        type=int,
        help="Minimum number of matched frames required to produce a score.",
    )
    compare_videos_parser.add_argument(
        "--largest-moment-count",
        default=5,
        type=int,
        help="Number of largest difference moments to include in the summary.",
    )

    compare_group_parser = exp002_subparsers.add_parser(
        "compare-group-video",
        help="Track people in one video and compare each track to a reference track.",
    )
    compare_group_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Group movement video.",
    )
    compare_group_parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="MediaPipe Pose Landmarker .task model path.",
    )
    compare_group_parser.add_argument(
        "--reference-track-id",
        default=1,
        type=int,
        help="Track ID to use as the reference person.",
    )
    compare_group_parser.add_argument(
        "--max-people",
        default=4,
        type=int,
        help="Maximum number of people to detect in each frame.",
    )
    compare_group_parser.add_argument(
        "--output-dir",
        default=Path("outputs/exp002"),
        type=Path,
        help="Directory for tracked video, track CSVs, and match reports.",
    )
    compare_group_parser.add_argument(
        "--max-keyframes",
        default=6,
        type=int,
        help="Maximum number of annotated key frames to save.",
    )
    compare_group_parser.add_argument(
        "--metric-min-average-visibility",
        default=0.5,
        type=float,
        help="Minimum average landmark visibility for metric quality checks.",
    )
    compare_group_parser.add_argument(
        "--smoothing-window",
        default=5,
        type=int,
        help="Moving-average window size for smoothed metric values.",
    )
    compare_group_parser.add_argument(
        "--start-time-seconds",
        default=None,
        type=float,
        help="Start tracking at this timestamp. Defaults to the beginning.",
    )
    compare_group_parser.add_argument(
        "--end-time-seconds",
        default=None,
        type=float,
        help="Stop tracking at this timestamp. Defaults to the end.",
    )
    compare_group_parser.add_argument(
        "--max-offset-seconds",
        default=1.0,
        type=float,
        help="Maximum early/late timing offset to search in either direction.",
    )
    compare_group_parser.add_argument(
        "--offset-step-seconds",
        default=None,
        type=float,
        help="Timing offset search step. Defaults to the estimated frame step.",
    )
    compare_group_parser.add_argument(
        "--timestamp-tolerance-seconds",
        default=None,
        type=float,
        help="Nearest-frame timestamp tolerance. Defaults to the estimated frame step.",
    )
    compare_group_parser.add_argument(
        "--min-compared-frames",
        default=5,
        type=int,
        help="Minimum number of matched frames required to produce a score.",
    )
    compare_group_parser.add_argument(
        "--largest-moment-count",
        default=5,
        type=int,
        help="Number of largest difference moments to include in the summary.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.experiment == "exp001" and args.command == "process":
        config = PoseEstimationConfig(
            model_path=args.model,
            max_keyframes=args.max_keyframes,
            frame_stride=args.frame_stride,
            min_pose_detection_confidence=args.min_detection_confidence,
            min_pose_presence_confidence=args.min_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            metric_min_average_visibility=args.metric_min_average_visibility,
            smoothing_window_frames=args.smoothing_window,
            start_time_seconds=args.start_time_seconds,
            end_time_seconds=args.end_time_seconds,
        )
        summary = process_video(
            input_path=args.input,
            output_dir=args.output_dir,
            config=config,
        )
        print(f"Processed frames: {summary.processed_frames}")
        print(f"Frames with pose: {summary.frames_with_pose}")
        print(f"Output video: {summary.output_video_path}")
        print(f"Movement CSV: {summary.analysis_csv_path}")
        print(f"Run manifest: {summary.manifest_path}")
        print("Movement summary:")
        for line in movement_summary_lines(summary.movement_summary):
            print(f"  - {line}")
        if summary.plot_paths:
            print("Metric plots:")
            for path in summary.plot_paths:
                print(f"  {path}")
        if summary.keyframe_paths:
            print("Key frames:")
            for path in summary.keyframe_paths:
                print(f"  {path}")
        return 0

    if args.experiment == "exp002" and args.command == "compare":
        summary = compare_movement_csvs(
            reference_csv_path=args.reference_csv,
            practice_csv_path=args.practice_csv,
            output_dir=args.output_dir,
            config=MatchConfig(
                max_offset_seconds=args.max_offset_seconds,
                offset_step_seconds=args.offset_step_seconds,
                timestamp_tolerance_seconds=args.timestamp_tolerance_seconds,
                min_compared_frames=args.min_compared_frames,
                largest_moment_count=args.largest_moment_count,
            ),
        )
        _print_match_summary(summary)
        return 0

    if args.experiment == "exp002" and args.command == "compare-videos":
        summary = compare_movement_videos(
            reference_video_path=args.reference_video,
            practice_video_path=args.practice_video,
            output_dir=args.output_dir,
            pose_config=PoseEstimationConfig(
                model_path=args.model,
                max_keyframes=args.max_keyframes,
                metric_min_average_visibility=args.metric_min_average_visibility,
                smoothing_window_frames=args.smoothing_window,
                start_time_seconds=args.start_time_seconds,
                end_time_seconds=args.end_time_seconds,
            ),
            match_config=MatchConfig(
                max_offset_seconds=args.max_offset_seconds,
                offset_step_seconds=args.offset_step_seconds,
                timestamp_tolerance_seconds=args.timestamp_tolerance_seconds,
                min_compared_frames=args.min_compared_frames,
                largest_moment_count=args.largest_moment_count,
            ),
        )
        print(f"Reference processed video: {summary.reference_processing.output_video_path}")
        print(f"Reference movement CSV: {summary.reference_processing.analysis_csv_path}")
        print(f"Practice processed video: {summary.practice_processing.output_video_path}")
        print(f"Practice movement CSV: {summary.practice_processing.analysis_csv_path}")
        _print_match_summary(summary.movement_match)
        return 0

    if args.experiment == "exp002" and args.command == "compare-group-video":
        summary = compare_group_video_to_reference(
            input_path=args.input,
            output_dir=args.output_dir,
            pose_config=PoseEstimationConfig(
                model_path=args.model,
                num_poses=args.max_people,
                max_keyframes=args.max_keyframes,
                metric_min_average_visibility=args.metric_min_average_visibility,
                smoothing_window_frames=args.smoothing_window,
                start_time_seconds=args.start_time_seconds,
                end_time_seconds=args.end_time_seconds,
            ),
            reference_track_id=args.reference_track_id,
            match_config=MatchConfig(
                max_offset_seconds=args.max_offset_seconds,
                offset_step_seconds=args.offset_step_seconds,
                timestamp_tolerance_seconds=args.timestamp_tolerance_seconds,
                min_compared_frames=args.min_compared_frames,
                largest_moment_count=args.largest_moment_count,
            ),
        )
        print(f"Tracked video: {summary.output_video_path}")
        print(f"Group summary JSON: {summary.manifest_path}")
        print(f"Reference track ID: {summary.reference_track_id}")
        print("Tracks:")
        for track in summary.track_summaries:
            print(
                f"  - track {track.track_id}: {track.frame_count} frames, "
                f"CSV {track.movement_csv_path}"
            )
        if summary.comparisons:
            print("Reference comparisons:")
            for comparison in summary.comparisons:
                if comparison.match_summary is None:
                    print(f"  - track {comparison.track_id}: {comparison.error}")
                else:
                    print(
                        f"  - track {comparison.track_id}: "
                        f"score {comparison.match_summary.overall_score:.1f}, "
                        f"offset {comparison.match_summary.best_time_offset_seconds:.3f}s"
                    )
        return 0

    parser.error("Unsupported command")
    return 2


def _print_match_summary(summary: MovementMatchSummary) -> None:
    print(f"Overall match score: {summary.overall_score:.1f}")
    print(f"Best time offset seconds: {summary.best_time_offset_seconds:.3f}")
    print(f"Compared frames: {summary.compared_frame_count}")
    print(f"Report CSV: {summary.report_csv_path}")
    print(f"Summary JSON: {summary.summary_json_path}")
    if summary.largest_difference_moments:
        print("Largest difference moments:")
        for moment in summary.largest_difference_moments:
            print(
                "  - "
                f"reference {moment.reference_timestamp_seconds:.2f}s, "
                f"practice {moment.practice_timestamp_seconds:.2f}s, "
                f"score {moment.match_score:.1f}, "
                f"largest metric {moment.largest_difference_metric}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
