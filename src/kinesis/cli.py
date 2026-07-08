from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kinesis.experiments.exp001.config import PoseEstimationConfig
from kinesis.experiments.exp001.movement_summary import movement_summary_lines
from kinesis.experiments.exp001.processor import process_video


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

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
