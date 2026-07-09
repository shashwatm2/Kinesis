from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from kinesis.experiments.exp001.config import PoseEstimationConfig
from kinesis.experiments.exp001.model_assets import ensure_pose_landmarker_model
from kinesis.experiments.exp001.movement_summary import movement_summary_lines
from kinesis.experiments.exp001.processor import process_video
from kinesis.experiments.exp002.group_reference import compare_group_video_to_reference
from kinesis.experiments.exp002.matching import MatchConfig, compare_movement_csvs
from kinesis.experiments.exp002.pipeline import compare_movement_videos
from kinesis.ui.streamlit_theme import apply_kinesis_theme, render_brand_header

ROOT = Path(__file__).resolve().parents[1]
ICON_PATH = ROOT / "assets" / "kinesis-icon.png"
WORDMARK_PATH = ROOT / "assets" / "kinesis-wordmark.png"


def main() -> None:
    icon_bytes = ICON_PATH.read_bytes()
    st.set_page_config(
        page_title="Kinesis Movement Lab",
        page_icon=icon_bytes,
        layout="wide",
    )
    apply_kinesis_theme()
    render_brand_header(WORDMARK_PATH)
    st.title("Kinesis Movement Lab")

    exp001_tab, exp002_tab = st.tabs(["EXP-001 Pose Analysis", "EXP-002 Reference Match"])
    with exp001_tab:
        _render_exp001()
    with exp002_tab:
        _render_exp002()


def _render_exp001() -> None:
    st.header("EXP-001: Dance Pose Estimation")
    uploaded_video = st.file_uploader(
        "Dance video",
        type=["mp4", "mov", "m4v", "avi"],
        accept_multiple_files=False,
        key="exp001_dance_video",
    )
    model_column, keyframe_column = st.columns([2, 1])
    with model_column:
        model_path_input = Path(
            st.text_input("MediaPipe model path", value="models/pose_landmarker_full.task")
        )
        model_path = _resolve_model_path(model_path_input)
    with keyframe_column:
        max_keyframes = st.number_input(
            "Key frames",
            min_value=0,
            max_value=60,
            value=6,
            step=1,
            help="Representative frames to save. Use 0 to skip key-frame images.",
        )
    with st.expander("Analysis settings"):
        settings_left, settings_right = st.columns(2)
        with settings_left:
            metric_min_average_visibility = st.slider(
                "Minimum average landmark visibility",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )
            smoothing_window_frames = st.slider(
                "Smoothing window",
                min_value=1,
                max_value=15,
                value=5,
            )
        with settings_right:
            start_time_seconds, end_time_seconds = _render_trim_settings(
                label_prefix="EXP-001",
                key_prefix="exp001",
            )

    if uploaded_video is None:
        return

    if st.button("Process video", type="primary"):
        prepared_model_path = _prepare_pose_model(model_path)
        if prepared_model_path is None:
            return

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = ROOT / "outputs" / "exp001" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = output_dir / Path(uploaded_video.name).name
        input_path.write_bytes(uploaded_video.getbuffer())

        config = PoseEstimationConfig(
            model_path=prepared_model_path,
            max_keyframes=max_keyframes,
            metric_min_average_visibility=metric_min_average_visibility,
            smoothing_window_frames=smoothing_window_frames,
            start_time_seconds=start_time_seconds,
            end_time_seconds=end_time_seconds,
        )

        try:
            with st.spinner("Processing video"):
                summary = process_video(
                    input_path=input_path,
                    output_dir=output_dir,
                    config=config,
                )
        except Exception as exc:
            st.exception(exc)
            return

        st.subheader("Processed Video")
        st.video(summary.output_video_path.read_bytes())

        st.subheader("Run Summary")
        st.write(
            {
                "frames_processed": summary.processed_frames,
                "frames_with_pose": summary.frames_with_pose,
                "fps": round(summary.fps, 2),
                "output": str(summary.output_video_path),
                "movement_csv": str(summary.analysis_csv_path),
                "manifest": str(summary.manifest_path),
            }
        )

        st.subheader("Movement Summary")
        st.markdown(
            "\n".join(f"- {line}" for line in movement_summary_lines(summary.movement_summary))
        )
        st.download_button(
            "Download movement CSV",
            data=summary.analysis_csv_path.read_bytes(),
            file_name=summary.analysis_csv_path.name,
            mime="text/csv",
        )
        st.download_button(
            "Download run manifest",
            data=summary.manifest_path.read_bytes(),
            file_name=summary.manifest_path.name,
            mime="application/json",
        )

        if summary.plot_paths:
            st.subheader("Metric Plots")
            for plot_path in summary.plot_paths:
                st.image(str(plot_path), caption=plot_path.name)

        if summary.keyframe_paths:
            _render_keyframes("Key Frames", summary.keyframe_paths)


def _render_exp002() -> None:
    st.header("EXP-002: Reference Movement Matching")
    input_mode = st.radio(
        "Input type",
        ["Raw videos", "Group video", "Movement CSVs"],
        horizontal=True,
        key="exp002_input_type",
    )

    match_config = _render_match_settings()
    if input_mode == "Raw videos":
        _render_exp002_video_mode(match_config)
    elif input_mode == "Group video":
        _render_exp002_group_mode(match_config)
    else:
        _render_exp002_csv_mode(match_config)


def _render_match_settings() -> MatchConfig:
    with st.expander("Matching settings"):
        max_offset_seconds = st.number_input(
            "Maximum timing offset seconds",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )
        min_compared_frames = st.number_input(
            "Minimum compared frames",
            min_value=1,
            max_value=10_000,
            value=5,
            step=1,
        )
        largest_moment_count = st.number_input(
            "Difference moments",
            min_value=0,
            max_value=25,
            value=5,
            step=1,
        )
    return MatchConfig(
        max_offset_seconds=float(max_offset_seconds),
        min_compared_frames=int(min_compared_frames),
        largest_moment_count=int(largest_moment_count),
    )


def _render_exp002_video_mode(match_config: MatchConfig) -> None:
    reference_column, practice_column = st.columns(2)
    with reference_column:
        reference_video = st.file_uploader(
            "Reference video",
            type=["mp4", "mov", "m4v", "avi"],
            accept_multiple_files=False,
            key="exp002_reference_video",
        )
    with practice_column:
        practice_video = st.file_uploader(
            "Practice video",
            type=["mp4", "mov", "m4v", "avi"],
            accept_multiple_files=False,
            key="exp002_practice_video",
        )

    model_path_input = Path(
        st.text_input(
            "EXP-002 MediaPipe model path",
            value="models/pose_landmarker_full.task",
        )
    )
    model_path = _resolve_model_path(model_path_input)
    with st.expander("Pose processing settings"):
        settings_left, settings_right = st.columns(2)
        with settings_left:
            max_keyframes = st.number_input(
                "EXP-002 key frames per video",
                min_value=0,
                max_value=60,
                value=3,
                step=1,
            )
            metric_min_average_visibility = st.slider(
                "EXP-002 minimum average landmark visibility",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )
            smoothing_window_frames = st.slider(
                "EXP-002 smoothing window",
                min_value=1,
                max_value=15,
                value=5,
            )
        with settings_right:
            start_time_seconds, end_time_seconds = _render_trim_settings(
                label_prefix="EXP-002",
                key_prefix="exp002_raw",
            )

    if reference_video is None or practice_video is None:
        return

    if st.button("Compare videos", type="primary", key="exp002_compare_videos"):
        prepared_model_path = _prepare_pose_model(model_path)
        if prepared_model_path is None:
            return

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = ROOT / "outputs" / "exp002" / run_id
        input_dir = output_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        reference_video_path = _save_uploaded_file(
            reference_video,
            input_dir / f"reference_{Path(reference_video.name).name}",
        )
        practice_video_path = _save_uploaded_file(
            practice_video,
            input_dir / f"practice_{Path(practice_video.name).name}",
        )

        try:
            with st.spinner("Processing and comparing videos"):
                summary = compare_movement_videos(
                    reference_video_path=reference_video_path,
                    practice_video_path=practice_video_path,
                    output_dir=output_dir,
                    pose_config=PoseEstimationConfig(
                        model_path=prepared_model_path,
                        max_keyframes=int(max_keyframes),
                        metric_min_average_visibility=metric_min_average_visibility,
                        smoothing_window_frames=int(smoothing_window_frames),
                        start_time_seconds=start_time_seconds,
                        end_time_seconds=end_time_seconds,
                    ),
                    match_config=match_config,
                )
        except Exception as exc:
            st.exception(exc)
            return

        _render_match_summary(summary.movement_match)
        st.subheader("Processed Videos")
        reference_col, practice_col = st.columns(2)
        with reference_col:
            st.video(summary.reference_processing.output_video_path.read_bytes())
            st.download_button(
                "Download reference movement CSV",
                data=summary.reference_processing.analysis_csv_path.read_bytes(),
                file_name=summary.reference_processing.analysis_csv_path.name,
                mime="text/csv",
            )
        with practice_col:
            st.video(summary.practice_processing.output_video_path.read_bytes())
            st.download_button(
                "Download practice movement CSV",
                data=summary.practice_processing.analysis_csv_path.read_bytes(),
                file_name=summary.practice_processing.analysis_csv_path.name,
                mime="text/csv",
            )

        if summary.reference_processing.keyframe_paths:
            _render_keyframes(
                "Reference Key Frames",
                summary.reference_processing.keyframe_paths,
            )
        if summary.practice_processing.keyframe_paths:
            _render_keyframes(
                "Practice Key Frames",
                summary.practice_processing.keyframe_paths,
            )


def _render_exp002_group_mode(match_config: MatchConfig) -> None:
    group_video_column, model_column = st.columns([1, 1])
    with group_video_column:
        group_video = st.file_uploader(
            "Group video",
            type=["mp4", "mov", "m4v", "avi"],
            accept_multiple_files=False,
            key="exp002_group_video",
        )
    with model_column:
        model_path_input = Path(
            st.text_input(
                "Group MediaPipe model path",
                value="models/pose_landmarker_full.task",
            )
        )
        model_path = _resolve_model_path(model_path_input)
    with st.expander("Group pose processing settings"):
        identity_column, analysis_column, trim_column = st.columns(3)
        with identity_column:
            max_people = st.number_input(
                "Maximum people",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
            )
            reference_track_id = st.number_input(
                "Reference track ID",
                min_value=1,
                max_value=8,
                value=1,
                step=1,
            )
        with analysis_column:
            max_keyframes = st.number_input(
                "Group key frames",
                min_value=0,
                max_value=60,
                value=3,
                step=1,
            )
            metric_min_average_visibility = st.slider(
                "Group minimum average landmark visibility",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )
            smoothing_window_frames = st.slider(
                "Group smoothing window",
                min_value=1,
                max_value=15,
                value=5,
            )
        with trim_column:
            start_time_seconds, end_time_seconds = _render_trim_settings(
                label_prefix="Group",
                key_prefix="exp002_group",
            )

    if group_video is None:
        return

    if st.button("Compare group video", type="primary", key="exp002_compare_group_video"):
        prepared_model_path = _prepare_pose_model(model_path)
        if prepared_model_path is None:
            return

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = ROOT / "outputs" / "exp002" / run_id
        input_dir = output_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        group_video_path = _save_uploaded_file(
            group_video,
            input_dir / f"group_{Path(group_video.name).name}",
        )

        try:
            with st.spinner("Tracking and comparing people"):
                summary = compare_group_video_to_reference(
                    input_path=group_video_path,
                    output_dir=output_dir,
                    pose_config=PoseEstimationConfig(
                        model_path=prepared_model_path,
                        num_poses=int(max_people),
                        max_keyframes=int(max_keyframes),
                        metric_min_average_visibility=metric_min_average_visibility,
                        smoothing_window_frames=int(smoothing_window_frames),
                        start_time_seconds=start_time_seconds,
                        end_time_seconds=end_time_seconds,
                    ),
                    reference_track_id=int(reference_track_id),
                    match_config=match_config,
                )
        except Exception as exc:
            st.exception(exc)
            return

        _render_group_reference_summary(summary)


def _render_exp002_csv_mode(match_config: MatchConfig) -> None:
    reference_column, practice_column = st.columns(2)
    with reference_column:
        reference_csv = st.file_uploader(
            "Reference movement CSV",
            type=["csv"],
            accept_multiple_files=False,
            key="exp002_reference_csv",
        )
    with practice_column:
        practice_csv = st.file_uploader(
            "Practice movement CSV",
            type=["csv"],
            accept_multiple_files=False,
            key="exp002_practice_csv",
        )

    if reference_csv is None or practice_csv is None:
        return

    if st.button("Compare movement", type="primary", key="exp002_compare_movement"):
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = ROOT / "outputs" / "exp002" / run_id
        input_dir = output_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        reference_csv_path = _save_uploaded_file(
            reference_csv,
            input_dir / f"reference_{Path(reference_csv.name).name}",
        )
        practice_csv_path = _save_uploaded_file(
            practice_csv,
            input_dir / f"practice_{Path(practice_csv.name).name}",
        )

        try:
            with st.spinner("Comparing movement"):
                summary = compare_movement_csvs(
                    reference_csv_path=reference_csv_path,
                    practice_csv_path=practice_csv_path,
                    output_dir=output_dir,
                    config=match_config,
                )
        except Exception as exc:
            st.exception(exc)
            return

        _render_match_summary(summary)


def _render_group_reference_summary(summary: Any) -> None:
    st.subheader("Group Summary")
    track_col, reference_col, frames_col, multi_col = st.columns(4)
    track_col.metric("Tracks detected", str(len(summary.track_summaries)))
    reference_col.metric("Reference track", str(summary.reference_track_id))
    frames_col.metric("Processed frames", str(summary.processed_frames))
    multi_col.metric("Multi-person frames", str(summary.frames_with_multiple_poses))

    st.subheader("Tracked Video")
    st.video(summary.output_video_path.read_bytes())

    if summary.keyframe_paths:
        _render_keyframes("Tracked Key Frames", summary.keyframe_paths)

    if summary.track_summaries:
        st.subheader("Tracks")
        st.dataframe(
            [
                {
                    "track_id": track.track_id,
                    "frames": track.frame_count,
                    "first_time": f"{track.first_timestamp_seconds:.2f}s",
                    "last_time": f"{track.last_timestamp_seconds:.2f}s",
                }
                for track in summary.track_summaries
            ],
            hide_index=True,
            width="stretch",
        )

    if summary.comparisons:
        st.subheader("Reference Comparisons")
        st.dataframe(
            [_group_comparison_row(comparison) for comparison in summary.comparisons],
            hide_index=True,
            width="stretch",
        )

    st.download_button(
        "Download group summary JSON",
        data=summary.manifest_path.read_bytes(),
        file_name=summary.manifest_path.name,
        mime="application/json",
    )
    for track in summary.track_summaries:
        st.download_button(
            f"Download track {track.track_id} movement CSV",
            data=track.movement_csv_path.read_bytes(),
            file_name=track.movement_csv_path.name,
            mime="text/csv",
        )


def _group_comparison_row(comparison: Any) -> dict[str, str | int]:
    if comparison.match_summary is None:
        return {
            "track_id": comparison.track_id,
            "score": "",
            "timing": "",
            "compared_frames": "",
            "largest_metric": "",
            "status": comparison.error or "not compared",
        }

    largest_metric = ""
    if comparison.match_summary.largest_difference_moments:
        largest_metric = _format_metric_name(
            comparison.match_summary.largest_difference_moments[0].largest_difference_metric
        )

    return {
        "track_id": comparison.track_id,
        "score": f"{comparison.match_summary.overall_score:.1f}",
        "timing": _format_timing_offset(comparison.match_summary.best_time_offset_seconds),
        "compared_frames": str(comparison.match_summary.compared_frame_count),
        "largest_metric": largest_metric,
        "status": "compared",
    }


def _render_match_summary(summary: Any) -> None:
    st.subheader("Match Summary")
    score_col, timing_col, frames_col = st.columns(3)
    score_col.metric("Overall match score", f"{summary.overall_score:.1f}")
    timing_col.metric(
        "Practice timing",
        _format_timing_offset(summary.best_time_offset_seconds),
    )
    frames_col.metric("Compared frames", str(summary.compared_frame_count))

    if summary.largest_difference_moments:
        st.subheader("Largest Difference Moments")
        st.dataframe(
            [
                {
                    "reference_time": f"{moment.reference_timestamp_seconds:.2f}s",
                    "practice_time": f"{moment.practice_timestamp_seconds:.2f}s",
                    "frame_score": f"{moment.match_score:.1f}",
                    "largest_metric": _format_metric_name(moment.largest_difference_metric),
                    "metric_difference": _format_metric_difference(
                        moment.largest_difference_metric,
                        moment.metric_differences,
                    ),
                }
                for moment in summary.largest_difference_moments
            ],
            hide_index=True,
            width="stretch",
        )

    if summary.report_csv_path.exists():
        st.download_button(
            "Download match report CSV",
            data=summary.report_csv_path.read_bytes(),
            file_name=summary.report_csv_path.name,
            mime="text/csv",
        )
        st.download_button(
            "Download match summary JSON",
            data=summary.summary_json_path.read_bytes(),
            file_name=summary.summary_json_path.name,
            mime="application/json",
        )


def _save_uploaded_file(uploaded_file: Any, destination: Path) -> Path:
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def _resolve_model_path(model_path: Path) -> Path:
    if model_path.is_absolute():
        return model_path
    return ROOT / model_path


def _prepare_pose_model(model_path: Path) -> Path | None:
    try:
        with st.spinner("Preparing MediaPipe pose model"):
            return ensure_pose_landmarker_model(model_path)
    except Exception as exc:
        st.error(
            "Could not prepare the MediaPipe pose model. "
            "Check the model path or network access, then try again."
        )
        st.exception(exc)
        return None


def _render_keyframes(title: str, keyframe_paths: list[Path]) -> None:
    st.subheader(title)
    columns = st.columns(min(3, len(keyframe_paths)))
    for index, keyframe_path in enumerate(keyframe_paths):
        with columns[index % len(columns)]:
            st.image(str(keyframe_path), caption=keyframe_path.name)


def _render_trim_settings(
    *,
    label_prefix: str,
    key_prefix: str,
) -> tuple[float | None, float | None]:
    start_time_seconds = st.number_input(
        f"{label_prefix} start time seconds",
        min_value=0.0,
        value=0.0,
        step=0.5,
        key=f"{key_prefix}_start_time_seconds",
    )
    end_time_seconds = st.number_input(
        f"{label_prefix} end time seconds",
        min_value=0.0,
        value=0.0,
        step=0.5,
        help="Use 0 to continue through the end of the video.",
        key=f"{key_prefix}_end_time_seconds",
    )
    return (
        _optional_time_seconds(start_time_seconds),
        _optional_time_seconds(end_time_seconds),
    )


def _optional_time_seconds(value: float) -> float | None:
    if value <= 0:
        return None
    return value


def _format_timing_offset(offset_seconds: float) -> str:
    if abs(offset_seconds) < 0.005:
        return "aligned"
    direction = "later" if offset_seconds > 0 else "earlier"
    return f"{abs(offset_seconds):.2f}s {direction}"


def _format_metric_name(metric_name: str | None) -> str:
    if metric_name is None:
        return ""
    return metric_name.replace("_", " ")


def _format_metric_difference(
    metric_name: str | None,
    metric_differences: dict[str, float],
) -> str:
    if metric_name is None:
        return ""
    value = metric_differences.get(metric_name)
    if value is None:
        return ""
    if metric_name.endswith("_degrees"):
        return f"{value:.1f} deg"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
