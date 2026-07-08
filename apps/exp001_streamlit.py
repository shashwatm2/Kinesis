from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from kinesis.experiments.exp001.config import PoseEstimationConfig
from kinesis.experiments.exp001.movement_summary import movement_summary_lines
from kinesis.experiments.exp001.processor import process_video

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    st.set_page_config(page_title="Kinesis EXP-001", layout="wide")
    st.title("EXP-001: Dance Pose Estimation")

    uploaded_video = st.file_uploader(
        "Dance video",
        type=["mp4", "mov", "m4v", "avi"],
        accept_multiple_files=False,
    )
    model_path_input = Path(
        st.text_input("MediaPipe model path", value="models/pose_landmarker_full.task")
    )
    model_path = model_path_input if model_path_input.is_absolute() else ROOT / model_path_input
    max_keyframes = st.slider("Key frames", min_value=1, max_value=12, value=6)
    with st.expander("Analysis settings"):
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

    if uploaded_video is None:
        return

    if st.button("Process video", type="primary"):
        if not model_path.exists():
            st.error(
                "Model file not found. Download the Pose Landmarker model into models/ first."
            )
            return

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = ROOT / "outputs" / "exp001" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = output_dir / Path(uploaded_video.name).name
        input_path.write_bytes(uploaded_video.getbuffer())

        config = PoseEstimationConfig(
            model_path=model_path,
            max_keyframes=max_keyframes,
            metric_min_average_visibility=metric_min_average_visibility,
            smoothing_window_frames=smoothing_window_frames,
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
            st.subheader("Key Frames")
            columns = st.columns(min(3, len(summary.keyframe_paths)))
            for index, keyframe_path in enumerate(summary.keyframe_paths):
                with columns[index % len(columns)]:
                    st.image(str(keyframe_path), caption=keyframe_path.name)


if __name__ == "__main__":
    main()
