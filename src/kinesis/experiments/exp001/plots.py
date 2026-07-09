from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from kinesis.experiments.exp001.frame_analysis import FrameAnalysis

PlotSeries = tuple[str, str]

PLOT_SPECS: tuple[tuple[str, str, str, tuple[PlotSeries, ...]], ...] = (
    (
        "plot_torso_lean.png",
        "Torso Lean",
        "degrees",
        (("torso_lean_angle_degrees", "torso lean"),),
    ),
    (
        "plot_knee_angles.png",
        "Knee Angles",
        "degrees",
        (
            ("left_knee_angle_degrees", "left knee"),
            ("right_knee_angle_degrees", "right knee"),
        ),
    ),
    (
        "plot_elbow_angles.png",
        "Elbow Angles",
        "degrees",
        (
            ("left_elbow_angle_degrees", "left elbow"),
            ("right_elbow_angle_degrees", "right elbow"),
        ),
    ),
    (
        "plot_center_visibility.png",
        "Center Position And Visibility",
        "normalized value",
        (
            ("center_of_body_x", "center x"),
            ("average_landmark_visibility", "visibility"),
        ),
    ),
)


def create_metric_plots(
    frame_analyses: Sequence[FrameAnalysis],
    *,
    output_dir: Path,
) -> list[Path]:
    if not frame_analyses:
        return []

    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: list[Path] = []
    for file_name, title, ylabel, series in PLOT_SPECS:
        if not _has_any_smoothed_values(frame_analyses, series):
            continue

        figure, axes = plt.subplots(figsize=(10, 4.5))
        for metric_name, label in series:
            times, raw_values, smoothed_values = _series_values(frame_analyses, metric_name)
            if not smoothed_values:
                continue

            axes.plot(
                times,
                raw_values,
                color="#a7b0ba",
                linewidth=0.8,
                alpha=0.35,
                label=f"{label} raw",
            )
            axes.plot(
                times,
                smoothed_values,
                linewidth=2.0,
                label=f"{label} smoothed",
            )

        axes.set_title(title)
        axes.set_xlabel("time (seconds)")
        axes.set_ylabel(ylabel)
        axes.grid(True, alpha=0.25)
        axes.legend(loc="best")
        figure.tight_layout()

        plot_path = output_dir / file_name
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)
        plot_paths.append(plot_path)

    return plot_paths


def _has_any_smoothed_values(
    frame_analyses: Sequence[FrameAnalysis],
    series: tuple[PlotSeries, ...],
) -> bool:
    for metric_name, _label in series:
        for analysis in frame_analyses:
            if getattr(analysis.smoothed_metrics, metric_name) is not None:
                return True
    return False


def _series_values(
    frame_analyses: Sequence[FrameAnalysis],
    metric_name: str,
) -> tuple[list[float], list[float | None], list[float | None]]:
    times: list[float] = []
    raw_values: list[float | None] = []
    smoothed_values: list[float | None] = []

    for analysis in frame_analyses:
        raw_value = getattr(analysis.raw_metrics, metric_name)
        smoothed_value = getattr(analysis.smoothed_metrics, metric_name)
        if raw_value is None and smoothed_value is None:
            continue

        times.append(analysis.timestamp_seconds)
        raw_values.append(raw_value)
        smoothed_values.append(smoothed_value)

    return times, raw_values, smoothed_values
