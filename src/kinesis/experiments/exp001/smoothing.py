from __future__ import annotations

from collections import deque

from kinesis.experiments.exp001.metrics import METRIC_COLUMNS, FrameMetrics
from kinesis.experiments.exp001.quality import MetricQualityResult

SMOOTHED_METRIC_COLUMNS: tuple[str, ...] = tuple(
    f"smoothed_{metric_name}" for metric_name in METRIC_COLUMNS
)


class MetricSmoother:
    def __init__(self, *, window_size: int) -> None:
        self._window_size = window_size
        self._histories: dict[str, deque[float]] = {
            metric_name: deque(maxlen=window_size) for metric_name in METRIC_COLUMNS
        }

    def smooth(
        self,
        *,
        metrics: FrameMetrics,
        quality: dict[str, MetricQualityResult],
    ) -> FrameMetrics:
        smoothed_values: dict[str, float | None] = {}

        for metric_name in METRIC_COLUMNS:
            value = getattr(metrics, metric_name)
            metric_quality = quality.get(metric_name)
            history = self._histories[metric_name]

            if value is not None and metric_quality is not None and metric_quality.usable:
                history.append(value)
                smoothed_values[metric_name] = sum(history) / len(history)
            else:
                smoothed_values[metric_name] = None

        return FrameMetrics(**smoothed_values)


def smoothed_metric_values(metrics: FrameMetrics) -> dict[str, float | None]:
    return {
        f"smoothed_{metric_name}": getattr(metrics, metric_name) for metric_name in METRIC_COLUMNS
    }
