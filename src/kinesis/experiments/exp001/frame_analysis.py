from __future__ import annotations

from dataclasses import dataclass

from kinesis.experiments.exp001.metrics import FrameMetrics
from kinesis.experiments.exp001.quality import MetricQualityResult


@dataclass(frozen=True)
class FrameAnalysis:
    frame_index: int
    timestamp_ms: int
    raw_metrics: FrameMetrics
    smoothed_metrics: FrameMetrics
    metric_quality: dict[str, MetricQualityResult]

    @property
    def timestamp_seconds(self) -> float:
        return self.timestamp_ms / 1000
