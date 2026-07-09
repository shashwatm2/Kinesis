from __future__ import annotations

import pytest

from kinesis.experiments.exp001.processor import _frame_window


def test_frame_window_converts_seconds_to_frame_bounds() -> None:
    assert _frame_window(
        fps=30.0,
        total_frames=300,
        start_time_seconds=2.0,
        end_time_seconds=5.0,
    ) == (60, 150)


def test_frame_window_defaults_to_full_video() -> None:
    assert _frame_window(
        fps=30.0,
        total_frames=300,
        start_time_seconds=None,
        end_time_seconds=None,
    ) == (0, 300)


def test_frame_window_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="no frames"):
        _frame_window(
            fps=30.0,
            total_frames=300,
            start_time_seconds=4.0,
            end_time_seconds=4.0,
        )
