from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float | None = None
    presence: float | None = None
