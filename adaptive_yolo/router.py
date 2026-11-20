"""
Graded routing policy: trajectory reading → exit level.

    distance < full_dist  AND  closing faster than closing_rate   → FULL   (P3+P4+P5)
    distance < mid_dist                                            → P3_P4
    otherwise                                                      → P3

Rationale: the distance is between the two tracked vessels, not between the
camera and a target. The camera ASV follows both at the same speed, so their
apparent size stays roughly constant; the distance is a collision-risk signal.
Far apart → low risk → cheapest level; closer → more of the network; close and
approaching (the safety-critical case) → everything.
With mid_dist=None there is no middle level: this is the paper's binary rule
(FULL when close and approaching, otherwise P3).

`hold_frames` adds hysteresis: after escalating, the router keeps the higher
level for at least that many frames before stepping down, which avoids
flicker when the reading hovers around a threshold. Missing readings route to
`fallback` (FULL by default — fail safe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import ExitLevel
from .trajectory import ProximityReading


@dataclass
class GradedRouter:
    full_dist: float = 30.0     # m
    mid_dist: float | None = 60.0   # m; None = binary P3/FULL (paper rule)
    closing_rate: float = 0.5   # m/s; FULL requires closure_rate < -closing_rate
    hold_frames: int = 0
    fallback: ExitLevel = ExitLevel.FULL
    _level: ExitLevel | None = field(default=None, init=False, repr=False)
    _held: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if self.mid_dist is not None and self.mid_dist < self.full_dist:
            raise ValueError("mid_dist must be >= full_dist (or None for the binary rule)")

    def rule(self, reading: ProximityReading | None) -> ExitLevel:
        """Stateless decision for one reading."""
        if reading is None:
            return self.fallback
        if reading.distance_m < self.full_dist and reading.closure_rate_ms < -self.closing_rate:
            return ExitLevel.FULL
        if self.mid_dist is not None and reading.distance_m < self.mid_dist:
            return ExitLevel.P3_P4
        return ExitLevel.P3

    def __call__(self, reading: ProximityReading | None) -> ExitLevel:
        """Decision with hysteresis (call once per frame)."""
        want = self.rule(reading)
        if self._level is None or want >= self._level or self._held >= self.hold_frames:
            if self._level is None or want != self._level:
                self._held = 0
            self._level = want
        self._held += 1
        return self._level

    def reset(self) -> None:
        self._level, self._held = None, 0
