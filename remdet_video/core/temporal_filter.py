"""Lightweight temporal presence rules for video-level target decisions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalRule:
    name: str
    window: int
    required: int
    exit_misses: int

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError('window must be at least 1')
        if not 1 <= self.required <= self.window:
            raise ValueError('required must be between 1 and window')
        if self.exit_misses < 1:
            raise ValueError('exit_misses must be at least 1')


DEFAULT_TEMPORAL_RULES = (
    TemporalRule('T0_1of1_exit1', window=1, required=1, exit_misses=1),
    TemporalRule('T1_2of3_exit2', window=3, required=2, exit_misses=2),
    TemporalRule('T2_3of5_exit3', window=5, required=3, exit_misses=3),
    TemporalRule('T3_2of5_exit5', window=5, required=2, exit_misses=5),
)


def apply_temporal_rule(
    raw_presence: list[bool],
    rule: TemporalRule,
) -> list[bool]:
    """Convert raw frame presence into a stable temporal state sequence."""
    history: deque[bool] = deque(maxlen=rule.window)
    active = False
    misses = 0
    output: list[bool] = []

    for present in raw_presence:
        history.append(bool(present))
        if active:
            if present:
                misses = 0
            else:
                misses += 1
                if misses >= rule.exit_misses:
                    active = False
                    misses = 0
        elif sum(history) >= rule.required:
            active = True
            misses = 0
        output.append(active)

    return output
