"""A reporter-loop broadcast clock; no polling clock is needed for PUB input."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sglang.srt.load_reporter.config import validate_session_timing

MAX_SESSIONS = 128


@dataclass
class _Timing:
    lease_ms: int
    expires_at: float


class _PeriodSchedule:
    """Fixed periods from an epoch; late ticks skip missed periods."""

    def __init__(self, interval_ms, now):
        self.interval_ns = interval_ms * 1_000_000
        self.epoch_ns = round(now * 1_000_000_000)
        self.advance(now)

    def advance(self, now):
        now_ns = round(now * 1_000_000_000)
        periods = max(0, now_ns - self.epoch_ns) // self.interval_ns
        self.next_deadline = (
            self.epoch_ns + (periods + 1) * self.interval_ns
        ) / 1_000_000_000


class ReportSchedule:
    """The first active session sets the clock; no sessions means no timer."""

    def __init__(self):
        self._period = None
        self._sessions: dict[int, _Timing] = {}

    def add(self, token, interval_ms, lease_ms, *, now):
        validate_session_timing(interval_ms, lease_ms)
        if len(self._sessions) >= MAX_SESSIONS:
            raise RuntimeError("load reporter session limit reached")
        if self._first_active(now) is None:
            self._period = _PeriodSchedule(interval_ms, now)
        self._sessions[token] = _Timing(lease_ms, now + lease_ms / 1000)

    def update(self, token, interval_ms=None, lease_ms=None, *, now):
        validate_session_timing(interval_ms, lease_ms)
        timing = self._sessions.get(token)
        if timing is None or timing.expires_at <= now:
            return False
        if (
            interval_ms is not None
            and token == self._first_active(now)
            and interval_ms * 1_000_000 != self._period.interval_ns
        ):
            self._period = _PeriodSchedule(interval_ms, now)
        if lease_ms is not None:
            timing.lease_ms = lease_ms
        timing.expires_at = now + timing.lease_ms / 1000
        return True

    def remove(self, token):
        self._sessions.pop(token, None)
        if not self._sessions:
            self._period = None

    def _first_active(self, now):
        return next((k for k, s in self._sessions.items() if s.expires_at > now), None)

    def next_deadline(self):
        return min(
            [self._period.next_deadline if self._period else math.inf]
            + [s.expires_at for s in self._sessions.values()]
        )

    def take_due(self, now):
        expired = tuple(k for k, s in self._sessions.items() if s.expires_at <= now)
        for token in expired:
            del self._sessions[token]
        if not self._sessions:
            self._period = None
        if self._period is None or self._period.next_deadline > now:
            return False, (), expired
        self._period.advance(now)
        return True, tuple(self._sessions), expired
