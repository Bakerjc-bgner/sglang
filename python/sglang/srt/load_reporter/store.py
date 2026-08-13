"""Atomic latest-snapshot store.

Temporary bridge during the push-channel refactor: still publishes merged
views for the runtime, but delegates single-pull validation to the stateless
validator in snapshot_validation.py.  The store itself is deleted once the
fire loop replaces it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence
from typing import Optional

from sglang.srt.load_reporter.snapshot_validation import (
    RankSnapshot,
    SnapshotValidationError,
    _require_finite_float,
    _require_non_negative_int64,
    validate_full_snapshot,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotView:
    ranks: tuple[RankSnapshot, ...]
    last_success_unix_ms: Optional[int]
    last_success_monotonic: Optional[float]
    last_error: Optional[str]

    @classmethod
    def empty(cls) -> SnapshotView:
        """Return the initial view before any successful sample."""
        return cls((), None, None, "no successful load sample")


class LatestSnapshotStore:
    def __init__(self) -> None:
        """Initialize the store with an unreachable empty view."""
        self._view = SnapshotView.empty()

    def view(self) -> SnapshotView:
        """Return the current immutable snapshot view."""
        # SnapshotView/RankSnapshot/tuple are immutable, so the same reference is safe.
        return self._view

    def apply_full_snapshot(
        self,
        loads: Sequence[LoadSnapshot],
        *,
        expected_dp_ranks: Collection[int],
        collected_at_unix_ms: int,
        collected_at_monotonic: float,
    ) -> SnapshotView:
        """Validate and atomically publish one authoritative full snapshot."""
        collected_at_unix_ms = _require_non_negative_int64(
            "collected_at_unix_ms", collected_at_unix_ms
        )
        collected_at_monotonic = _require_finite_float(
            "collected_at_monotonic", collected_at_monotonic
        )
        if collected_at_monotonic < 0:
            raise SnapshotValidationError(
                "collected_at_monotonic must be finite and non-negative"
            )

        candidates = validate_full_snapshot(
            loads,
            expected_dp_ranks=expected_dp_ranks,
            fallback_time_unix_ms=collected_at_unix_ms,
        )

        # Previous-rank merge: prevent an older sample from overwriting an
        # already-published value.  This is the behavior the push-channel
        # refactor deletes along with the store; until then it stays here.
        previous_by_rank = {rank.dp_rank: rank for rank in self._view.ranks}
        merged: list[RankSnapshot] = []
        for incoming in candidates:
            previous = previous_by_rank.get(incoming.dp_rank)
            if (
                previous is not None
                and previous.snapshot_time_unix_ms > incoming.snapshot_time_unix_ms
            ):
                merged.append(previous)
            else:
                # Equal timestamp: use raw metrics from this full sample.
                merged.append(incoming)

        new_view = SnapshotView(
            ranks=tuple(merged),
            last_success_unix_ms=collected_at_unix_ms,
            last_success_monotonic=collected_at_monotonic,
            last_error=None,
        )
        self._view = new_view
        return new_view

    def record_error(self, error: BaseException | str) -> SnapshotView:
        """Publish a sampling error while preserving the last good ranks."""
        message = str(error).strip()
        if not message:
            message = (
                type(error).__name__
                if isinstance(error, BaseException)
                else "unknown load sampling error"
            )
        current = self._view
        new_view = SnapshotView(
            ranks=current.ranks,
            last_success_unix_ms=current.last_success_unix_ms,
            last_success_monotonic=current.last_success_monotonic,
            last_error=message,
        )
        self._view = new_view
        return new_view
