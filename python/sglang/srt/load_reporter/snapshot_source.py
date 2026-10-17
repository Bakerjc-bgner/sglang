"""Reporter-owned latest snapshots for one fixed publisher topology."""

from __future__ import annotations

import math

from sglang.srt.managers.load_snapshot import LoadSnapshot


class SnapshotCache:
    """One fixed-topology generation. Equal or older source time never renews it."""

    def __init__(self, dp_size: int):
        if dp_size <= 0:
            raise ValueError("dp_size must be positive")
        self.dp_size = dp_size
        self.latest_by_rank: dict[int, LoadSnapshot] = {}

    def update(self, loads) -> dict[int, LoadSnapshot]:
        changed = {}
        for load in loads:
            rank = load.dp_rank
            if not (0 <= rank < self.dp_size) or not (
                math.isfinite(load.timestamp) and load.timestamp > 0
            ):
                continue
            old = self.latest_by_rank.get(rank)
            if old is not None and load.timestamp <= old.timestamp:
                continue
            self.latest_by_rank[rank] = load
            changed[rank] = load
        return changed

    def snapshot(self) -> list[LoadSnapshot]:
        # Published snapshots are not mutated; copy the rank container only.
        return [self.latest_by_rank[r] for r in sorted(self.latest_by_rank)]
