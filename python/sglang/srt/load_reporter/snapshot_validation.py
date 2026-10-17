"""Map decoded SGLang snapshots to a complete, semantically valid rank report."""

from __future__ import annotations

import math
from collections.abc import Sequence

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.managers.load_snapshot import LoadSnapshot

_COUNT_FIELDS = (
    "num_running_reqs",
    "num_waiting_reqs",
    "num_waiting_uncached_tokens",
    "num_used_tokens",
    "num_total_tokens",
    "max_total_num_tokens",
    "max_running_requests",
)

_FLOAT_FIELDS = (
    "token_usage",
    "gen_throughput",
    "cache_hit_rate",
    "utilization",
)


def validate_rank_snapshot(load: LoadSnapshot, *, dp_size: int) -> pb.RankLoad:
    """Check a rank before it can replace a cached source or enter a report."""
    if not 0 <= load.dp_rank < dp_size:
        raise ValueError(f"unexpected dp_rank {load.dp_rank}")
    timestamp_ms = load.timestamp * 1000
    if not math.isfinite(timestamp_ms) or timestamp_ms < 1:
        raise ValueError("timestamp must be a positive finite source time")
    counts = {field: getattr(load, field) for field in _COUNT_FIELDS}
    if any(value < 0 for value in counts.values()):
        raise ValueError("load counts must be non-negative")
    if load.num_used_tokens > load.max_total_num_tokens:
        raise ValueError("num_used_tokens must not exceed max_total_num_tokens")
    if load.num_running_reqs > load.max_running_requests:
        raise ValueError("num_running_reqs must not exceed max_running_requests")
    floats = {field: getattr(load, field) for field in _FLOAT_FIELDS}
    if not all(math.isfinite(value) for value in floats.values()):
        raise ValueError("load metrics must be finite")
    return pb.RankLoad(
        dp_rank=load.dp_rank,
        snapshot_time_unix_ms=int(timestamp_ms),
        **counts,
        **floats,
    )


def validate_full_snapshot(
    loads: Sequence[LoadSnapshot], *, dp_size: int
) -> list[pb.RankLoad]:
    """Validate decoded values and copy only protocol fields, ordered by rank.

    The shared snapshot decoder checks Python field types. Protobuf construction
    checks target integer ranges. This boundary owns the remaining load semantics;
    it never substitutes collection time for an absent scheduler timestamp.
    The caller validates the fixed topology once, when creating the collector.
    """
    ranks = {}
    for load in loads:
        if not 0 <= load.dp_rank < dp_size or load.dp_rank in ranks:
            raise ValueError(f"unexpected or duplicate dp_rank {load.dp_rank}")
        ranks[load.dp_rank] = validate_rank_snapshot(load, dp_size=dp_size)
    if len(ranks) != dp_size:
        missing = sorted(set(range(dp_size)) - ranks.keys())
        raise ValueError(f"incomplete rank set: missing={missing}")
    return [ranks[rank] for rank in range(dp_size)]
