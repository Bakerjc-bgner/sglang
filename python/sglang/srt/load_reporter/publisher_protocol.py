"""Additive PUB metadata; importing this module never imports gRPC."""

from __future__ import annotations

import math

import msgspec


class PublisherMetadata(msgspec.Struct, frozen=True):
    version: int
    instance_id: str
    topology_generation: str
    publisher_boot_id: str
    # Exactly LoadSnapshot.timestamp, not send, receive or report construction time.
    snapshot_timestamp: float
    token_usage: float
    gen_throughput: float
    cache_hit_rate: float
    utilization: float


def metadata_for_snapshot(snapshot, config, publisher_boot_id):
    return PublisherMetadata(
        version=1,
        instance_id=config.instance_id,
        topology_generation=config.topology_generation,
        publisher_boot_id=publisher_boot_id,
        snapshot_timestamp=snapshot.timestamp,
        token_usage=snapshot.token_usage,
        gen_throughput=snapshot.gen_throughput,
        cache_hit_rate=snapshot.cache_hit_rate,
        utilization=snapshot.utilization,
    )


def snapshot_from_stat(stat, config):
    from sglang.srt.managers.load_snapshot import LoadSnapshot

    metadata = stat.reporter_metadata
    if metadata is None:
        raise ValueError("load PUB message has no source timestamp/metadata")
    if (
        metadata.version != 1
        or metadata.instance_id != config.instance_id
        or metadata.topology_generation != config.topology_generation
        or stat.attn_dp_rank is None
        or not 0 <= stat.attn_dp_rank < config.dp_size
        or not metadata.publisher_boot_id
    ):
        raise ValueError("load PUB source does not match the instance topology")
    if (
        not math.isfinite(metadata.snapshot_timestamp)
        or metadata.snapshot_timestamp <= 0
    ):
        raise ValueError("load PUB source timestamp must be positive and finite")
    return LoadSnapshot(
        dp_rank=stat.attn_dp_rank,
        timestamp=metadata.snapshot_timestamp,
        num_running_reqs=stat.num_running_reqs,
        num_waiting_reqs=stat.num_waiting_reqs,
        num_waiting_uncached_tokens=stat.num_waiting_uncached_tokens,
        num_used_tokens=stat.num_tokens,
        num_total_tokens=stat.num_total_tokens,
        max_total_num_tokens=stat.max_total_num_tokens,
        max_running_requests=stat.max_running_requests,
        token_usage=metadata.token_usage,
        gen_throughput=metadata.gen_throughput,
        cache_hit_rate=metadata.cache_hit_rate,
        utilization=metadata.utilization,
    )
