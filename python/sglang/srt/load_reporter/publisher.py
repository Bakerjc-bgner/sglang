"""Scheduler publishers connect to the instance leader's SUB listener."""

from __future__ import annotations

import uuid

import zmq

from sglang.srt.disaggregation.kv_events import select_kv_publisher_dp_rank
from sglang.srt.load_reporter.ingress import collector_config_from_runtime
from sglang.srt.load_reporter.publisher_protocol import metadata_for_snapshot
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.network import is_zmq_endpoint_ipv6


def open_publisher_socket(endpoint: str, *, hwm: int) -> zmq.Socket:
    sock = zmq.Context.instance().socket(zmq.PUB)
    try:
        sock.set_hwm(hwm)
        sock.setsockopt(zmq.LINGER, 0)
        if is_zmq_endpoint_ipv6(endpoint):
            sock.setsockopt(zmq.IPV6, 1)
        # The HTTP-owned collector may start after the Scheduler is ready.
        # Connecting must not wait for it; subsequent samples repair early loss.
        sock.connect(endpoint)
    except Exception:
        sock.close()
        raise
    return sock


class ConnectingPublisher:
    """Own reporter source identity without changing native publishing policy."""

    def __init__(self, *, hwm: int):
        self.socket = None
        self.rank = 0
        self._config = None
        self._boot_id = str(uuid.uuid4())
        parallel = get_parallel()
        if parallel.pp_rank or parallel.attn_tp_rank or parallel.attn_cp_rank:
            return
        self._config = collector_config_from_runtime()
        self.rank = select_kv_publisher_dp_rank(
            parallel.attn_dp_size, parallel.attn_dp_rank, parallel.dp_rank
        )
        self.socket = open_publisher_socket(self._config.endpoint, hwm=hwm)

    @staticmethod
    def extra_counts(load):
        return (
            load.token_usage,
            load.gen_throughput,
            load.cache_hit_rate,
            load.utilization,
        )

    def metadata(self, load):
        return metadata_for_snapshot(load, self._config, self._boot_id)
