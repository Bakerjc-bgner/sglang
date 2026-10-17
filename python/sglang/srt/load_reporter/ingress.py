"""One instance-local SUB endpoint, shared by all Scheduler publishers."""

from __future__ import annotations

from urllib.parse import urlsplit

import msgspec

from sglang.srt.utils.network import NetworkAddress


class CollectorConfig(msgspec.Struct, frozen=True):
    endpoint: str
    dp_size: int
    instance_id: str
    topology_generation: str

    def bind_address(self) -> str:
        address = urlsplit(self.endpoint)
        wildcard = "[::]" if ":" in address.hostname else "*"
        return f"tcp://{wildcard}:{address.port}"

    @classmethod
    def from_values(
        cls,
        *,
        reporter_port: int,
        dist_init_addr: str | None,
        nnodes: int,
        dp_size: int,
    ) -> CollectorConfig:
        if not 1 <= reporter_port <= 65534:
            raise ValueError(
                "--load-reporter-port must be in [1, 65534]; "
                "the next port is reserved for the collector"
            )
        if dist_init_addr:
            host = NetworkAddress.parse(dist_init_addr).host
        elif nnodes == 1:
            host = "127.0.0.1"
        else:
            raise ValueError("multi-node reporting requires --dist-init-addr")
        endpoint = NetworkAddress(host, reporter_port + 1).to_tcp()
        address = urlsplit(endpoint)
        if (
            address.scheme != "tcp"
            or not address.hostname
            or address.hostname in ("*", "0.0.0.0", "::")
            or address.port is None
            or not 1 <= address.port <= 65535
            or address.path
            or address.query
            or address.fragment
            or address.username is not None
            or address.password is not None
        ):
            raise ValueError(
                "load reporter requires a reachable node-0 host in --dist-init-addr"
            )
        if dp_size < 1:
            raise ValueError("load reporter requires at least one DP rank")
        # These wire tags identify the listener and fixed rank set, not a
        # deployment incarnation. Process restarts use publisher_boot_id.
        return cls(endpoint, dp_size, endpoint, f"dp-size:{dp_size}")


def collector_config_from_runtime() -> CollectorConfig:
    from sglang.srt.runtime_context import get_observability, get_parallel

    observability, parallel = get_observability(), get_parallel()
    return CollectorConfig.from_values(
        reporter_port=observability.load_reporter_port,
        dist_init_addr=parallel.dist_init_addr,
        nnodes=parallel.nnodes,
        dp_size=parallel.num_dp_ranks,
    )
