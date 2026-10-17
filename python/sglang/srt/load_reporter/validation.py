"""Feature-specific startup validation for instance PUB reporting."""

from typing import Any
from urllib.parse import urlsplit

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.runtime_context import attn_dp_enabled_of, num_dp_ranks_of
from sglang.srt.utils.network import NetworkAddress


def check_load_reporter_args(server_args: Any):
    """Validate the derived listeners only when instance reporting is enabled."""
    cfg = resolving_view(server_args)
    if cfg.load_reporter_port is None:
        return
    from sglang.srt.load_reporter.ingress import CollectorConfig

    if cfg.load_reporter_port in (cfg.port, cfg.grpc_port, cfg.nccl_port):
        raise ValueError(
            "--load-reporter-port must differ from --port, --grpc-port and --nccl-port"
        )
    if (cfg.load_publish_endpoint or "").strip().lower() not in ("", "off"):
        raise ValueError(
            "load reporter and --load-publish-endpoint are mutually exclusive"
        )

    config = CollectorConfig.from_values(
        reporter_port=cfg.load_reporter_port,
        dist_init_addr=cfg.dist_init_addr,
        nnodes=cfg.nnodes,
        dp_size=num_dp_ranks_of(cfg),
    )
    if urlsplit(config.endpoint).port in (
        cfg.load_reporter_port,
        cfg.port,
        cfg.grpc_port,
        cfg.nccl_port,
        NetworkAddress.parse(cfg.dist_init_addr).port if cfg.dist_init_addr else None,
    ):
        raise ValueError(
            "load reporter collector port must differ from --load-reporter-port, "
            "--port, --grpc-port, --nccl-port and the --dist-init-addr port"
        )

    from sglang.srt.environ import envs

    if not (
        (attn_dp_enabled_of(cfg) and cfg.nnodes > 1)
        or envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.get()
    ):
        raise ValueError(
            "load reporter requires SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1 "
            "or multi-node DP attention to preserve the internal PUSH path"
        )
