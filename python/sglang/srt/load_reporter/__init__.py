"""Instance load reporting; optional gRPC imports stay behind feature startup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.load_reporter.lifecycle import LoadReporterHandle


def is_load_reporter_enabled() -> bool:
    from sglang.srt.runtime_context import get_observability

    return get_observability().load_reporter_port is not None


def is_reporter_host(*, http_parent: bool = False) -> bool:
    from sglang.srt.runtime_context import get_parallel, get_serving

    if not is_load_reporter_enabled() or get_parallel().node_rank != 0:
        return False
    return http_parent == (get_serving().tokenizer_worker_num > 1)


def init_load_reporter(*, http_parent: bool = False) -> LoadReporterHandle | None:
    """Create only the instance leader's PUB reporter; never adopt a native reader."""
    if not is_reporter_host(http_parent=http_parent):
        return None
    from functools import partial

    from sglang.srt.load_reporter.config import ReporterSettings
    from sglang.srt.load_reporter.ingress import collector_config_from_runtime
    from sglang.srt.load_reporter.lifecycle import LoadReporterHandle
    from sglang.srt.load_reporter.pub_source import ZmqPubSnapshotSource
    from sglang.srt.managers.load_snapshot import should_use_zmq
    from sglang.srt.runtime_context import (
        get_disagg,
        get_observability,
        get_serving,
    )

    if not should_use_zmq():
        raise ValueError("PUB reporting requires the internal PUSH snapshot writer")
    config = collector_config_from_runtime()
    settings = ReporterSettings(
        host=get_serving().host,
        port=get_observability().load_reporter_port,
        disaggregation_mode=get_disagg().disaggregation_mode,
        model=get_serving().served_model_name,
    )
    return LoadReporterHandle(partial(ZmqPubSnapshotSource, config), settings)


def start_load_reporter(reporter: LoadReporterHandle | None) -> None:
    if reporter is not None:
        reporter.start()


def close_load_reporter(reporter: LoadReporterHandle | None) -> None:
    if reporter is not None:
        reporter.close()


def describe_load_reporter(server_args: Any) -> Optional[dict]:
    """Return the load reporter endpoint description, or None when not hosted."""
    from sglang.srt.arg_groups.overrides import resolving_view

    resolved = resolving_view(server_args)
    if resolved.load_reporter_port is None or resolved.node_rank != 0:
        return None
    return {
        "port": resolved.load_reporter_port,
        "protocol": "grpc",
        "service": "sglang.router.loadmonitor.v1.LoadMonitorService",
    }
