"""Load reporter lifecycle management."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


class LoadReporterHandle:
    """Own reporter resources for one process."""

    def __init__(self) -> None:
        self._runtime: Optional[Any] = None
        self._server: Optional[Any] = None
        self._close_task: Optional[asyncio.Task[None]] = None

    def update_expected_dp_ranks(self, ranks: Iterable[int]) -> bool:
        """Update the rank-aware source after elastic scaling."""
        if self._runtime is None:
            return False
        return self._runtime.update_expected_dp_ranks(ranks)

    async def close(self) -> None:
        """Tear down reporter resources once."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_impl(), name="load-reporter-handle-close"
            )
        await asyncio.shield(self._close_task)

    async def _close_impl(self) -> None:
        """Run one shared teardown attempt to completion for every caller."""
        if self._server is not None:
            try:
                await self._server.stop(grace=None)
            except Exception:
                logger.exception("Load reporter gRPC server stop failed")
        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception:
                logger.exception("Load reporter runtime shutdown failed")


async def start_load_reporter(
    server_args: Any,
    snapshot_source: Any,
) -> Optional[LoadReporterHandle]:
    """Start the reporter and return its handle when enabled.

    Args:
        server_args: ServerArgs with load_reporter_port and host.
        snapshot_source: Required when enabled; the LoadSnapshotSource to sample.

    Returns:
        Handle if the reporter is enabled, None otherwise.
    """
    if getattr(server_args, "load_reporter_port", None) is None:
        return None

    if snapshot_source is None:
        raise ValueError(
            "snapshot_source is required when load reporter is enabled"
        )

    return await _start_owner(server_args, snapshot_source)


async def _start_owner(
    server_args: Any,
    snapshot_source: Any,
) -> LoadReporterHandle:
    """Start the reporter runtime and gRPC listener on the reporter port."""
    import grpc.aio

    from sglang.srt.load_reporter.runtime import LoadReporterRuntime
    from sglang.srt.load_reporter.service import add_service_to_server

    handle = LoadReporterHandle()
    try:
        runtime = LoadReporterRuntime(snapshot_source, server_args)
        handle._runtime = runtime

        server = grpc.aio.server()
        add_service_to_server(runtime, server)
        server.add_insecure_port(f"{server_args.host}:{server_args.load_reporter_port}")
        await server.start()
        handle._server = server
    except BaseException:
        await handle.close()
        raise
    return handle
