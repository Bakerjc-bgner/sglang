"""Single composition root for the embedded load reporter.

``start_load_reporter`` is the only public bootstrap symbol.  Every serving
mode (HTTP, native gRPC, embedded Engine, multi-tokenizer, standalone SMG RPC)
calls it and only sees ``start`` and ``close``; no reporter-internal type
(runtime, sampler, IPC notifier, gRPC/protobuf) leaks into serving entrypoints.

Path selection
--------------
* ``load_reporter_port is None`` → return ``None`` *before* importing the
  optional gRPC/protobuf stack.  Zero socket, task, or dependency overhead.
* ``snapshot_source is None`` (multi-tokenizer HTTP worker) → install a
  coalescing refresh notifier bound to ``event_owner`` that forwards refresh
  hints to the sole router over IPC.  No gRPC server, no port is bound.
* otherwise (single-owner) → own a ``LoadReporterRuntime`` + a ``grpc.aio``
  server listening on ``host:load_reporter_port``, and optionally bind
  ``event_owner`` so decorator events wake the sampler.

The returned :class:`LoadReporterHandle` owns every resource it created and
tears them down in reverse order on an idempotent ``close()``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)


class LoadReporterHandle:
    """Owns the reporter resources for one Worker process.

    Serving entrypoints keep the handle opaque: they call ``close()`` on
    shutdown and, in the multi-tokenizer router only, ``notify_refresh`` /
    ``update_expected_dp_ranks``.  All methods are idempotent and safe to call
    after ``close()``.
    """

    def __init__(self) -> None:
        self._runtime: Optional[Any] = None
        self._server: Optional[Any] = None
        self._notifier: Optional[Any] = None
        # Reverse-ordered teardown steps registered during startup.  Each entry
        # is a zero-arg callable returning either ``None`` or an awaitable.
        self._closers: List[Callable[[], Any]] = []
        self._closed = False

    # -- construction helpers (used by start_load_reporter only) -------------

    def _push(self, closer: Callable[[], Any]) -> None:
        self._closers.append(closer)

    # -- delegation surface (multi-tokenizer router) -------------------------

    def notify_refresh(self) -> None:
        """Wake the sampler once (router IPC refresh).  No-op without runtime."""
        if self._runtime is not None:
            self._runtime.notify_refresh()

    def update_expected_dp_ranks(self, ranks: Iterable[int]) -> bool:
        """Update the rank-aware source after elastic scaling.

        Returns ``False`` when there is no owning runtime (IPC-worker handle)
        or the source did not accept a changed rank set.
        """
        if self._runtime is None:
            return False
        return self._runtime.update_expected_dp_ranks(ranks)

    # -- shutdown ------------------------------------------------------------

    async def close(self) -> None:
        """Idempotent, reverse-order teardown of every acquired resource."""
        if self._closed:
            return
        self._closed = True
        for closer in reversed(self._closers):
            try:
                result = closer()
                if result is not None and hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.exception("Load reporter shutdown step failed")
        self._closers.clear()


async def start_load_reporter(
    server_args: Any,
    snapshot_source: Optional[Any],
    *,
    event_owner: Optional[Any] = None,
) -> Optional[LoadReporterHandle]:
    """Start the embedded load reporter for one serving entrypoint.

    Args:
        server_args: Resolved SGLang server configuration.  Only
            ``load_reporter_port`` gates activation.
        snapshot_source: A ``LoadSnapshotSource`` for the owner path, or
            ``None`` for a multi-tokenizer HTTP worker (IPC-forwarding path).
        event_owner: The instance whose decorated ``generate_request`` /
            ``_dispatch_to_scheduler`` should wake the sampler.  ``None`` means
            interval + register-time sampling only.

    Returns:
        A :class:`LoadReporterHandle` when reporting is enabled, else ``None``.
    """
    if getattr(server_args, "load_reporter_port", None) is None:
        return None

    if snapshot_source is None:
        return await _start_ipc_worker(server_args, event_owner)
    return await _start_owner(server_args, snapshot_source, event_owner)


async def _start_ipc_worker(
    server_args: Any, event_owner: Optional[Any]
) -> Optional[LoadReporterHandle]:
    """Multi-tokenizer HTTP worker: coalesce refresh hints to the sole owner.

    Binds a refresh notifier to ``event_owner`` and forwards its coalesced
    events to the router over the existing scheduler IPC channel.  No gRPC
    server is started and no reporter port is bound.
    """
    if event_owner is None:
        return None

    from sglang.srt.load_reporter.decorator import bind_load_monitor
    from sglang.srt.load_reporter.ipc import LoadReporterRefreshNotifier

    handle = LoadReporterHandle()
    notifier = LoadReporterRefreshNotifier(
        worker_id=f"http-worker-{os.getpid()}",
        send=event_owner._dispatch_to_scheduler,
    )
    handle._notifier = notifier
    await notifier.start()
    unbind = bind_load_monitor(event_owner, notifier.notify)
    handle._push(unbind)
    handle._push(notifier.close)
    return handle


async def _start_owner(
    server_args: Any,
    snapshot_source: Any,
    event_owner: Optional[Any],
) -> LoadReporterHandle:
    """Single-owner path: own a runtime + gRPC listener on the reporter port."""
    import grpc.aio

    from sglang.srt.load_reporter.decorator import bind_load_monitor
    from sglang.srt.load_reporter.runtime import LoadReporterRuntime
    from sglang.srt.load_reporter.service import add_service_to_server

    handle = LoadReporterHandle()
    try:
        runtime = LoadReporterRuntime(snapshot_source, server_args)
        handle._runtime = runtime
        handle._push(runtime.close)

        server = grpc.aio.server()
        add_service_to_server(runtime, server)
        # Explicit bind: grpc.aio raises RuntimeError on failure (never a
        # silent random-port fallback), which we surface after cleanup.
        server.add_insecure_port(
            f"{server_args.host}:{server_args.load_reporter_port}"
        )
        await server.start()
        handle._server = server
        handle._push(lambda: server.stop(grace=None))

        if event_owner is not None:
            unbind = bind_load_monitor(
                event_owner, lambda reason, count: runtime.notify_refresh()
            )
            handle._push(unbind)
    except BaseException:
        await handle.close()
        raise
    return handle
