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

import asyncio
import concurrent.futures
import logging
import os
import threading
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

_BACKGROUND_OPERATION_TIMEOUT_SECONDS = 10.0


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
        self._unbind: Optional[Callable[[], None]] = None
        self._restore: Optional[Callable[[], None]] = None
        self._closed = False

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
        """Idempotent teardown.

        Order: stop accepting Router sessions/reports, stop sampling, close the
        IPC notifier, then unbind the decorator registry callback and restore
        any shadowed bound method.  Each step is guarded so a partially started
        handle (e.g. failed port bind) closes cleanly.
        """
        if self._closed:
            return
        self._closed = True

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
        if self._notifier is not None:
            try:
                await self._notifier.close()
            except Exception:
                logger.exception("Load reporter notifier shutdown failed")
        if self._unbind is not None:
            try:
                self._unbind()
            except Exception:
                logger.exception("Load reporter unbind failed")
        if self._restore is not None:
            try:
                self._restore()
            except Exception:
                logger.exception("Load reporter method restore failed")


class BackgroundLoadReporter:
    """Own an async load reporter on a continuously running loop thread."""

    def __init__(
        self,
        server_args: Any,
        snapshot_source: Any,
        event_owner: Optional[Any],
    ) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="load-reporter-event-loop",
            daemon=True,
        )
        self._handle: Optional[LoadReporterHandle] = None
        self._unbind: Optional[Callable[[], None]] = None
        self._closed = False
        self._thread.start()

        future = asyncio.run_coroutine_threadsafe(
            start_load_reporter(server_args, snapshot_source, event_owner=None),
            self._loop,
        )
        try:
            self._handle = future.result(
                timeout=_BACKGROUND_OPERATION_TIMEOUT_SECONDS
            )
            if event_owner is not None:
                from sglang.srt.load_reporter.decorator import bind_load_monitor

                self._unbind = bind_load_monitor(
                    event_owner, lambda _reason, _count: self.notify_refresh()
                )
        except BaseException:
            future.cancel()
            self._stop_loop()
            raise

    def _run_loop(self) -> None:
        """Run the owned event loop and clean up any residual tasks on exit."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    def notify_refresh(self) -> None:
        """Thread-safely forward a request-lifecycle refresh hint."""
        handle = self._handle
        if self._closed or handle is None:
            return
        try:
            self._loop.call_soon_threadsafe(handle.notify_refresh)
        except RuntimeError:
            if not self._closed:
                logger.exception("Load reporter background notification failed")

    def close(self) -> None:
        """Synchronously close the reporter and its event-loop thread."""
        if self._closed:
            return
        self._closed = True

        if self._unbind is not None:
            try:
                self._unbind()
            except Exception:
                logger.exception("Load reporter background unbind failed")
            self._unbind = None

        handle, self._handle = self._handle, None
        if handle is not None:
            future = asyncio.run_coroutine_threadsafe(handle.close(), self._loop)
            try:
                future.result(timeout=_BACKGROUND_OPERATION_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                future.cancel()
                logger.warning("Timed out while closing background load reporter")
            except Exception:
                logger.exception("Background load reporter shutdown failed")

        self._stop_loop()

    def _stop_loop(self) -> None:
        """Stop and join the owned loop thread."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=_BACKGROUND_OPERATION_TIMEOUT_SECONDS)


def start_load_reporter_in_background(
    server_args: Any,
    snapshot_source: Any,
    *,
    event_owner: Optional[Any] = None,
) -> Optional[BackgroundLoadReporter]:
    """Start a synchronously owned reporter on a dedicated event-loop thread."""
    if getattr(server_args, "load_reporter_port", None) is None:
        return None
    return BackgroundLoadReporter(server_args, snapshot_source, event_owner)


async def start_load_reporter(
    server_args: Any,
    snapshot_source: Optional[Any],
    *,
    event_owner: Optional[Any] = None,
    request_lifecycle_method: Optional[str] = None,
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
        request_lifecycle_method: When set (standalone SMG RPC), the named bound
            async-generator method on ``event_owner`` is wrapped at runtime with
            the same ``enable_load_monitor("request_lifecycle")`` decorator and
            installed on that single instance; restored on ``close()``.

    Returns:
        A :class:`LoadReporterHandle` when reporting is enabled, else ``None``.
    """
    if getattr(server_args, "load_reporter_port", None) is None:
        return None

    if snapshot_source is None:
        return await _start_ipc_worker(server_args, event_owner)
    return await _start_owner(
        server_args, snapshot_source, event_owner, request_lifecycle_method
    )


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
    handle._unbind = bind_load_monitor(event_owner, notifier.notify)
    return handle


async def _start_owner(
    server_args: Any,
    snapshot_source: Any,
    event_owner: Optional[Any],
    request_lifecycle_method: Optional[str],
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

        server = grpc.aio.server()
        add_service_to_server(runtime, server)
        # Explicit bind: grpc.aio raises RuntimeError on failure (never a
        # silent random-port fallback), which we surface after cleanup.
        server.add_insecure_port(f"{server_args.host}:{server_args.load_reporter_port}")
        await server.start()
        handle._server = server

        if event_owner is not None:
            handle._unbind = bind_load_monitor(
                event_owner, lambda reason, count: runtime.notify_refresh()
            )
        if request_lifecycle_method is not None:
            _install_lifecycle_shadow(handle, event_owner, request_lifecycle_method)
    except BaseException:
        await handle.close()
        raise
    return handle


def _install_lifecycle_shadow(
    handle: LoadReporterHandle, owner: Any, method_name: str
) -> None:
    """Wrap ``owner.<method_name>`` with the request-lifecycle decorator.

    Installs the decorated callable as an instance attribute on this single
    ``owner`` only — the class method and every other instance are untouched.
    Registers an identity-safe restore that removes the instance shadow only
    while it still resolves to this wrapper.
    """
    from sglang.srt.load_reporter.decorator import enable_load_monitor

    original = getattr(owner, method_name)
    decorated = enable_load_monitor("request_lifecycle")(original)
    setattr(owner, method_name, decorated)

    def _restore() -> None:
        # Only undo our own shadow; never clobber a later replacement.
        if owner.__dict__.get(method_name, None) is decorated:
            del owner.__dict__[method_name]

    handle._restore = _restore
