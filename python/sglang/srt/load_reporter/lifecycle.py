"""PUB collection, report clock and gRPC share one independent reporter loop."""

from __future__ import annotations

import asyncio
import atexit
import logging
import math
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future

from sglang.srt.load_reporter.config import SHUTDOWN_TIMEOUT_SECONDS
from sglang.srt.load_reporter.schedule import ReportSchedule

logger = logging.getLogger(__name__)


class CollectorDriver:
    def __init__(self, source, schedule, sink, metadata):
        from sglang.srt.load_reporter.report_builder import ReportBuilder

        if source.dp_size <= 0:
            raise ValueError("load collector requires a positive dp_size")
        self._dp_size = source.dp_size
        self.source, self.schedule, self.sink, self.metadata = (
            source,
            schedule,
            sink,
            metadata,
        )
        self.builder = ReportBuilder(str(uuid.uuid4()))
        self.stats = Counter()

    def run_due(self, now=None):
        now = time.monotonic() if now is None else now
        deadline = self.schedule.next_deadline()
        if now < deadline:
            return
        broadcast, due, expired = self.schedule.take_due(now)
        report = None
        if broadcast:
            self.stats["ticks"] += 1
            self.stats["max_lateness_seconds"] = max(
                self.stats["max_lateness_seconds"], now - deadline
            )
            try:
                from sglang.srt.load_reporter.snapshot_validation import (
                    validate_full_snapshot,
                )

                ranks = validate_full_snapshot(
                    self.source.snapshot(), dp_size=self._dp_size
                )
                report = self.builder.build(
                    ranks,
                    self.metadata,
                    report_time_unix_ms=time.time_ns() // 1_000_000,
                )
            except Exception as exc:
                self.stats["errors"] += 1
                if self.stats["errors"] == 1:
                    logger.warning(
                        "Load collector could not build a complete view: %s", exc
                    )
                report = self.builder.build_unreachable(
                    self.metadata,
                    report_time_unix_ms=time.time_ns() // 1_000_000,
                    error=exc,
                )
        if due or expired:
            self.sink.offer(due, report, expired)


class AsyncCollectorRunner:
    """One reporter-loop timer; control notifications preserve broadcast phase."""

    def __init__(self, driver, loop, on_failure):
        self.driver, self.loop, self.on_failure = driver, loop, on_failure
        self._timer = self._wakeup = None
        self._closed = False
        self._tick()

    def notify(self):
        if not self._closed and self._wakeup is None:
            self._wakeup = self.loop.call_soon(self._tick)

    def _cancel_callbacks(self):
        for handle in (self._timer, self._wakeup):
            if handle is not None:
                handle.cancel()
        self._timer = self._wakeup = None

    def _tick(self):
        if self._closed:
            return
        self._cancel_callbacks()
        try:
            self.driver.run_due()
            deadline = self.driver.schedule.next_deadline()
            if math.isfinite(deadline):
                self._timer = self.loop.call_later(
                    max(0, deadline - time.monotonic()), self._tick
                )
        except Exception:
            logger.exception("Load reporter timer stopped")
            self.close()
            self.on_failure()

    def close(self):
        self._closed = True
        self._cancel_callbacks()


class ReporterTransport:
    def __init__(self, source_factory, metadata):
        self._source_factory, self._metadata = source_factory, metadata
        self._schedule = ReportSchedule()
        self.source = self.driver = self._runner = None
        self._ready = Future()
        self._loop = self._serve_task = None
        self._stop_requested = threading.Event()
        self.runtime = None
        self._thread = threading.Thread(
            target=self._run, name="load-reporter", daemon=True
        )
        self._thread.start()
        try:
            self.bound_port = self._ready.result(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except BaseException:
            self.close()
            raise

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._serve_task = self._loop.create_task(self._serve())
            if self._stop_requested.is_set():
                self._serve_task.cancel()
            self._loop.run_until_complete(self._serve_task)
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.exception("Load reporter stopped")
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    async def _serve(self):
        import grpc.aio

        from sglang.srt.load_reporter.runtime import LoadReporterRuntime
        from sglang.srt.load_reporter.service import add_service_to_server

        # Imports may have blocked past the startup timeout. Do not bind late.
        if self._stop_requested.is_set():
            return
        self.runtime = LoadReporterRuntime(self._schedule, self._wake_collection)
        server = grpc.aio.server(options=[("grpc.so_reuseport", 0)])
        try:
            self.source = self._source_factory()
            self.source.start()
            self.driver = CollectorDriver(
                self.source, self._schedule, self.runtime, self._metadata
            )
            self._runner = AsyncCollectorRunner(
                self.driver, self._loop, self._serve_task.cancel
            )
            add_service_to_server(self.runtime, server)
            port = server.add_insecure_port(self._metadata.worker_addr)
            if not port:
                raise RuntimeError(
                    f"Cannot bind load reporter at {self._metadata.worker_addr}"
                )
            await server.start()
            self._ready.set_result(port)
            await asyncio.Future()
        finally:
            if self._runner is not None:
                self._runner.close()
            if self.source is not None:
                await self.source.aclose()
            self.runtime.close()
            await server.stop(grace=None)

    def _wake_collection(self):
        if self._runner is not None:
            self._runner.notify()

    def close(self):
        if not self._stop_requested.is_set():
            self._stop_requested.set()
            if self._loop is not None and self._serve_task is not None:
                try:
                    self._loop.call_soon_threadsafe(self._serve_task.cancel)
                except RuntimeError:
                    if not self._loop.is_closed():
                        raise
        self._thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("load reporter did not stop")


class LoadReporterHandle:
    """Deferred PUB/gRPC startup, with no ownership of internal readers or loops."""

    def __init__(self, source_factory, settings):
        self._source_factory, self._settings = source_factory, settings
        self.transport = self.source = self.driver = None
        self._closed = False

    def start(self):
        if self._closed:
            raise RuntimeError("load reporter is closed")
        if self.transport is not None:
            return
        try:
            self.transport = ReporterTransport(
                self._source_factory, self._settings.metadata()
            )
            self.source, self.driver = self.transport.source, self.transport.driver
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        if self.transport is not None:
            self.transport.close()
