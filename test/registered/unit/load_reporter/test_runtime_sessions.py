"""Unit tests for LoadReporterRuntime inbound Router session management.

Tests exercise runtime.register_session, session lifecycle, sampler
activation, and time-controlled lease expiry using monkeypatching.
No grpc.aio server is needed; sessions are driven directly.
"""

from __future__ import annotations

import asyncio
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest_plugins = ("pytest_asyncio",)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_server_args(dp_size: int = 1) -> types.SimpleNamespace:
    args = types.SimpleNamespace()
    args.host = "127.0.0.1"
    args.load_reporter_port = 9999
    args.load_reporter_snapshot_stale_after_ms = 30_000
    args.disaggregation_mode = "none"
    args.served_model_name = "test-model"
    args.load_reporter_zone = None
    args.dp_size = dp_size
    return args


class FakeSnapshotSource:
    """Minimal LoadSnapshotSource for testing."""

    def __init__(self, dp_size: int = 1) -> None:
        self._dp_size = dp_size
        self.get_loads_calls = 0

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        return []

    def expected_dp_ranks(self) -> frozenset:
        return frozenset(range(self._dp_size))


async def drain_queue(q: asyncio.Queue, count: int, timeout: float = 2.0) -> list:
    """Drain up to count non-None items from q within timeout seconds."""
    items = []
    deadline = time.monotonic() + timeout
    while len(items) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            item = await asyncio.wait_for(q.get(), timeout=remaining)
            if item is None:
                break
            items.append(item)
        except asyncio.TimeoutError:
            break
    return items


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterSession:
    @pytest.mark.asyncio
    async def test_register_returns_ack(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack, session = rt.register_session("r1", 500, 3000)
            assert ack.lease_ttl_ms == 3000
            assert ack.renew_after_ms == max(1, 3000 // 3)
            assert ack.renew_after_ms == 1000
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_initial_report_enqueued(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack, session = rt.register_session("r1", 500, 3000)
            # The session enqueues the first report immediately.
            reports = await drain_queue(session.queue, 1)
            assert len(reports) == 1, "Expected 1 initial report"
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_periodic_reports(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack, session = rt.register_session("r1", 30, 3000)
            # drain the immediate report + wait for at least 2 periodic ones
            reports = await drain_queue(session.queue, 3, timeout=2.0)
            assert len(reports) >= 2, f"Expected >=2 reports, got {len(reports)}"
        finally:
            session.stop()
            await rt.close()


class TestUpdateConfig:
    @pytest.mark.asyncio
    async def test_update_config_changes_interval(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack, session = rt.register_session("r1", 500, 3000)
            session.update_config(report_interval_ms=20)
            assert session.report_interval_ms == 20
            # fast interval means we get multiple reports quickly
            reports = await drain_queue(session.queue, 3, timeout=2.0)
            assert len(reports) >= 2
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_update_config_refreshes_lease(self):
        """After update_config extends the lease, session should keep reporting."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            # Short initial lease of 50ms.
            ack, session = rt.register_session("r1", 10, 50)
            # Extend the lease to 5000ms before the original 50ms expires.
            await asyncio.sleep(0.02)
            session.update_config(lease_ttl_ms=5000)
            # Wait well past the original 50ms window.
            await asyncio.sleep(0.1)
            # Session should still be emitting reports (queue not terminated).
            reports = await drain_queue(session.queue, 1, timeout=0.3)
            assert len(reports) >= 1, (
                "Session should still report after update_config extended the lease"
            )
        finally:
            session.stop()
            await rt.close()


class TestLeaseExpiry:
    @pytest.mark.asyncio
    async def test_keepalive_does_not_publish_before_report_deadline(self):
        """Lease renewals must not accelerate the periodic report cadence."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        keepalive_task = None
        try:
            _, session = rt.register_session("r1", 500, 200)
            initial_report = await asyncio.wait_for(session.queue.get(), timeout=1.0)
            assert initial_report is not None

            async def keep_lease_alive():
                while True:
                    await asyncio.sleep(0.05)
                    session.refresh_lease()

            keepalive_task = asyncio.create_task(keep_lease_alive())

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(session.queue.get(), timeout=0.3)
        finally:
            if keepalive_task is not None:
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)
            await rt.close()

    @pytest.mark.asyncio
    async def test_lease_expiry_stops_session(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            # Very short lease: session should expire quickly.
            ack, session = rt.register_session("r1", 5000, 20)
            # Wait for None sentinel in queue (lease expires)
            try:
                sentinel = await asyncio.wait_for(session.queue.get(), timeout=1.5)
                # might be a report first, then None
                if sentinel is not None:
                    sentinel = await asyncio.wait_for(session.queue.get(), timeout=1.5)
                assert sentinel is None, "Expected None sentinel after lease expiry"
            except asyncio.TimeoutError:
                pytest.fail("Session did not stop after lease expiry")
        finally:
            await rt.close()


class TestSameRouterIdReplacement:
    @pytest.mark.asyncio
    async def test_same_router_id_replaces_session(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack1, session1 = rt.register_session("r1", 500, 3000)
            # Re-register same router_id — replaces session1.
            ack2, session2 = rt.register_session("r1", 200, 3000)
            # session1 should receive None sentinel (stopped).
            sentinel = await asyncio.wait_for(session1.queue.get(), timeout=1.0)
            # consume any initial report first, then None
            if sentinel is not None:
                sentinel = await asyncio.wait_for(session1.queue.get(), timeout=1.0)
            assert sentinel is None, "Old session should have been stopped"
            assert session2 is not session1
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_replacement_does_not_corrupt_session_table(self):
        """C1 regression: old session's on_close must not delete the new session.

        Bug mechanism: _on_session_closed was generation-blind — it did
        self._sessions.pop(router_id) unconditionally, so when the old session's
        cleanup ran asynchronously it silently removed the new session from the
        table, deactivated the sampler, and leaked the new session's task.

        This test reproduces the exact asynchronous interleaving and asserts the
        behavioral invariants that would have failed on the buggy code:
        - sampler stays active after replacement (session2 is in the table)
        - session2 emits a fresh report (write loop is still running)
        """
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            ack1, session1 = rt.register_session("r1", 30, 3000)
            ack2, session2 = rt.register_session("r1", 30, 3000)

            # Let old session's _run() finally block execute (needs event loop yield).
            # Drain session1 until we see the None sentinel.
            sentinel = None
            for _ in range(10):
                await asyncio.sleep(0.01)
                while not session1.queue.empty():
                    item = session1.queue.get_nowait()
                    if item is None:
                        sentinel = item
                        break
                if sentinel is None and session1._done.is_set():
                    sentinel = None  # done event set; on_close has fired
                    break

            # Allow a bit more time for on_close to execute.
            await asyncio.sleep(0.05)

            # Sampler must still be active: session2 is in the table.
            calls_before = source.get_loads_calls
            await asyncio.sleep(0.15)
            calls_after = source.get_loads_calls
            assert calls_after > calls_before, (
                "Sampler must stay active after same-router-id replacement; "
                "generation-blind on_close would have deactivated it"
            )

            # session2 must still emit reports.
            reports = await drain_queue(session2.queue, 1, timeout=0.5)
            assert len(reports) >= 1, (
                "session2 must still emit reports after replacement; "
                "generation-blind on_close would have leaked its task"
            )
        finally:
            await rt.close()


class TestMultiRouter:
    @pytest.mark.asyncio
    async def test_different_routers_coexist(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack1, session1 = rt.register_session("r1", 30, 3000)
            ack2, session2 = rt.register_session("r2", 30, 3000)
            # Both sessions receive reports independently.
            reports1 = await drain_queue(session1.queue, 1)
            reports2 = await drain_queue(session2.queue, 1)
            assert len(reports1) >= 1
            assert len(reports2) >= 1
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_min_interval_sampling(self):
        """Sampler must use the shortest session interval (behavioral)."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            # s1 wants 1000ms, s2 wants 30ms: sampler should run ~at 30ms cadence.
            ack1, s1 = rt.register_session("r1", 1000, 5000)
            ack2, s2 = rt.register_session("r2", 30, 5000)
            before = source.get_loads_calls
            await asyncio.sleep(0.2)
            after = source.get_loads_calls
            # At 30ms interval over 200ms we expect at least 4 samples; at
            # 1000ms interval we'd expect at most 1.  Assert a clear majority.
            assert after - before >= 3, (
                f"Expected >=3 samples at 30ms min interval, got {after - before}"
            )
        finally:
            s1.stop()
            s2.stop()
            await rt.close()


class TestSamplerActivation:
    @pytest.mark.asyncio
    async def test_sampler_activates_on_first_session(self):
        """Sampler must sample after a session is registered (behavioral)."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            before = source.get_loads_calls
            ack, session = rt.register_session("r1", 500, 3000)
            await asyncio.sleep(0.15)
            after = source.get_loads_calls
            assert after > before, (
                "Sampler should start sampling when first session is registered"
            )
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_sampler_deactivates_on_last_close(self):
        """Sampler must stop sampling after the last session closes (behavioral)."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            ack, session = rt.register_session("r1", 500, 3000)
            session.stop()
            # Allow on_close to fire and deactivate the sampler.
            await asyncio.sleep(0.1)
            # Capture call count after deactivation.
            snapshot = source.get_loads_calls
            await asyncio.sleep(0.15)
            after = source.get_loads_calls
            assert after == snapshot, (
                "Sampler should stop sampling after last session closes"
            )
        finally:
            await rt.close()


class TestDecoratorEvents:
    @pytest.mark.asyncio
    async def test_notify_refresh_wakes_sampler(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            ack, session = rt.register_session("r1", 5000, 30000)
            before = source.get_loads_calls
            rt.notify_refresh()
            await asyncio.sleep(0.1)
            after = source.get_loads_calls
            assert after > before, "notify_refresh should trigger a sample"
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_notify_request_finished_wakes_sampler(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            ack, session = rt.register_session("r1", 5000, 30000)
            before = source.get_loads_calls
            rt.notify_request_finished()
            await asyncio.sleep(0.1)
            after = source.get_loads_calls
            assert after > before
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_notify_source_changed_wakes_sampler(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            ack, session = rt.register_session("r1", 5000, 30000)
            before = source.get_loads_calls
            rt.notify_source_changed()
            await asyncio.sleep(0.1)
            after = source.get_loads_calls
            assert after > before
        finally:
            session.stop()
            await rt.close()
