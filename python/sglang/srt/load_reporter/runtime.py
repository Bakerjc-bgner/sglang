"""Reporter-loop sessions with one latest-value mailbox per Router."""

from __future__ import annotations

import asyncio
import itertools
import time

from sglang.srt.load_reporter.config import validate_session_timing
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb


class _RouterSession:
    def __init__(self, runtime, router_id, token):
        self.runtime, self.router_id, self.token = runtime, router_id, token
        self.queue = asyncio.Queue(maxsize=1)
        self.done = False
        self.closed = asyncio.Event()

    def enqueue(self, report):
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(report)

    def refresh_lease(self):
        self.runtime.schedule.update(self.token, now=time.monotonic())
        self.runtime.wake_reporter()

    def stop(self):
        if self.done:
            return
        self.done = True
        self.closed.set()
        self.runtime.remove(self)
        self.enqueue(None)


class LoadReporterRuntime:
    """All control frames, broadcasts and mailbox updates run on one loop."""

    def __init__(self, schedule, wake_reporter=lambda: None):
        self.schedule, self.wake_reporter = schedule, wake_reporter
        self._sessions = {}
        self._routers = {}
        self._tokens = itertools.count(1)
        self._closed = False

    def register_session(self, router_id, report_interval_ms, lease_ttl_ms):
        if self._closed:
            raise RuntimeError("load reporter is shutting down")
        if not router_id.strip():
            raise ValueError("router_id must be non-empty")
        validate_session_timing(report_interval_ms, lease_ttl_ms)
        old = self._routers.get(router_id)
        if old is not None:
            old.stop()
        token = next(self._tokens)
        session = _RouterSession(self, router_id, token)
        self.schedule.add(token, report_interval_ms, lease_ttl_ms, now=time.monotonic())
        self._sessions[token] = self._routers[router_id] = session
        self.wake_reporter()
        return pb.RegisterResponse(
            lease_ttl_ms=lease_ttl_ms, renew_after_ms=max(1, lease_ttl_ms // 3)
        ), session

    def update_session_config(
        self, session, report_interval_ms=None, lease_ttl_ms=None
    ):
        updated = self.schedule.update(
            session.token, report_interval_ms, lease_ttl_ms, now=time.monotonic()
        )
        self.wake_reporter()
        return updated

    def remove(self, session):
        self.schedule.remove(session.token)
        self._sessions.pop(session.token, None)
        if self._routers.get(session.router_id) is session:
            del self._routers[session.router_id]
        self.wake_reporter()

    def offer(self, due, report, expired=()):
        if self._closed:
            return
        for token in expired:
            session = self._sessions.get(token)
            if session is not None:
                session.stop()
        for token in due:
            session = self._sessions.get(token)
            if session is not None:
                if report is None:
                    session.stop()
                else:
                    session.enqueue(report)

    def close(self):
        self._closed = True
        for session in list(self._sessions.values()):
            session.stop()
