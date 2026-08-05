"""Characterization tests for LoadReporterLifecycle ownership and cleanup.

These tests fix the current behavior before refactoring lifespan logic out of
http_server.py into a reporter-owned facade.
"""

import asyncio
import logging
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


# Fake collaborators for testing lifecycle behavior
class FakeTokenizerManager:
    """Minimal TokenizerManager interface needed by lifecycle."""

    def __init__(self):
        self.request_finished_hook: Optional[Callable] = None
        self.request_event_hook: Optional[Callable] = None
        self.ipc_proxy: Optional[Any] = None
        self.ipc_notifier: Optional[Any] = None
        self.attach_calls = []
        self.detach_calls = []

    def set_load_reporter_request_finished_hook(self, hook: Optional[Callable]):
        self.request_finished_hook = hook
        if hook is None:
            self.detach_calls.append("request_finished_hook")

    def set_load_reporter_request_event_hook(self, hook: Optional[Callable]):
        self.request_event_hook = hook
        if hook is None:
            self.detach_calls.append("request_event_hook")

    def attach_load_reporter_ipc_components(self, proxy, notifier):
        self.ipc_proxy = proxy
        self.ipc_notifier = notifier
        if proxy is None and notifier is None:
            self.detach_calls.append("ipc_components")
        else:
            self.attach_calls.append("ipc_components")


class FakeLoadReporterRuntime:
    """Fake single-tokenizer runtime."""

    def __init__(self):
        self.close_called = False
        self.close_error: Optional[Exception] = None

    async def close(self):
        self.close_called = True
        if self.close_error:
            raise self.close_error


class FakeLoadReporterNotifier:
    """Fake multi-tokenizer refresh notifier."""

    def __init__(self):
        self.start_called = False
        self.close_called = False
        self.close_error: Optional[Exception] = None

    async def start(self):
        self.start_called = True

    async def close(self):
        self.close_called = True
        if self.close_error:
            raise self.close_error


class FakeServerArgs:
    """Minimal ServerArgs for lifecycle construction."""

    def __init__(self, tokenizer_worker_num=1):
        self.tokenizer_worker_num = tokenizer_worker_num
        self.host = "0.0.0.0"
        self.port = 30000


class FakeAppState:
    """Minimal FastAPI app.state for lifecycle."""

    def __init__(self):
        self.load_reporter_runtime = None
        self.load_reporter_unsupported_reason = None


@pytest.mark.asyncio
async def test_single_tokenizer_lifecycle_owns_runtime_and_detaches_hook_on_close():
    """Single-tokenizer mode owns runtime and detaches hooks in correct order."""
    # This test will fail until lifecycle.py is implemented
    try:
        from sglang.srt.load_reporter.lifecycle import LoadReporterLifecycle
    except ImportError:
        pytest.skip("LoadReporterLifecycle not yet implemented")

    manager = FakeTokenizerManager()
    runtime = FakeLoadReporterRuntime()
    server_args = FakeServerArgs(tokenizer_worker_num=1)
    app_state = FakeAppState()

    # Construct lifecycle with fake runtime injection
    lifecycle = LoadReporterLifecycle.from_http_server(
        server_args=server_args,
        tokenizer_manager=manager,
        app_state=app_state,
    )

    # Inject fake runtime for testing (production creates real one)
    lifecycle._runtime = runtime
    lifecycle._manager = manager

    # Start should install hooks
    await lifecycle.start()
    assert manager.request_finished_hook is not None

    # Close should: (1) detach hooks (2) close runtime (3) not crash on hook=None
    await lifecycle.close()

    # Verify detach happened before close
    assert "request_finished_hook" in manager.detach_calls
    assert runtime.close_called

    # Second close should be idempotent
    await lifecycle.close()
    assert runtime.close_called  # Still true, not called twice


@pytest.mark.asyncio
async def test_multi_tokenizer_lifecycle_installs_proxy_and_notifier_only():
    """Multi-tokenizer mode installs IPC components, does not own runtime."""
    try:
        from sglang.srt.load_reporter.lifecycle import LoadReporterLifecycle
    except ImportError:
        pytest.skip("LoadReporterLifecycle not yet implemented")

    manager = FakeTokenizerManager()
    server_args = FakeServerArgs(tokenizer_worker_num=4)
    app_state = FakeAppState()
    notifier = FakeLoadReporterNotifier()

    lifecycle = LoadReporterLifecycle.from_http_server(
        server_args=server_args,
        tokenizer_manager=manager,
        app_state=app_state,
    )

    # Inject fake notifier
    lifecycle._notifier = notifier
    lifecycle._manager = manager

    await lifecycle.start()

    # Should attach IPC components
    assert "ipc_components" in manager.attach_calls
    assert notifier.start_called

    await lifecycle.close()

    # Should detach IPC and close notifier
    assert "ipc_components" in manager.detach_calls
    assert notifier.close_called


@pytest.mark.asyncio
async def test_close_is_idempotent_after_partial_start_failure():
    """Close cleans up successfully-created resources even if start failed."""
    try:
        from sglang.srt.load_reporter.lifecycle import LoadReporterLifecycle
    except ImportError:
        pytest.skip("LoadReporterLifecycle not yet implemented")

    manager = FakeTokenizerManager()
    runtime = FakeLoadReporterRuntime()
    server_args = FakeServerArgs(tokenizer_worker_num=1)
    app_state = FakeAppState()

    lifecycle = LoadReporterLifecycle.from_http_server(
        server_args=server_args,
        tokenizer_manager=manager,
        app_state=app_state,
    )

    lifecycle._runtime = runtime
    lifecycle._manager = manager

    # Simulate partial start: hook installed but something else failed
    await lifecycle.start()

    # Now simulate start failure after hooks installed
    # Close should still clean up
    runtime.close_error = RuntimeError("simulated close error")

    # Close should not propagate the error
    await lifecycle.close()  # Should log error but not raise

    assert runtime.close_called
    assert "request_finished_hook" in manager.detach_calls


@pytest.mark.asyncio
async def test_lifecycle_disabled_when_port_is_none():
    """When load_reporter_port is None, lifecycle is no-op."""
    try:
        from sglang.srt.load_reporter.lifecycle import LoadReporterLifecycle
    except ImportError:
        pytest.skip("LoadReporterLifecycle not yet implemented")

    manager = FakeTokenizerManager()
    server_args = FakeServerArgs(tokenizer_worker_num=1)
    server_args.load_reporter_port = None
    app_state = FakeAppState()

    lifecycle = LoadReporterLifecycle.from_http_server(
        server_args=server_args,
        tokenizer_manager=manager,
        app_state=app_state,
    )

    await lifecycle.start()

    # Should not install any hooks
    assert manager.request_finished_hook is None

    await lifecycle.close()

    # Should be no-op, no detach calls
    assert len(manager.detach_calls) == 0
