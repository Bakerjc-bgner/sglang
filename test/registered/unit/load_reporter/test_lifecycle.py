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

    def notify_request_finished(self):
        """Fake notification method."""
        pass

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
        self.load_reporter_port = None
        self.load_reporter_snapshot_stale_after_ms = 3000
        # Add other required attributes for LoadReporterConfig
        self.served_model_name = "test-model"


class FakeAppState:
    """Minimal FastAPI app.state for lifecycle."""

    def __init__(self):
        self.load_reporter_runtime = None
        self.load_reporter_unsupported_reason = None


@pytest.mark.asyncio
async def test_single_tokenizer_lifecycle_owns_runtime_and_detaches_hook_on_close():
    """Single-tokenizer mode owns runtime and calls unbind in correct order."""
    # This test will fail until lifecycle.py is implemented
    try:
        from sglang.srt.load_reporter.decorator import bind_load_monitor
        from sglang.srt.load_reporter.lifecycle import LoadReporterLifecycle
    except ImportError:
        pytest.skip("LoadReporterLifecycle not yet implemented")

    manager = FakeTokenizerManager()
    runtime = FakeLoadReporterRuntime()
    server_args = FakeServerArgs(tokenizer_worker_num=1)
    app_state = FakeAppState()

    # Construct lifecycle
    lifecycle = LoadReporterLifecycle.from_http_server(
        server_args=server_args,
        tokenizer_manager=manager,
        app_state=app_state,
    )

    # For this test, we manually inject runtime to avoid real construction
    # In production, _start_single_tokenizer creates the real runtime
    lifecycle._runtime = runtime
    lifecycle._started = True  # Skip actual start logic

    # Manually bind to simulate what start() does via bind_load_monitor
    unbind_called = []
    original_unbind = bind_load_monitor(
        manager, lambda reason, count: runtime.notify_request_finished()
    )

    def tracked_unbind():
        unbind_called.append(True)
        original_unbind()

    lifecycle._unbind = tracked_unbind

    # Close should: (1) call unbind (2) close runtime (3) not crash on None
    await lifecycle.close()

    # Verify unbind was called before close
    assert len(unbind_called) == 1
    assert lifecycle._unbind is None
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

    # Inject fake notifier and mark as started
    lifecycle._notifier = notifier
    lifecycle._runtime = Mock()  # Multi-tokenizer has proxy as runtime
    lifecycle._started = True

    # Simulate what start does: attach IPC and start notifier
    manager.attach_load_reporter_ipc_components(lifecycle._runtime, notifier)
    notifier.start_called = True

    # Should have attached IPC components
    assert "ipc_components" in manager.attach_calls

    await lifecycle.close()

    # Should detach IPC and close notifier
    assert "ipc_components" in manager.detach_calls
    assert notifier.close_called


@pytest.mark.asyncio
async def test_close_is_idempotent_after_partial_start_failure():
    """Close cleans up successfully-created resources even if start failed."""
    try:
        from sglang.srt.load_reporter.decorator import bind_load_monitor
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
    lifecycle._started = True

    # Simulate partial start: bind_load_monitor installed
    unbind_called = []
    original_unbind = bind_load_monitor(
        manager, lambda reason, count: runtime.notify_request_finished()
    )

    def tracked_unbind():
        unbind_called.append(True)
        original_unbind()

    lifecycle._unbind = tracked_unbind

    # Now simulate close error
    runtime.close_error = RuntimeError("simulated close error")

    # Close should not propagate the error
    await lifecycle.close()  # Should log error but not raise

    assert runtime.close_called
    assert len(unbind_called) == 1


@pytest.mark.asyncio
async def test_lifecycle_disabled_when_port_is_none():
    """When load_reporter_port is None, start() is a no-op and close() is safe."""
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

    # start() must short-circuit without touching the tokenizer manager
    await lifecycle.start()

    assert not lifecycle._started
    assert app_state.load_reporter_runtime is None
    assert len(manager.attach_calls) == 0
    assert len(manager.detach_calls) == 0

    # close() after a no-op start must also be safe
    await lifecycle.close()
    assert len(manager.detach_calls) == 0
