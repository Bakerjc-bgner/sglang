"""LoadReporterLifecycle facade for serving entrypoints.

This module encapsulates all load reporter initialization, hook installation,
and cleanup logic. Serving entrypoints (http_server.py, grpc_server.py,
engine.py) only interact with this facade, never directly with runtime,
sampler, IPC components, or gRPC dependencies.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from starlette.datastructures import State

    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class LoadReporterLifecycle:
    """Facade that owns load reporter creation, startup, and shutdown.

    Serving entrypoints create one lifecycle instance per process, call
    ``start()`` to activate reporting, and call ``close()`` during shutdown.
    The lifecycle determines whether to own a full runtime (single-tokenizer)
    or only install IPC components (multi-tokenizer worker).
    """

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        tokenizer_manager: TokenizerManager,
        app_state: State,
    ) -> None:
        """Initialize lifecycle (does not start services).

        Args:
            server_args: SGLang server configuration.
            tokenizer_manager: The tokenizer manager to attach hooks to.
            app_state: FastAPI app.state for storing runtime reference.
        """
        self._server_args = server_args
        self._manager = tokenizer_manager
        self._app_state = app_state
        self._runtime: Optional[Any] = None
        self._notifier: Optional[Any] = None
        self._started = False
        self._closed = False

    @classmethod
    def from_http_server(
        cls,
        *,
        server_args: ServerArgs,
        tokenizer_manager: TokenizerManager,
        app_state: State,
    ) -> LoadReporterLifecycle:
        """Factory for HTTP server lifespan.

        Args:
            server_args: SGLang server configuration.
            tokenizer_manager: The tokenizer manager instance.
            app_state: FastAPI app.state.

        Returns:
            A configured LoadReporterLifecycle instance.
        """
        return cls(
            server_args=server_args,
            tokenizer_manager=tokenizer_manager,
            app_state=app_state,
        )

    async def start(self) -> None:
        """Start load reporter and install hooks.

        Returns immediately if ``server_args.load_reporter_port`` is ``None``.
        Single-tokenizer mode creates a runtime and installs request hooks.
        Multi-tokenizer mode installs IPC proxy/notifier and starts the notifier.

        This method is idempotent.
        """
        if self._server_args.load_reporter_port is None:
            return

        if self._started or self._closed:
            return

        self._started = True

        if self._server_args.tokenizer_worker_num == 1:
            await self._start_single_tokenizer()
        else:
            await self._start_multi_tokenizer()

    async def _start_single_tokenizer(self) -> None:
        """Start single-tokenizer runtime."""
        try:
            from sglang.srt.load_reporter import describe_optional_dependency_error
            from sglang.srt.load_reporter.runtime import LoadReporterRuntime
            from sglang.srt.load_reporter.sampler import (
                TokenizerManagerLoadSnapshotSource,
            )
        except (ModuleNotFoundError, RuntimeError) as exc:
            unsupported_reason = describe_optional_dependency_error(exc)
            if unsupported_reason is None:
                raise
            self._app_state.load_reporter_unsupported_reason = unsupported_reason
            logger.info(
                "Load reporter disabled because optional dependencies are unavailable: %s",
                unsupported_reason,
            )
            return

        snapshot_source = TokenizerManagerLoadSnapshotSource(self._manager)
        self._runtime = LoadReporterRuntime(
            snapshot_source,
            self._server_args,
            active_changed=lambda active: logger.info(
                "Load reporter active=%s",
                active,
            ),
        )
        self._manager.set_load_reporter_request_finished_hook(
            self._runtime.notify_request_finished
        )
        self._app_state.load_reporter_unsupported_reason = None
        self._app_state.load_reporter_runtime = self._runtime

    async def _start_multi_tokenizer(self) -> None:
        """Start multi-tokenizer IPC components."""
        from sglang.srt.load_reporter.ipc import (
            LoadReporterControlProxy,
            LoadReporterRefreshNotifier,
        )

        proxy = LoadReporterControlProxy(self._manager._dispatch_to_scheduler)
        notifier = LoadReporterRefreshNotifier(
            worker_id=f"http-worker-{os.getpid()}",
            send=self._manager._dispatch_to_scheduler,
        )
        self._manager.attach_load_reporter_ipc_components(proxy, notifier)
        self._manager.set_load_reporter_request_event_hook(notifier.notify)
        await notifier.start()

        self._runtime = proxy
        self._notifier = notifier
        self._app_state.load_reporter_unsupported_reason = None
        self._app_state.load_reporter_runtime = proxy

    async def close(self) -> None:
        """Bounded, idempotent shutdown.

        Detaches hooks before closing runtime/notifier. Individual component
        failures are logged but do not prevent cleanup of other resources.
        """
        if self._closed:
            return

        self._closed = True

        # Step 1: Detach hooks so no late request can wake torn-down components
        if self._runtime is not None:
            self._manager.set_load_reporter_request_finished_hook(None)
            self._manager.set_load_reporter_request_event_hook(None)

        # Step 2: Close runtime (single-tokenizer) or notifier (multi-tokenizer)
        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception:
                logger.exception("Load reporter runtime shutdown failed")

        # Step 3: Close notifier if multi-tokenizer
        if self._notifier is not None:
            try:
                await self._notifier.close()
            except Exception:
                logger.exception("Load reporter notifier shutdown failed")
            # Detach IPC components
            self._manager.attach_load_reporter_ipc_components(None, None)
