"""Load monitor decorator seam.

Exposes two public symbols only:

    enable_load_monitor(kind)
        Decorator factory. Wraps a synchronous function (``"scheduler_message"``)
        or an async generator (``"request_lifecycle"``) to fire load-monitor events
        without touching the function body.

    bind_load_monitor(owner, notify) -> unbind_closure
        Bind a (reason, count) callback to *owner* via a weak-key registry.
        Returns an idempotent zero-argument closure that removes the binding.

Fast bypass
-----------
Both wrappers check ``self.server_args.load_reporter_port`` first.  When the
attribute is absent or ``None``, the wrapper is a transparent passthrough — no
registry lookup, no event classification, no optional-dependency imports.

Callback contract
-----------------
``notify(reason: LoadReporterRefreshReason, count: int) -> None``
Synchronous and non-throwing from the caller's perspective: any exception the
callback raises is logged and swallowed; it never alters the wrapped function's
return value or propagates its exception.
"""

from __future__ import annotations

import functools
import logging
import weakref
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Callback type: (reason, count) -> None
# reason is a LoadReporterRefreshReason enum value; typed Any here to avoid
# importing io_struct at module load time (keeps disabled overhead to zero).
_NotifyFn = Callable[[Any, int], None]

# Weak-key registry: owner instance -> callback.
# Owner garbage collection automatically removes the entry.
_REGISTRY: weakref.WeakKeyDictionary[Any, _NotifyFn] = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bind_load_monitor(owner: Any, notify: _NotifyFn) -> Callable[[], None]:
    """Bind *notify* to *owner* in the weak-key registry.

    Args:
        owner: The instance that the decorated methods belong to.  Typically a
            ``TokenizerManager``.  Stored as a weak reference — when the owner
            is garbage-collected the entry is removed automatically.
        notify: Synchronous callback ``(reason, count) -> None``.  Must be
            non-blocking and non-throwing from its caller's perspective
            (exceptions are caught by the decorator wrapper).

    Returns:
        An idempotent zero-argument closure that removes the binding.  Safe to
        call multiple times or after the owner has been garbage-collected.
    """
    _REGISTRY[owner] = notify

    def unbind() -> None:
        _REGISTRY.pop(owner, None)

    return unbind


def enable_load_monitor(kind: str) -> Callable[..., Any]:
    """Decorator factory for load-monitor observation points.

    Args:
        kind: One of:

            * ``"scheduler_message"`` — wraps a synchronous void method whose
              first positional argument after *self* is a scheduler payload.
              Fires after a successful return with the classified event.  No
              event is fired on exception.

            * ``"request_lifecycle"`` — wraps an async generator method.
              Fires exactly one ``COMPLETION`` event in the generator's
              ``finally`` block, covering normal exhaustion, early ``aclose()``,
              task cancellation, and unhandled exceptions.

    Returns:
        A single-argument decorator (the function to wrap).

    Raises:
        ValueError: If *kind* is not one of the recognised values.
    """
    if kind == "scheduler_message":
        return _make_scheduler_message_decorator
    if kind == "request_lifecycle":
        return _make_request_lifecycle_decorator
    raise ValueError(
        f"enable_load_monitor: unknown kind {kind!r}. "
        "Expected 'scheduler_message' or 'request_lifecycle'."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_notify(self: Any) -> Optional[_NotifyFn]:
    """Return the bound callback for *self*, or None if absent or disabled."""
    if getattr(getattr(self, "server_args", None), "load_reporter_port", None) is None:
        return None
    return _REGISTRY.get(self)


def _classify_scheduler_obj(obj: Any) -> Optional[tuple[Any, int]]:
    """Return (LoadReporterRefreshReason, count) for a scheduler payload, or None.

    Imports are deferred to this function so that the decorator module can be
    imported without pulling in io_struct or proto dependencies.
    """
    from sglang.srt.managers.io_struct import (
        AbortReq,
        BatchTokenizedEmbeddingReqInput,
        BatchTokenizedGenerateReqInput,
        TokenizedEmbeddingReqInput,
        TokenizedGenerateReqInput,
    )
    from sglang.srt.managers.io_struct import LoadReporterRefreshReason as Reason

    if isinstance(obj, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
        return (Reason.DISPATCH, 1)
    if isinstance(obj, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)):
        return (Reason.DISPATCH, len(obj.batch))
    if isinstance(obj, AbortReq):
        return (Reason.ABORT, 1)
    return None


def _make_scheduler_message_decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a synchronous scheduler-dispatch method."""

    @functools.wraps(fn)
    def _scheduler_wrapper(self: Any, obj: Any, *args: Any, **kwargs: Any) -> Any:
        result = fn(self, obj, *args, **kwargs)  # let exceptions propagate unchanged
        notify = _get_notify(self)
        if notify is None:
            return result
        classified = _classify_scheduler_obj(obj)
        if classified is None:
            return result
        try:
            notify(*classified)
        except Exception:
            logger.exception("Load monitor callback raised on scheduler_message")
        return result

    return _scheduler_wrapper


def _make_request_lifecycle_decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async generator request method to fire COMPLETION on exit."""

    @functools.wraps(fn)
    async def _lifecycle_wrapper(self: Any, *args: Any, **kwargs: Any):
        notify = _get_notify(self)
        if notify is None:
            # Pure passthrough — no try/finally overhead
            async for item in fn(self, *args, **kwargs):
                yield item
            return
        from sglang.srt.managers.io_struct import LoadReporterRefreshReason as Reason

        try:
            async for item in fn(self, *args, **kwargs):
                yield item
        finally:
            try:
                notify(Reason.COMPLETION, 1)
            except Exception:
                logger.exception("Load monitor callback raised on request_lifecycle")

    return _lifecycle_wrapper
