"""Load reporter configuration and worker metadata."""

from __future__ import annotations

from typing import Optional

import msgspec

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb

SHUTDOWN_TIMEOUT_SECONDS = 5.0


def validate_session_timing(
    report_interval_ms: Optional[int] = None,
    lease_ttl_ms: Optional[int] = None,
) -> None:
    """Reject non-positive session timing values."""
    if report_interval_ms is not None and report_interval_ms <= 0:
        raise ValueError("report_interval_ms must be greater than zero")
    if lease_ttl_ms is not None and lease_ttl_ms <= 0:
        raise ValueError("lease_ttl_ms must be greater than zero")


class WorkerMetadata(msgspec.Struct, frozen=True):
    """Stable identity fields reported with every load snapshot."""

    worker_addr: str
    worker_type: int
    model: Optional[str] = None

    @classmethod
    def from_values(
        cls, *, host: str, port: int, disaggregation_mode: str, model: Optional[str]
    ) -> WorkerMetadata:
        """Build worker identity from resolved serving configuration."""
        worker_type = {
            "prefill": pb.WORKER_TYPE_PREFILL,
            "decode": pb.WORKER_TYPE_DECODE,
        }.get(disaggregation_mode, pb.WORKER_TYPE_REGULAR)
        host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        worker_addr = f"{host}:{port}"
        return cls(
            worker_addr=worker_addr,
            worker_type=worker_type,
            model=model,
        )


class ReporterSettings(msgspec.Struct, frozen=True):
    """Internal copy of settings already validated by the launcher."""

    host: str
    port: int
    disaggregation_mode: str
    model: Optional[str]

    def metadata(self):
        return WorkerMetadata.from_values(
            host=self.host,
            port=self.port,
            disaggregation_mode=self.disaggregation_mode,
            model=self.model,
        )
