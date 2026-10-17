"""Pure report builder for the embedded load reporter."""

from __future__ import annotations

from typing import Optional, Sequence

from sglang.srt.load_reporter.config import WorkerMetadata
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb


class ReportBuilder:
    """Assemble validated rank messages into protocol load reports."""

    def __init__(self, source_instance_id: str) -> None:
        """Initialize report identity and sequence allocation."""
        self._source_instance_id = source_instance_id
        self._sequence_id = 0

    def _next_sequence_id(self) -> int:
        """Return the next positive process-local sequence ID."""
        self._sequence_id += 1
        return self._sequence_id

    def build(
        self,
        ranks: Sequence[pb.RankLoad],
        identity: WorkerMetadata,
        *,
        report_time_unix_ms: int,
    ) -> pb.LoadReport:
        """Report validated ranks unchanged; the router owns freshness policy."""
        if not ranks:
            raise ValueError("a healthy report requires at least one rank")
        return self._new_report(
            ranks=ranks,
            identity=identity,
            report_time_unix_ms=report_time_unix_ms,
            status=pb.REPORT_STATUS_HEALTHY,
            error=None,
        )

    def build_unreachable(
        self,
        identity: WorkerMetadata,
        *,
        report_time_unix_ms: int,
        error: BaseException | str,
    ) -> pb.LoadReport:
        """Build an empty report for an unavailable or invalid instance view."""
        message = str(error).strip() or "load snapshot unavailable"
        return self._new_report(
            ranks=(),
            identity=identity,
            report_time_unix_ms=report_time_unix_ms,
            status=pb.REPORT_STATUS_UNREACHABLE,
            error=message,
        )

    def _new_report(
        self,
        *,
        ranks: Sequence[pb.RankLoad],
        identity: WorkerMetadata,
        report_time_unix_ms: int,
        status: int,
        error: Optional[str],
    ) -> pb.LoadReport:
        """Allocate one protobuf report; success and failure share this path."""
        report = pb.LoadReport(
            source_instance_id=self._source_instance_id,
            sequence_id=self._next_sequence_id(),
            report_time_unix_ms=report_time_unix_ms,
            worker=pb.Worker(
                worker_addr=identity.worker_addr,
                worker_type=identity.worker_type,
            ),
            status=status,
            ranks=ranks,
        )
        if identity.model is not None:
            report.worker.model = identity.model
        if error is not None:
            report.last_error = error
        return report
