"""Smoke test: a real DP server publishes complete reports without HTTP polling."""

import socket
import threading
import unittest
from urllib.parse import urlparse

import grpc
import requests

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=60, stage="base-b", runner_config="2-gpu-large")


class TestPubDpServer(DefaultServerBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    server_env = {"SGLANG_LOAD_SNAPSHOT_USE_ZMQ": "1"}

    @classmethod
    def setUpClass(cls):
        for _ in range(100):
            with socket.socket() as reporter, socket.socket() as collector:
                reporter.bind(("127.0.0.1", 0))
                cls.reporter_port = reporter.getsockname()[1]
                if cls.reporter_port == 65535:
                    continue
                try:
                    collector.bind(("127.0.0.1", cls.reporter_port + 1))
                except OSError:
                    continue
                break
        else:
            raise RuntimeError("could not reserve adjacent reporter/collector ports")
        cls.other_args = [
            "--load-reporter-port",
            str(cls.reporter_port),
            "--mem-fraction-static",
            "0.5",
            "--dp-size",
            "2",
            "--load-balance-method",
            "total_tokens",
        ]
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_reports_progress_before_any_http_load_query(self):
        stopped = threading.Event()

        def frames():
            yield pb.RouterFrame(
                register=pb.RegisterRequest(
                    router_id="serving-test",
                    report_interval_ms=100,
                    lease_ttl_ms=10_000,
                )
            )
            stopped.wait(15)

        address = f"{urlparse(self.base_url).hostname}:{self.reporter_port}"
        with grpc.insecure_channel(address) as channel:
            call = pb_grpc.LoadMonitorServiceStub(channel).Monitor(frames(), timeout=10)
            reports = []
            try:
                for frame in call:
                    if frame.WhichOneof("payload") != "report":
                        continue
                    report = frame.report
                    # Listener readiness deliberately does not wait for the
                    # first complete instance view.
                    if not reports and report.status == pb.REPORT_STATUS_UNREACHABLE:
                        continue
                    self.assertEqual(
                        report.status, pb.REPORT_STATUS_HEALTHY, report.last_error
                    )
                    self.assertEqual([r.dp_rank for r in report.ranks], [0, 1])
                    self.assertTrue(
                        all(r.max_total_num_tokens > 0 for r in report.ranks)
                    )
                    reports.append(report)
                    # Unchanged loads are PUB heartbeats (up to one second),
                    # so five 100ms gRPC reports need not contain a new sample.
                    if len(reports) >= 5 and all(
                        after.snapshot_time_unix_ms > before.snapshot_time_unix_ms
                        for before, after in zip(reports[0].ranks, reports[-1].ranks)
                    ):
                        break
            finally:
                stopped.set()
                call.cancel()
        self.assertGreaterEqual(len(reports), 5)
        self.assertGreater(reports[-1].sequence_id, reports[0].sequence_id)
        for before, after in zip(reports[0].ranks, reports[-1].ranks):
            self.assertGreater(
                after.snapshot_time_unix_ms, before.snapshot_time_unix_ms
            )
        info = requests.get(f"{self.base_url}/server_info", timeout=10).json()
        self.assertEqual(info["load_reporter"]["port"], self.reporter_port)


if __name__ == "__main__":
    unittest.main()
