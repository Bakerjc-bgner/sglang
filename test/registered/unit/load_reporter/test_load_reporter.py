"""CPU load reporter tests: protocol, scheduling, lifecycle and real IPC."""

import asyncio
import fcntl
import os
import queue
import socket
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future, TimeoutError
from contextlib import contextmanager
from unittest import mock
from urllib.parse import urlsplit

import grpc
import msgspec
import zmq

import sglang.srt.runtime_context as rc
from sglang.srt.arg_groups.validation_hook import check_load_reporter_args
from sglang.srt.environ import envs
from sglang.srt.load_reporter import (
    close_load_reporter,
    describe_load_reporter,
    init_load_reporter,
    is_load_reporter_enabled,
    is_reporter_host,
    start_load_reporter,
)
from sglang.srt.load_reporter.config import ReporterSettings, WorkerMetadata
from sglang.srt.load_reporter.ingress import (
    CollectorConfig,
    collector_config_from_runtime,
)
from sglang.srt.load_reporter.lifecycle import (
    CollectorDriver,
    LoadReporterHandle,
    ReporterTransport,
)
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc
from sglang.srt.load_reporter.pub_source import ZmqPubSnapshotSource
from sglang.srt.load_reporter.publisher_protocol import metadata_for_snapshot
from sglang.srt.load_reporter.report_builder import ReportBuilder
from sglang.srt.load_reporter.runtime import LoadReporterRuntime
from sglang.srt.load_reporter.schedule import ReportSchedule
from sglang.srt.load_reporter.service import LoadMonitorService
from sglang.srt.load_reporter.snapshot_source import SnapshotCache
from sglang.srt.load_reporter.snapshot_validation import validate_full_snapshot
from sglang.srt.managers.load_snapshot import (
    LoadSnapshot,
    ShmLoadSnapshotReader,
    ZmqLoadSnapshotWriter,
    ZmqShmLoadSnapshotReader,
    snapshot_decoder,
    snapshot_encoder,
)
from sglang.srt.managers.scheduler_components.load_publisher import (
    LoadStat,
    SchedulerLoadPublisher,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def sample(rank=0, timestamp=100.0, running=2):
    return LoadSnapshot(
        dp_rank=rank,
        timestamp=timestamp,
        num_running_reqs=running,
        max_running_requests=16,
        max_total_num_tokens=1000,
    )


def collector_config(dp_size=2, collector_port=None):
    return CollectorConfig.from_values(
        reporter_port=(
            collector_port - 1
            if collector_port is not None
            else free_port(adjacent=True)
        ),
        dist_init_addr="127.0.0.1:29500",
        nnodes=1,
        dp_size=dp_size,
    )


def reporter_fields(config):
    return dict(
        load_reporter_port=urlsplit(config.endpoint).port - 1,
        dist_init_addr="127.0.0.1:29500",
    )


def pub_frames(snapshot, config, *, boot="publisher-1", sequence=0):
    stat = LoadStat(
        num_running_reqs=snapshot.num_running_reqs,
        num_waiting_reqs=snapshot.num_waiting_reqs,
        num_tokens=snapshot.num_used_tokens,
        max_total_num_tokens=snapshot.max_total_num_tokens,
        attn_dp_rank=snapshot.dp_rank,
        num_waiting_uncached_tokens=snapshot.num_waiting_uncached_tokens,
        num_total_tokens=snapshot.num_total_tokens,
        max_running_requests=snapshot.max_running_requests,
        reporter_metadata=metadata_for_snapshot(snapshot, config, boot),
    )
    return [b"load", sequence.to_bytes(8, "big"), msgspec.msgpack.encode(stat)]


class TestPublisherProtocol(CustomTestCase):
    def test_non_publisher_ranks_skip_connecting(self):
        """Replicated TP/CP/PP ranks must not publish a duplicate rank stream."""
        for rank, topology in (
            ("pp_rank", dict(pp_size=2)),
            ("attn_tp_rank", dict(tp_size=2)),
            ("attn_cp_rank", dict(tp_size=2, attn_cp_size=2)),
        ):
            with (
                self.subTest(rank=rank),
                get_context().override_server_args(
                    dp_size=1, nnodes=1, load_reporter_port=30100, **topology
                ),
                rc.get_parallel().override(
                    **{**dict(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0), rank: 1}
                ),
            ):
                # Non-publishing ranks must not create a duplicate rank stream
                # even when the reporter is enabled.
                publisher = SchedulerLoadPublisher(kv_events_config=None)
                try:
                    self.assertFalse(publisher.enable)
                finally:
                    publisher.close()

    def test_producer_suffix_preserves_legacy_decoder_and_source_metrics(self):
        class LegacyLoadStat(msgspec.Struct, array_like=True, tag="LoadStat"):
            num_running_reqs: int
            num_waiting_reqs: int
            num_tokens: int
            max_total_num_tokens: int
            attn_dp_rank: int | None = None

        config = collector_config(dp_size=1)
        captured = []
        socket_boundary = mock.Mock()
        socket_boundary.send_multipart.side_effect = captured.append
        with (
            get_context().override_server_args(
                dp_size=1, nnodes=1, **reporter_fields(config)
            ),
            rc.get_parallel().override(
                dp_rank=0, pp_rank=0, attn_dp_rank=0, attn_tp_rank=0, attn_cp_rank=0
            ),
            mock.patch(
                "sglang.srt.load_reporter.publisher.open_publisher_socket",
                return_value=socket_boundary,
            ),
        ):
            publisher = SchedulerLoadPublisher(kv_events_config=None)
        try:
            snapshot = sample(timestamp=123.125)
            snapshot.token_usage, snapshot.gen_throughput = 0.25, 17.5
            snapshot.cache_hit_rate, snapshot.utilization = 0.5, 0.75
            publisher.publish_load_stat(lambda: snapshot, force=True)
            old = msgspec.msgpack.decode(captured[0][2], type=LegacyLoadStat)
            self.assertEqual((old.num_running_reqs, old.attn_dp_rank), (2, 0))
            source = ZmqPubSnapshotSource(config)
            source.accept(captured[0])
            self.assertEqual(source.snapshot(), [snapshot])
            snapshot.gen_throughput = 18.5
            publisher.publish_load_stat(lambda: snapshot, force=True)
            self.assertEqual(len(captured), 2)
        finally:
            publisher.close()

    def test_pub_sequence_restart_and_source_time_contract(self):
        """Rank/instance validation precedes boot replacement on a shared socket."""
        config = collector_config(dp_size=1)
        source = ZmqPubSnapshotSource(config)
        original = sample(timestamp=100.125, running=3)
        source.accept(pub_frames(original, config, sequence=7))
        source.accept(pub_frames(sample(timestamp=110), config, sequence=6))
        self.assertEqual(source.snapshot(), [original])
        source.accept(pub_frames(original, config, sequence=8))
        self.assertEqual(source.snapshot()[0].timestamp, 100.125)
        replacement = sample(timestamp=120.25, running=9)
        source.accept(pub_frames(replacement, config, boot="publisher-2"))
        source.accept(pub_frames(sample(timestamp=130), config, sequence=99))
        self.assertEqual(source.snapshot(), [replacement])
        frames = pub_frames(original, config)
        stat = msgspec.msgpack.decode(frames[2], type=LoadStat)
        invalid_stats = [
            msgspec.structs.replace(stat, reporter_metadata=None),
            msgspec.structs.replace(stat, attn_dp_rank=None),
            msgspec.structs.replace(stat, attn_dp_rank=-1),
            msgspec.structs.replace(stat, attn_dp_rank=1),
            msgspec.structs.replace(
                stat,
                num_running_reqs=-1,
                reporter_metadata=msgspec.structs.replace(
                    stat.reporter_metadata, publisher_boot_id="invalid-boot"
                ),
            ),
        ]
        for field, value in (
            ("snapshot_timestamp", 0),
            ("instance_id", "another-instance"),
            ("topology_generation", "another-rank-set"),
            ("publisher_boot_id", ""),
            ("version", 2),
        ):
            invalid_stats.append(
                msgspec.structs.replace(
                    stat,
                    reporter_metadata=msgspec.structs.replace(
                        stat.reporter_metadata, **{field: value}
                    ),
                )
            )
        for invalid in invalid_stats:
            source.accept([b"load", bytes(8), msgspec.msgpack.encode(invalid)])
        source.accept([b"load", bytes(1), frames[2]])
        self.assertEqual(source.stats["invalid"], len(invalid_stats) + 1)
        self.assertEqual(source.snapshot(), [replacement])


class TestLoadReportContract(CustomTestCase):
    def test_wire_contract(self):
        service = pb.DESCRIPTOR.services_by_name["LoadMonitorService"]
        self.assertEqual(
            service.full_name, "sglang.router.loadmonitor.v1.LoadMonitorService"
        )
        self.assertEqual(list(service.methods_by_name), ["Monitor"])
        self.assertTrue(service.methods[0].client_streaming)
        self.assertTrue(service.methods[0].server_streaming)
        self.assertEqual(
            {f.number: f.name for f in pb.LoadReport.DESCRIPTOR.fields},
            {
                1: "source_instance_id",
                2: "sequence_id",
                3: "report_time_unix_ms",
                4: "worker",
                5: "status",
                6: "last_error",
                7: "ranks",
            },
        )
        config = pb.UpdateConfigRequest.FromString(b"\x08\x00")
        self.assertTrue(config.HasField("report_interval_ms"))
        self.assertFalse(config.HasField("lease_ttl_ms"))

    def test_invalid_snapshots_are_rejected(self):
        for loads in ([sample(0)], [sample(0), sample(0)], [sample(0), sample(2)]):
            with self.subTest(loads=loads), self.assertRaises(ValueError):
                validate_full_snapshot(loads, dp_size=2)
        # Test reporter semantics; primitive types and integer ranges are
        # enforced by the shared decoder and protobuf runtime.
        for field, value in (
            ("timestamp", 0),
            ("timestamp", float("nan")),
            ("num_running_reqs", -1),
            ("num_running_reqs", 17),
            ("num_used_tokens", 1001),
            ("token_usage", float("nan")),
        ):
            invalid = sample()
            setattr(invalid, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_full_snapshot([invalid], dp_size=1)

    def test_decoded_snapshot_maps_only_protocol_fields_and_is_copied(self):
        source = LoadSnapshot(
            timestamp=123.125,
            num_running_reqs=3,
            num_waiting_reqs=4,
            num_waiting_uncached_tokens=5,
            num_used_tokens=6,
            num_total_tokens=7,
            max_total_num_tokens=100,
            max_running_requests=10,
            token_usage=0.5,
            gen_throughput=12.5,
            cache_hit_rate=0.25,
            utilization=0.75,
            # Internal snapshot fields have no counterpart in the Monitor wire.
            num_active_tokens=6,
            total_prefill_uncached_tokens=99,
            decode_moments=[1.0, 2.0],
        )
        decoded = snapshot_decoder.decode(snapshot_encoder.encode(source))
        ranks = validate_full_snapshot([decoded], dp_size=1)
        report = ReportBuilder("boot").build(
            ranks,
            WorkerMetadata("worker:1", pb.WORKER_TYPE_REGULAR),
            report_time_unix_ms=123_200,
        )
        expected = pb.RankLoad(
            dp_rank=0,
            snapshot_time_unix_ms=123_125,
            num_running_reqs=3,
            num_waiting_reqs=4,
            num_waiting_uncached_tokens=5,
            num_used_tokens=6,
            num_total_tokens=7,
            max_total_num_tokens=100,
            max_running_requests=10,
            token_usage=0.5,
            gen_throughput=12.5,
            cache_hit_rate=0.25,
            utilization=0.75,
        )
        decoded.num_running_reqs = 9
        ranks[0].num_running_reqs = 8
        wire = pb.LoadReport.FromString(report.SerializeToString())
        self.assertEqual(wire.ranks[0], expected)
        self.assertEqual(wire.status, pb.REPORT_STATUS_HEALTHY)


class _Source:
    """In-memory input for report-clock and gRPC session tests."""

    def __init__(self):
        self.dp_size = 1
        self.cache = SnapshotCache(1)

    def start(self):
        pass

    async def aclose(self):
        pass

    def snapshot(self):
        return self.cache.snapshot()


class _Sink:
    def __init__(self):
        self.deliveries = []

    def offer(self, due, report, expired):
        self.deliveries.append((due, report, expired))


class TestBroadcastSchedule(CustomTestCase):
    def test_late_ticks_and_nonleader_updates_preserve_shared_phase(self):
        schedule = ReportSchedule()
        schedule.add(1, 200, 2000, now=0.03)
        schedule.add(2, 50, 2000, now=0.1)
        self.assertEqual(schedule.take_due(0.22), (False, (), ()))
        self.assertEqual(schedule.take_due(0.23), (True, (1, 2), ()))
        self.assertEqual(schedule.take_due(0.95), (True, (1, 2), ()))
        self.assertAlmostEqual(schedule.next_deadline(), 1.03)
        self.assertEqual(schedule.take_due(0.95), (False, (), ()))
        schedule.update(2, interval_ms=50, now=0.96)
        schedule.update(1, interval_ms=200, now=0.97)
        schedule.update(2, now=0.98)  # keepalive
        self.assertAlmostEqual(schedule.next_deadline(), 1.03)
        self.assertEqual(schedule.take_due(1.03), (True, (1, 2), ()))

    def test_leader_update_disconnect_and_new_first_session(self):
        schedule = ReportSchedule()
        schedule.add(1, 200, 2000, now=0)
        schedule.add(2, 200, 2000, now=0.03)
        schedule.update(1, interval_ms=50, now=0.21)
        self.assertAlmostEqual(schedule.next_deadline(), 0.26)
        self.assertEqual(schedule.take_due(0.26), (True, (1, 2), ()))
        schedule.remove(1)
        self.assertAlmostEqual(schedule.next_deadline(), 0.31)
        schedule.update(2, interval_ms=400, now=0.28)
        self.assertEqual(schedule.take_due(0.68), (True, (2,), ()))
        schedule.remove(2)
        self.assertEqual(schedule.next_deadline(), float("inf"))
        self.assertEqual(schedule.take_due(0.8), (False, (), ()))
        schedule.add(3, 250, 2000, now=0.83)
        self.assertAlmostEqual(schedule.next_deadline(), 1.08)
        self.assertEqual(schedule.take_due(1.08), (True, (3,), ()))

    def test_expiry_preserves_live_clock_and_restarts_after_last_session(self):
        schedule = ReportSchedule()
        schedule.add(1, 500, 200, now=0)
        schedule.add(2, 500, 1000, now=0.07)
        self.assertFalse(schedule.update(1, now=0.2))
        self.assertEqual(schedule.take_due(0.2), (False, (), (1,)))
        self.assertAlmostEqual(schedule.next_deadline(), 0.5)
        self.assertEqual(schedule.take_due(0.5), (True, (2,), ()))
        # An expired stream awaiting cleanup cannot set the new session's clock.
        schedule.add(3, 250, 1000, now=1.08)
        self.assertEqual(schedule.take_due(1.08), (False, (), (2,)))
        self.assertAlmostEqual(schedule.next_deadline(), 1.33)
        self.assertEqual(schedule.take_due(1.33), (True, (3,), ()))
        self.assertEqual(schedule.take_due(2.08), (False, (), (3,)))
        self.assertEqual(schedule.next_deadline(), float("inf"))
        self.assertEqual(schedule.take_due(2.18), (False, (), ()))

    def test_missing_rank_does_not_become_healthy(self):
        source, sink = _Source(), _Sink()
        source.dp_size = 2
        source.cache.update([sample(timestamp=time.time())])
        schedule = ReportSchedule()
        schedule.add(1, 100, 1000, now=0)
        driver = CollectorDriver(
            source, schedule, sink, WorkerMetadata("worker:1", pb.WORKER_TYPE_DECODE)
        )
        driver.run_due(0.1)
        report = sink.deliveries[0][1]
        self.assertEqual(report.status, pb.REPORT_STATUS_UNREACHABLE)
        self.assertEqual(list(report.ranks), [])
        self.assertIn("missing=[1]", report.last_error)


class TestMonitorSessions(CustomTestCase):
    def test_lease_expiry_cancels_an_inflight_blocked_write(self):
        """Lease expiry releases a blocked registration ACK as well as a report."""

        async def check(blocked_frame):
            schedule = ReportSchedule()
            runtime = LoadReporterRuntime(schedule)
            registered, writing, cancelled = (
                asyncio.Event(),
                asyncio.Event(),
                asyncio.Event(),
            )

            class Context:
                async def write(self, frame):
                    if frame.WhichOneof("payload") == blocked_frame:
                        writing.set()
                        try:
                            await asyncio.Event().wait()
                        finally:
                            cancelled.set()
                    else:
                        registered.set()

            async def frames():
                yield pb.RouterFrame(
                    register=pb.RegisterRequest(
                        router_id="slow", report_interval_ms=100, lease_ttl_ms=1000
                    )
                )
                await asyncio.Event().wait()

            task = asyncio.create_task(
                LoadMonitorService(runtime).Monitor(frames(), Context())
            )
            try:
                if blocked_frame == "report":
                    await asyncio.wait_for(registered.wait(), 1)
                    runtime.offer(
                        tuple(runtime._sessions), pb.LoadReport(sequence_id=1)
                    )
                await asyncio.wait_for(writing.wait(), 1)
                _, _, expired = schedule.take_due(time.monotonic() + 2)
                runtime.offer((), None, expired)
                done, _ = await asyncio.wait({task}, timeout=0.2)
                self.assertIn(
                    task, done, "lease expiry left a blocked network write alive"
                )
                self.assertTrue(cancelled.is_set())
                self.assertEqual(runtime._sessions, {})
            finally:
                task.cancel()
                await task
                runtime.close()

        for blocked_frame in ("registered", "report"):
            with self.subTest(blocked_frame=blocked_frame):
                asyncio.run(check(blocked_frame))

    def test_latest_mailboxes_replacement_and_expiry(self):
        async def check():
            schedule = ReportSchedule()
            runtime = LoadReporterRuntime(schedule)
            try:
                _, old = runtime.register_session("a", 100, 1000)
                _, fast = runtime.register_session("b", 100, 1000)
                _, replacement = runtime.register_session("a", 100, 1000)
                old.stop()
                for sequence in (1, 2):
                    runtime.offer(
                        (old.token, fast.token, replacement.token),
                        pb.LoadReport(sequence_id=sequence),
                    )
                self.assertEqual(
                    (fast.queue.qsize(), replacement.queue.qsize()), (1, 1)
                )
                await asyncio.sleep(0)
                self.assertEqual((await fast.queue.get()).sequence_id, 2)
                self.assertEqual(replacement.queue.qsize(), 1)
                self.assertEqual((await replacement.queue.get()).sequence_id, 2)
                _, _, expired = schedule.take_due(time.monotonic() + 2)
                runtime.offer((), None, expired)
                await asyncio.sleep(0)
                self.assertIsNone(await replacement.queue.get())
                self.assertIsNone(await fast.queue.get())
                self.assertEqual(runtime._sessions, {})
            finally:
                runtime.close()

        asyncio.run(check())

    def test_grpc_broadcast_late_join_update_stop_and_invalid_first_frame(self):
        requests, late_requests = queue.Queue(), queue.Queue()
        late_call = None

        def frames(requests):
            while (frame := requests.get()) is not None:
                yield frame

        source = _Source()
        source.cache.update([sample(timestamp=time.time(), running=9)])
        transport = ReporterTransport(
            lambda: source, WorkerMetadata("127.0.0.1:0", pb.WORKER_TYPE_REGULAR)
        )
        try:
            with grpc.insecure_channel(f"127.0.0.1:{transport.bound_port}") as channel:
                stub = pb_grpc.LoadMonitorServiceStub(channel)
                rejected = stub.Monitor(
                    iter([pb.RouterFrame(keep_alive=pb.KeepAlive())]), timeout=5
                )
                self.assertEqual(next(rejected).error.code, "INVALID_FIRST_FRAME")
                requests.put(
                    pb.RouterFrame(
                        register=pb.RegisterRequest(
                            router_id="router",
                            report_interval_ms=100,
                            lease_ttl_ms=5000,
                        )
                    )
                )
                call = stub.Monitor(frames(requests), timeout=5)
                self.assertEqual(next(call).registered.lease_ttl_ms, 5000)
                first = next(call).report
                self.assertEqual(first.status, pb.REPORT_STATUS_HEALTHY)
                self.assertEqual(first.ranks[0].num_running_reqs, 9)
                late_requests.put(
                    pb.RouterFrame(
                        register=pb.RegisterRequest(
                            router_id="late-router",
                            report_interval_ms=100,
                            lease_ttl_ms=5000,
                        )
                    )
                )
                late_call = stub.Monitor(frames(late_requests), timeout=5)
                self.assertEqual(next(late_call).registered.lease_ttl_ms, 5000)
                broadcast = next(late_call).report
                while (
                    matching := next(call).report
                ).sequence_id < broadcast.sequence_id:
                    pass
                self.assertEqual(matching, broadcast)
                self.assertGreater(broadcast.sequence_id, first.sequence_id)
                requests.put(
                    pb.RouterFrame(
                        update_config=pb.UpdateConfigRequest(report_interval_ms=0)
                    )
                )
                while (error_frame := next(call)).WhichOneof("payload") == "report":
                    pass
                self.assertEqual(error_frame.error.code, "INVALID_ARGUMENT")
                with self.assertRaises(StopIteration):
                    next(call)
                stopped = stub.Monitor(
                    iter(
                        [
                            pb.RouterFrame(
                                register=pb.RegisterRequest(
                                    router_id="stop",
                                    report_interval_ms=100,
                                    lease_ttl_ms=5000,
                                )
                            ),
                            pb.RouterFrame(stop=pb.StopRequest()),
                        ]
                    ),
                    timeout=5,
                )
                self.assertEqual(next(stopped).WhichOneof("payload"), "registered")
                self.assertTrue(
                    all(f.WhichOneof("payload") == "report" for f in stopped)
                )
        finally:
            requests.put(None)
            late_requests.put(None)
            if late_call is not None:
                late_call.cancel()
            transport.close()
            transport.close()
        self.assertFalse(transport._thread.is_alive())


class TestReporterConfiguration(CustomTestCase):
    def test_ingress_resolution_is_shared_by_nodes(self):
        """All nodes derive the same address and wire identity from existing flags."""
        cases = (
            (None, 1, "tcp://127.0.0.1:30101", "tcp://*:30101"),
            ("leader:29500", 2, "tcp://leader:30101", "tcp://*:30101"),
            ("[::1]:29500", 2, "tcp://[::1]:30101", "tcp://[::]:30101"),
        )
        for dist_addr, nnodes, expected, bind in cases:
            configs = []
            for node in range(nnodes):
                with (
                    self.subTest(dist_addr=dist_addr, node=node),
                    get_context().override_server_args(
                        load_reporter_port=30100,
                        dist_init_addr=dist_addr,
                        nnodes=nnodes,
                        node_rank=node,
                        tp_size=nnodes,
                    ),
                ):
                    config = collector_config_from_runtime()
                    self.assertEqual(config.endpoint, expected)
                    self.assertEqual(config.bind_address(), bind)
                    configs.append(config)
            self.assertTrue(all(config == configs[0] for config in configs))

    def test_disabled_reporter_skips_ingress_validation_and_advertisement(self):
        """An unset port must bypass reporting even in an unsupported topology."""
        fields = dict(
            model_path="dummy",
            load_reporter_port=None,
            nnodes=2,
            tp_size=2,
            dist_init_addr=None,
            kv_events_config='{"publisher": "zmq", "endpoint": "tcp://*:5557"}',
            load_publish_endpoint="auto",
        )
        with (
            envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.override(False),
            get_context().override_server_args(**fields),
            mock.patch.dict(
                sys.modules,
                {
                    "sglang.srt.load_reporter.ingress": None,
                },
            ),
        ):
            check_load_reporter_args(ServerArgs(**fields))
            self.assertFalse(is_load_reporter_enabled())
            self.assertIsNone(describe_load_reporter(ServerArgs(**fields)))

    def test_reporter_requires_push_and_selects_one_instance_host(self):
        fields = dict(
            model_path="dummy",
            dp_size=1,
            nnodes=1,
            **reporter_fields(collector_config(1)),
        )
        with envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.override(False):
            with self.assertRaisesRegex(ValueError, "internal PUSH"):
                check_load_reporter_args(ServerArgs(**fields))
        with envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.override(True):
            check_load_reporter_args(ServerArgs(**fields))
            with self.assertRaises(ValueError):
                check_load_reporter_args(
                    ServerArgs(**{**fields, "load_publish_endpoint": "tcp://*:6000"})
                )
            for overrides, expected in (
                ({"dp_size": 2, "load_balance_method": "total_requests"}, False),
                ({"dp_size": 2, "tokenizer_worker_num": 2}, True),
                ({"node_rank": 1}, None),
                ({"load_reporter_port": None}, None),
            ):
                with get_context().override_server_args(**{**fields, **overrides}):
                    hosts = {
                        parent
                        for parent in (False, True)
                        if is_reporter_host(http_parent=parent)
                    }
                    self.assertEqual(
                        hosts, {expected} if expected is not None else set()
                    )
        with envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.override(False):
            check_load_reporter_args(
                ServerArgs(**{**fields, "attn_dp_size": 2, "tp_size": 2, "nnodes": 2})
            )

    def test_invalid_startup_settings(self):
        fields = dict(model_path="dummy", load_reporter_port=30100)
        for overrides in (
            {"load_reporter_port": 0},
            {"load_reporter_port": 65536},
            {"load_reporter_port": 65535},  # derived collector port overflows
            {"port": 30100},
            {"grpc_port": 30100},
            {"nccl_port": 30100},
            {"port": 30101},
            {"grpc_port": 30101},
            {"nccl_port": 30101},
            {"dist_init_addr": "leader:30101"},
            {"nnodes": 2, "tp_size": 2},  # no leader address
            *(
                {"dist_init_addr": address}
                for address in ("*:29500", "0.0.0.0:29500", "[::]:29500")
            ),
        ):
            with (
                self.subTest(overrides=overrides),
                envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.override(True),
                self.assertRaises(ValueError),
            ):
                check_load_reporter_args(ServerArgs(**{**fields, **overrides}))


def free_port(*, adjacent=False):
    for _ in range(100):
        with socket.socket() as sock, socket.socket() as neighbor:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            if adjacent:
                if port == 65535:
                    continue
                try:
                    neighbor.bind(("127.0.0.1", port + 1))
                except OSError:
                    continue
            return port
    raise RuntimeError("could not reserve adjacent reporter/collector ports")


def load(rank, value):
    return LoadSnapshot(
        dp_rank=rank,
        timestamp=time.time(),
        num_running_reqs=value,
        max_running_requests=100,
    )


async def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("observable state did not converge")


class Client:
    def __init__(self, port, interval=100):
        self.requests = queue.Queue()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.requests.put(
            pb.RouterFrame(
                register=pb.RegisterRequest(
                    router_id="cpu-client",
                    report_interval_ms=interval,
                    lease_ttl_ms=20_000,
                )
            )
        )
        self.call = pb_grpc.LoadMonitorServiceStub(self.channel).Monitor(
            self._frames(), timeout=15
        )

    def _frames(self):
        while (frame := self.requests.get()) is not None:
            yield frame

    async def report(self, predicate):
        for _ in range(100):
            frame = await asyncio.wait_for(
                asyncio.to_thread(next, self.call), timeout=5
            )
            if frame.WhichOneof("payload") == "report" and predicate(frame.report):
                return frame.report
        raise AssertionError("matching report was not delivered")

    def close(self):
        self.call.cancel()
        self.requests.put(None)
        self.channel.close()


class TestRealLoadReporter(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(load_reporter_port=30100)
        override.install()
        self.addCleanup(override.restore)

    def test_pub_startup_failure_closes_subscribers_on_reporter_thread(self):
        for listener in ("grpc", "collector"):
            with self.subTest(listener=listener), socket.socket() as occupied:
                occupied.bind(("127.0.0.1", 0))
                occupied.listen()
                port = occupied.getsockname()[1]
                config = collector_config(
                    collector_port=port if listener == "collector" else None
                )
                created = []

                def source_factory():
                    source = ZmqPubSnapshotSource(config)
                    created.append(source)
                    return source

                handle = LoadReporterHandle(
                    source_factory,
                    ReporterSettings(
                        "127.0.0.1", port if listener == "grpc" else 0, "decode", "cpu"
                    ),
                )
                with self.assertRaises((RuntimeError, zmq.ZMQError)):
                    handle.start()
                self.assertTrue(created)
                self.assertTrue(all(sock.closed for sock in created[0]._sockets))
                self.assertTrue(all(task.done() for task in created[0]._tasks))

    def test_publishers_connect_before_listener_and_reconnect_after_restart(self):
        """One bound SUB identifies pure-DP ranks and tolerates either peer restarting."""

        async def check():
            config = collector_config()

            def make_publisher(rank):
                with (
                    get_context().override_server_args(
                        dp_size=2, nnodes=1, **reporter_fields(config)
                    ),
                    rc.get_parallel().override(
                        dp_rank=rank,
                        pp_rank=0,
                        attn_dp_rank=0,
                        attn_tp_rank=0,
                        attn_cp_rank=0,
                    ),
                ):
                    return SchedulerLoadPublisher(kv_events_config=None)

            publishers = [make_publisher(rank) for rank in range(2)]
            source = ZmqPubSnapshotSource(config)
            try:
                # Producers are ready before the listener; no inbound port on
                # either producer is provided to the receiver.
                for rank, publisher in enumerate(publishers):
                    publisher.publish_load_stat(lambda: sample(rank), force=True)
                source.start()
                self.assertEqual(len(source._sockets), 1)
                self.assertEqual(source.snapshot(), [])

                async def deliver(samples):
                    def received():
                        for snapshot in samples:
                            publishers[snapshot.dp_rank].publish_load_stat(
                                lambda: snapshot, snapshot=snapshot, force=True
                            )
                        latest = {s.dp_rank: s for s in source.snapshot()}
                        return all(latest.get(s.dp_rank) == s for s in samples)

                    await eventually(received)

                first = [sample(0, 101, 3), sample(1, 102, 4)]
                await deliver(first)
                publishers[1].close()
                publishers[1] = make_publisher(1)
                await deliver([sample(1, 103, 5)])
                self.assertEqual(source.stats["publisher_restarts"], 1)
                self.assertEqual(source.snapshot()[0], first[0])

                await source.aclose()

                async def restart_listener():
                    # libzmq releases the old listener on its I/O thread after
                    # socket.close(); wait for that release before rebinding.
                    while True:
                        restarted = ZmqPubSnapshotSource(config)
                        try:
                            restarted.start()
                        except zmq.ZMQError as exc:
                            await restarted.aclose()
                            if exc.errno != zmq.EADDRINUSE:
                                raise
                            await asyncio.sleep(0.01)
                        else:
                            return restarted

                source = await asyncio.wait_for(restart_listener(), timeout=5)
                self.assertEqual(source.snapshot(), [])
                await deliver([sample(0, 104, 6), sample(1, 105, 7)])
                self.assertTrue(all(not p._socket.closed for p in publishers))
            finally:
                await source.aclose()
                for publisher in publishers:
                    publisher.close()

        asyncio.run(check())

    def test_pub_receives_new_instance_samples_while_owner_waits_for_shm_lock(self):
        """A blocked internal projection must not freeze new rank data in gRPC."""
        from sglang.srt.managers import load_snapshot as legacy

        async def check(temporary):
            config = collector_config()
            shm_path = os.path.join(temporary, "shm")
            endpoint = f"ipc://{temporary}/internal"
            commands, ready = queue.Queue(), Future()
            entered_lock, projection_done = threading.Event(), threading.Event()
            original_lock = legacy.file_lock

            @contextmanager
            def observe_real_lock(fd, mode):
                if (
                    threading.current_thread().name == "test-shm-owner"
                    and mode == fcntl.LOCK_EX
                ):
                    entered_lock.set()
                with original_lock(fd, mode):
                    yield

            def owner_loop():
                reader = None
                try:
                    reader = ZmqShmLoadSnapshotReader(endpoint, shm_path, 2)
                    ready.set_result(reader)
                    while (command := commands.get()) is not None:
                        try:
                            reader.poll()
                            command.set_result(None)
                        except BaseException as exc:
                            command.set_exception(exc)
                finally:
                    if reader is not None:
                        reader.close()
                    projection_done.set()

            publishers, writers = [], []
            owner_thread = threading.Thread(
                target=owner_loop, name="test-shm-owner", daemon=True
            )
            owner_thread.start()
            ready.result(timeout=5)
            handle = None
            client, shared, lock_fd = None, ShmLoadSnapshotReader(shm_path, 2), None
            try:
                # Each node knows only the shared leader ingress, before it starts.
                for rank in range(2):
                    with (
                        get_context().override_server_args(
                            dp_size=1,
                            attn_dp_size=2,
                            tp_size=2,
                            nnodes=2,
                            node_rank=rank,
                            **reporter_fields(config),
                        ),
                        rc.get_parallel().override(
                            dp_rank=0,
                            tp_rank=rank,
                            pp_rank=0,
                            attn_dp_rank=rank,
                            attn_tp_rank=0,
                            attn_cp_rank=0,
                        ),
                    ):
                        publishers.append(SchedulerLoadPublisher(kv_events_config=None))
                    writers.append(ZmqLoadSnapshotWriter(endpoint, 2, rank))
                    self.assertTrue(writers[-1]._socket.poll(5000, zmq.POLLOUT))
                    writers[-1].write(sample(rank, timestamp=100.125, running=rank + 2))
                    completed = Future()
                    commands.put(completed)
                    await asyncio.to_thread(completed.result, 5)

                    # A connected PUSH can still be in transit when poll first runs.
                    async def project_rank():
                        while not any(s.dp_rank == rank for s in shared.read_all()):
                            writers[rank].write(
                                sample(rank, timestamp=100.125, running=rank + 2)
                            )
                            completed = Future()
                            commands.put(completed)
                            await asyncio.to_thread(completed.result, 5)
                            await asyncio.sleep(0.01)

                    await asyncio.wait_for(project_rank(), 5)
                with (
                    envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.override(True),
                    get_context().override_server_args(
                        host="127.0.0.1",
                        **reporter_fields(config),
                        dp_size=1,
                        attn_dp_size=2,
                        tp_size=2,
                        nnodes=2,
                        node_rank=0,
                        disaggregation_mode="decode",
                        served_model_name="cpu",
                    ),
                ):
                    handle = init_load_reporter()
                    start_load_reporter(handle)
                # No business-owner run_due or owner-loop timer is used.
                self.assertIsNot(handle.transport._loop, asyncio.get_running_loop())

                def receive_without_subscribers():
                    for rank in range(2):
                        snapshot = sample(rank, 101.125, rank + 3)
                        publishers[rank].publish_load_stat(
                            lambda: snapshot, snapshot=snapshot, force=True
                        )
                    return handle.source.stats["updated"] >= 2

                await eventually(receive_without_subscribers)
                self.assertEqual(handle.driver.stats["ticks"], 0)
                self.assertIsNone(handle.transport._runner._timer)
                client = Client(handle.transport.bound_port, interval=20)

                async def deliver(samples):
                    async def publish():
                        while True:
                            for snapshot in samples:
                                # The Scheduler's real order: internal PUSH, then PUB.
                                writers[snapshot.dp_rank].write(snapshot)
                                publishers[snapshot.dp_rank].publish_load_stat(
                                    lambda: snapshot, snapshot=snapshot, force=True
                                )
                            await asyncio.sleep(0.02)

                    task = asyncio.create_task(publish())
                    try:
                        return await client.report(
                            lambda r: (
                                r.status == pb.REPORT_STATUS_HEALTHY
                                and len(r.ranks) == 2
                                and all(
                                    r.ranks[s.dp_rank].num_running_reqs
                                    == s.num_running_reqs
                                    and r.ranks[s.dp_rank].snapshot_time_unix_ms
                                    == int(s.timestamp * 1000)
                                    for s in samples
                                )
                            )
                        )
                    finally:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

                first = await deliver([sample(0, 101.125, 3), sample(1, 101.125, 4)])
                lock_fd = os.open(shm_path, os.O_RDONLY)
                fcntl.flock(lock_fd, fcntl.LOCK_SH)
                with mock.patch.object(legacy, "file_lock", observe_real_lock):
                    # Queue an internal update, then stop the owner at a real LOCK_EX.
                    writers[0].write(sample(0, 102.125, 5))
                    blocked = Future()
                    commands.put(blocked)
                    self.assertTrue(await asyncio.to_thread(entered_lock.wait, 5))
                    self.assertFalse(blocked.done())
                    latest = await deliver(
                        [sample(0, 103.125, 7), sample(1, 103.125, 8)]
                    )
                    self.assertGreater(latest.sequence_id, first.sequence_id)
                    self.assertEqual(
                        sum(rank.num_running_reqs for rank in latest.ranks), 15
                    )
                    self.assertFalse(blocked.done())
                    # SHM remains old while independent PUB samples have advanced.
                    self.assertTrue(all(s.timestamp < 103 for s in shared.read_all()))
                    changed = await deliver([sample(0, 104.125, 9)])
                    self.assertEqual(changed.ranks[1], latest.ranks[1])
                    repeated = await client.report(
                        lambda r: r.sequence_id > changed.sequence_id
                    )
                    self.assertEqual(repeated.ranks, changed.ranks)
                    self.assertGreaterEqual(handle.source.stats["updated"], 5)
                    handle.close()
                    self.assertFalse(handle.transport._thread.is_alive())
                    self.assertFalse(blocked.done())
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    await asyncio.to_thread(blocked.result, 5)
                    self.assertTrue(any(s.timestamp >= 101 for s in shared.read_all()))
            finally:
                if lock_fd is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                if client is not None:
                    client.close()
                if handle is not None:
                    close_load_reporter(handle)
                commands.put(None)
                owner_thread.join(timeout=5)
                self.assertTrue(projection_done.is_set())
                for writer in writers:
                    writer.close()
                for publisher in publishers:
                    publisher.close()
                if shared is not None:
                    shared.close()

        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(check(temporary))

    def test_disabled_lifecycle_does_not_import_transport_or_create_resources(self):
        with (
            get_context().override_server_args(load_reporter_port=None),
            mock.patch.dict(
                sys.modules,
                {
                    "sglang.srt.load_reporter.lifecycle": None,
                    "sglang.srt.load_reporter.pub_source": None,
                },
            ),
        ):
            threads = set(threading.enumerate())
            for parent in (False, True):
                reporter = init_load_reporter(http_parent=parent)
                start_load_reporter(reporter)
                close_load_reporter(reporter)
                self.assertIsNone(reporter)
            self.assertEqual(set(threading.enumerate()), threads)

    def test_startup_timeout_cannot_leave_a_late_listener(self):
        """A timeout before initialization must survive a delayed sender startup."""
        entered, release = threading.Event(), threading.Event()
        captured = []
        serve = ReporterTransport._serve

        async def delayed_serve(transport):
            captured.append(transport)
            entered.set()
            release.wait(5)  # Simulate a blocking import before gRPC initialization.
            await serve(transport)

        # Gate the timeout on entry so the interleaving is independent of CPU load.
        from concurrent.futures import Future

        class StartupFuture(Future):
            def result(self, timeout=None):
                if not entered.wait(5):
                    raise AssertionError("sender never entered startup")
                return super().result(timeout=0)

        with (
            mock.patch.object(ReporterTransport, "_serve", delayed_serve),
            mock.patch("sglang.srt.load_reporter.lifecycle.Future", StartupFuture),
            mock.patch(
                "sglang.srt.load_reporter.lifecycle.SHUTDOWN_TIMEOUT_SECONDS", 0.05
            ),
        ):
            try:
                with self.assertRaises((TimeoutError, RuntimeError)):
                    ReporterTransport(
                        _Source, WorkerMetadata("127.0.0.1:0", pb.WORKER_TYPE_REGULAR)
                    )
                release.set()
                captured[0]._thread.join(timeout=2)
                self.assertFalse(captured[0]._thread.is_alive())
                self.assertIsNone(captured[0].runtime)
            finally:
                release.set()
                if captured:
                    # Also clean up the pre-fix implementation when reproducing.
                    if captured[0]._thread.is_alive():
                        Future.result(captured[0]._ready, timeout=2)
                    captured[0].close()


if __name__ == "__main__":
    unittest.main()
