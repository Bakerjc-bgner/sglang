"""One reporter-owned SUB listener, independent of the internal PULL/SHM owner."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter

import msgspec
import zmq
import zmq.asyncio

from sglang.srt.load_reporter.publisher_protocol import snapshot_from_stat
from sglang.srt.load_reporter.snapshot_source import SnapshotCache
from sglang.srt.load_reporter.snapshot_validation import validate_rank_snapshot
from sglang.srt.managers.scheduler_components.load_publisher import LoadStat
from sglang.srt.utils.network import is_zmq_endpoint_ipv6

logger = logging.getLogger(__name__)


class ZmqPubSnapshotSource:
    """All sockets, sequence state and cached samples belong to the reporter loop."""

    def __init__(self, config):
        self.config = config
        self.dp_size = config.dp_size
        self.cache = SnapshotCache(self.dp_size)
        self.stats = Counter()
        self._decoder = msgspec.msgpack.Decoder(LoadStat)
        self._active = {}
        self._retired = {}
        self._sockets = []
        self._tasks = []

    def start(self):
        sock = zmq.asyncio.Context.instance().socket(zmq.SUB)
        self._sockets.append(sock)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 8)
        sock.setsockopt(zmq.SUBSCRIBE, b"load")
        endpoint = self.config.bind_address()
        if is_zmq_endpoint_ipv6(endpoint):
            sock.setsockopt(zmq.IPV6, 1)
        sock.bind(endpoint)
        self._tasks.append(asyncio.create_task(self._receive(sock)))

    async def _receive(self, sock):
        batch = 0
        while True:
            frames = await sock.recv_multipart()
            self.accept(frames)
            batch += 1
            if batch == 32:
                # recv can complete immediately under sustained traffic. Keep the
                # report clock and gRPC sessions runnable even then.
                batch = 0
                await asyncio.sleep(0)

    def accept(self, frames):
        """Validate a complete frame before it can change the active source."""
        self.stats["received"] += 1
        try:
            if len(frames) != 3 or frames[0] != b"load" or len(frames[1]) != 8:
                raise ValueError("expected [load, 8-byte sequence, LoadStat]")
            sequence = int.from_bytes(frames[1], "big")
            stat = self._decoder.decode(frames[2])
            snapshot = snapshot_from_stat(stat, self.config)
            validate_rank_snapshot(snapshot, dp_size=self.dp_size)
            boot = stat.reporter_metadata.publisher_boot_id
            rank = snapshot.dp_rank
            if boot in self._retired.get(rank, ()):
                self.stats["out_of_order"] += 1
                return
            active = self._active.get(rank)
            if active is not None:
                old_boot, old_sequence = active
                if old_boot == boot and sequence <= old_sequence:
                    self.stats["out_of_order"] += 1
                    return
                if old_boot != boot:
                    self._retired.setdefault(rank, set()).add(old_boot)
                    self.cache.latest_by_rank.pop(rank, None)
                    self.stats["publisher_restarts"] += 1
            self._active[rank] = boot, sequence
            if self.cache.update([snapshot]):
                self.stats["updated"] += 1
        except (ValueError, msgspec.DecodeError) as exc:
            self.stats["invalid"] += 1
            if self.stats["invalid"] == 1:
                logger.warning(
                    "Rejected load PUB message at %s: %s", self.config.endpoint, exc
                )

    def snapshot(self):
        for task in self._tasks:
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise RuntimeError("load PUB receiver stopped") from task.exception()
        return self.cache.snapshot()

    async def aclose(self):
        for task in self._tasks:
            task.cancel()
        for sock in self._sockets:
            sock.close()
        await asyncio.gather(*self._tasks, return_exceptions=True)
