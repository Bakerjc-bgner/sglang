# Instance load reporter

## Enable and configure

Reporting uses an independent PUB/SUB path with one SUB listener on node 0.
Each publishing Scheduler connects its PUB to this shared ingress. The leader
needs no list of follower IPs, publisher ports or rank-to-node mappings.
The supported lifecycle remains Python HTTP serving with fixed DP topology.

The only reporter option is `--load-reporter-port`. Pass the same port on every
node; the ingress and wire identity are derived internally from existing config:

```bash
SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1 python -m sglang.launch_server \
  --model-path <model> \
  --tp-size 2 --attn-dp-size 2 --nnodes 2 \
  --node-rank <0-or-1> --dist-init-addr <leader-host:port> \
  --load-reporter-port 30100
```

Only node 0 hosts the SUB listener and gRPC reporter. The PUB connection address
is `tcp://<dist-init-host>:30101`: the host from `--dist-init-addr` and
`--load-reporter-port + 1`. Single-node serving without `--dist-init-addr` uses
`127.0.0.1`. The distributed-init host must be the reachable node-0 host; an
external rendezvous host on another machine is outside this topology.

The collector binds a wildcard IPv4/IPv6 address at that port. Wildcard hosts
cannot be used as PUB connection destinations. IPv6 literals use brackets. Both
ports must be available; the collector port must differ from the HTTP, native
gRPC, NCCL and distributed-init ports. The reporter port must be in [1, 65534]
because it reserves the next port. A bind failure aborts reporter startup and
reclaims its socket and thread.

The existing metadata fields carry the derived collector address as the instance
tag and the configured DP size as the topology tag. They are consistent across
nodes without additional CLI options. These tags detect a mismatched endpoint or
rank set; they do not fence an old deployment that reuses the same endpoint and
DP size. Deployment tooling must ensure only one live publisher owns each rank.
A boot UUID still distinguishes individual publisher restarts. No publisher
address directory, instance ID or topology generation option is required.

For single-node serving, also set `SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1`; multi-node DP
attention already selects the internal PUSH writer. Startup still rejects a
direct-SHM writer, so the Scheduler cannot wait for a direct SHM write before its
PUB call. Reporting does not require KV event publishing. Do not combine it with
`--load-publish-endpoint`; that option retains its existing native port-range
publishing behavior.

Publishing remains one source per logical DP rank, selected on
pp/attention-TP/attention-CP rank 0. It uses attention-DP rank under DP attention
and ordinary DP rank otherwise. Other Scheduler processes create no reporter
socket. All publishers may start before the SUB listener: `connect` does not
wait for reporter readiness, and later samples/heartbeats repair early loss.
An established connection is not evidence that a fresh sample has arrived.

## Snapshot paths and ownership

```text
Scheduler -> PUSH connect -> existing PULL bind -> synchronous SHM -> DP budget
          -> PUB connect  -> node-0 SUB bind -> latest per rank -> gRPC -> AIgate
```

The internal snapshot chain above is still part of SGLang. The earlier removal
of legacy snapshot-owner integration removed the reporter's borrowed reader and
owner callbacks; it did not remove the business PUSH/PULL/SHM implementation.
Its owner selection, SHM format, locks, sampling and budget reads remain unchanged.

The independent reporter thread owns its SUB, per-rank cache, report timer and
gRPC listener. It never calls or closes an internal reader or accesses SHM.
Collection runs continuously without Monitor subscribers. Bounded receive batches
yield so the shared ingress cannot monopolize reporting. Source rank comes from
the validated message, rather than a socket-to-directory mapping. The expected
rank set still comes from the configured `num_dp_ranks`; silently missing a
publisher cannot shrink that set into a complete instance.

A publisher restarting at the same ingress has a new boot UUID. The receiver
resets that rank's sequence/cache and rejects messages from retired boots.
Other ranks retain their samples. A reporter restart starts with an empty cache;
existing publishers reconnect and repopulate it. Dynamic DP resizing, leader
migration and overlapping publishers for one rank are not supported by this
fixed-topology lifecycle. If the leader's advertised address changes, deployment
tooling must provide a reachable stable endpoint or coordinate reconnection.

The existing `[load, 8-byte sequence, MsgPack LoadStat]` framing and field prefix
are preserved. The optional `reporter_metadata` suffix carries protocol version,
derived instance/topology tags, publisher boot UUID, the original
`LoadSnapshot.timestamp`, and all four floating-point metrics in `RankLoad`.
The receiver rejects mismatched instance/topology tags, unknown ranks, malformed data,
missing source timestamps and invalid metrics before replacing any cached boot.
Do not use ZMQ CONFLATE on this multipart path; latest samples are kept per rank
in the application cache.

The gRPC schema, reporter-lifecycle `source_instance_id`, report sequences and
session semantics remain unchanged. That reporter boot UUID is distinct from the
derived instance tag. Cross-node source clocks still need synchronization.
Unchanged gauges retain the publisher's one-second heartbeat and decode-iteration
throttle. A 100ms report interval does not create 100ms source samples. Source
sampling or a Scheduler stall before PUB remains outside this transport change.

## Runtime and protocol

Install the optional dependencies with `pip install 'sglang[load-reporter]'`
(or `pip install -e 'python[load-reporter]'` from a checkout).
The bindings require grpcio >= 1.78.0 and protobuf >= 6.31.1. The canonical schema
is `proto/sglang/router/loadmonitor/v1/load_monitor.proto`.

The leader's `/server_info` advertises the reporter gRPC port, protocol and service.
AIgate connects to that leader endpoint; it does not enumerate or connect rank
publishers. Only the leader should be exposed as the routing endpoint for one
multi-node engine. Each independent instance has its own collector listener and
identity; unrelated P/D instances must not share them.

- Without Monitor sessions, SUB reception continues but no broadcast timer runs.
- The first Register starts a period and receives its first report after one
  interval. Late joins and keepalives preserve the phase. Only the oldest active
  session may change the interval. Late ticks skip missed periods.
- Individual leases expire even when a network write is blocked. A replacement
  session closes the old stream. Each session has a one-report latest mailbox;
  there are at most 128 sessions.
- Reports contain exactly the configured rank set, preserving source timestamps.
  Missing ranks produce UNREACHABLE. A complete valid view produces HEALTHY
  regardless of age: AIgate must check per-rank source age and receipt timeout.
  New report sequences never renew old source time.
- Stopping the reporter cancels only its own receive task, timer, sessions and
  listeners. Internal PUSH/PULL ownership remains independent.
- Omitting `--load-reporter-port` creates no reporter PUB/SUB sockets, thread or
  gRPC listener, and applies no reporter-specific validation. Native snapshot
  backend selection and `--load-publish-endpoint` behavior remain unchanged.
  `is_load_reporter_enabled()` is the shared switch for Scheduler publishing
  and reporter startup.

Address resolution and identity configuration live in `load_reporter/ingress.py`;
PUB connection and rank selection live in `load_reporter/publisher.py`.
The existing load publisher checks the shared feature switch itself.
Native throttling, heartbeat, encoding and socket cleanup stay in that component.
Argument checks live in `load_reporter/validation.py` through the existing hook.
HTTP lifespan owns the reporter in single-worker mode; the HTTP parent owns it
in multi-worker mode. No MultiTokenizerRouter or DPController integration is added.

## Validation

Run CPU files in their own processes with `PYTHONPATH=python python <file> -v`:

- `test/registered/unit/load_reporter/test_load_reporter.py`: shared ingress
  resolution, disabled-mode isolation, rank/source validation, legacy decoder compatibility, boot/sequence
  handling, broadcast scheduling, sessions, blocked writes, both listener bind
  failures, publisher-before-listener startup and either peer reconnecting.
- Its real PUB/gRPC lock-isolation test holds a shared-file lock while the native
  owner waits for SHM projection. New rank snapshots still reach gRPC, a single
  rank can update independently, repeated reports preserve source timestamps,
  and reporter shutdown does not wait for the native owner.
- `test/registered/unit/managers/test_load_snapshot_backends.py` and
  `test_data_parallel_controller.py`: native synchronous SHM projection,
  PUSH/PULL ownership and dispatch-budget behavior.
- `test/registered/unit/entrypoints/test_load_reporter_lifecycle.py`: HTTP parent
  ownership and cleanup on normal exit or startup failure.
- Configuration namespace/read-ratchet and CLI tests cover the configuration wiring.

The existing two-GPU smoke in
`test/registered/e2e/load_reporter/test_load_reporter.py` now connects both DP
publishers to one collector, with no directory file. It checks source timestamp
progress without HTTP load polling. CPU tests do not establish real multi-node
GPU serving, Kubernetes networking or live AIgate interoperability.
