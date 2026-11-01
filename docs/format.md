# The RQSSEG01 segment format

All times are unsigned 64-bit `CLOCK_MONOTONIC` nanoseconds.

A segment file starts with the 8 ASCII bytes `RQSSEG01`. Each frame is a
little-endian `uint32 length` and `uint32 crc32`, followed by `length` bytes
of UTF-8 JSON. The CRC is IEEE CRC32, the same as Python's `zlib.crc32`.
The first frame is a `Header`; every later frame is one complete record.
Frames are only appended. A corrupt frame or a truncated tail is never
treated as evidence.

## Header

A header has `type=Header`, `format_version=1`, `producer`, `boot_id`,
`pid`, `start_time_ticks` and `clock=CLOCK_MONOTONIC`. `producer` must equal
`boot_id:pid:start_time_ticks`.

`clock_error_ns` missing or null means the bound was not measured. Zero is
only for synthetic or manual records that really share one clock; it is
never a default for GPU time. A plugin-only capture (`REQSCOPE_MODE=plugin`:
the vLLM plugin's records, CUPTI never started) has no GPU records and no
`clock_error_ns`. The collector also records the libcupti path
and API version it loaded, how it subscribed, its state, its mode and its
buffer configuration.

## Records

Every record has `type`, `producer` and `seq`; the collector also adds `pid`
and `timestamp`, the time it wrote the record. `seq` starts at 1 and keeps
increasing across all of a producer's segments, and processes never share a
sequence space. Records are ordered by producer and `seq` before gaps are
looked for; a repeated `seq` or a conflicting identity also makes the trace
Partial. The BPF loader writes as its own producer and names the observed
process in `target_producer`.

| Type | Main fields |
|---|---|
| ApiCall | corr, cbid, tid, start, end |
| ExtCorr | corr, step_id |
| GpuActivity | corr, kind, ctx, stream, start, end, graph_id, graph_node_id, bytes |
| GraphLaunch | corr, graph_id, start, end; the CPU launch, not GPU execution |
| GraphResource | action, timestamp, graph_id, graph_exec_id, node_id, dependency_node_id, original_graph_id, original_node_id, ids_valid; from CUPTI's public resource callbacks |
| Sync | corr, sync_type, stream, event, start, end |
| CudaEvent | corr, ctx, stream, event, event_sync_id, device_timestamp; a recorded CUDA event, matched to syncs by ctx and event_sync_id |
| ClockProbe | probe, t0, t1, ctx, stream; host clock reads before and after the probe thread's 4-byte copy on its own stream, which the analyzer pairs with that copy's GpuActivity to bound clock error |
| ClockCalibration | clock, calibrated, source; how GPU timestamps relate to the host clock, written at session start |
| Diagnostic | source, reason, stopped; a component that could not run or gave up (`stopped`: true), such as the clock probe (with the driver `error` and the `probes` completed) or the vLLM plugin after a collector failure |
| AdapterConfig | vllm_version, runner, executor, async_scheduling, cudagraph_mode, tensor_parallel_size, pipeline_parallel_size; written once by the vLLM plugin |
| Step | step_id, t_begin, t_end, tid, members[{req_id, tokens, phase}], preempted, finished |
| Request | req_id, start, end, prompt_tokens; optional queued_until, token_times |
| RequestTokens | req_id, first, token_times; optional token_steps (one chunk per 1,024 tokens) |
| Sched | tid, start, end, state = on_cpu / runnable / sleeping |
| SchedulerSample | runtime and run-queue deltas with resolution_ns; it does not replace Sched |
| Loss | reason, count; optional seq_range, time_range, count_unit |
| Session | state, with the backend error or exit signal if any |
| CollectorStats | sites{name: calls, ns, max_ns, engine_calls}, timestamp_callbacks, cupti_records, cupti_bytes, writer_cpu_ns, writer_sleep_ns, probe_cpu_ns, since_ns; the collector's own time per site, cumulative since `since_ns`, written every second and on flush. `engine_calls` counts calls on a thread that has pushed a step id |

## Losses

A `Loss` count must state its unit: a dropped CUPTI buffer is not a known
number of lost kernels. Ring overwrite and sequence gaps can both occur, and
must not be added together as an exact kernel count. A loss without an exact
time range affects its whole producer.

## Reader limits

- A frame holds at most 1 MiB, and a trace at most 4,096 segments and
  20 million records.
- By default the reader accepts 2 GiB of frame payload and 2 GiB of
  estimated decoded records. `REQSCOPE_TRACE_BUDGET_MIB` (1–65,536) raises
  both; it is not a bound on the analyzer's total memory.
- Each JSON document is limited to 65,536 nodes and a depth of 64, and a
  record passed to the collector's C ABI to 64 KiB.
- Invalid UTF-8, duplicate keys, non-finite numbers and out-of-range
  integers are rejected. An unknown record type is never used as causal
  evidence.
- On damage the reader keeps the verified prefix and reports a diagnostic.
  A trace over a limit fails instead of being truncated silently.

Segments are created with `O_EXCL` and mode 0600. Rotation only writes; the
collector syncs on dump, flush and finalize.
