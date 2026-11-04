# Design

## Question

vLLM serves many requests in one batch, so one request's latency depends on
everything it shares steps with. When a request's tokens slow down, the
interesting question is *why*. Was the GPU busy with someone else's prefill?
Was the step's host work slow? Or was the request simply not scheduled?
Answering that needs three things joined on one timeline:

1. **Engine steps and their members:** which requests each step scheduled,
   with how many tokens and in which phase.
2. **GPU work:** every kernel, copy and memset, and which step launched it.
3. **Request lifecycles:** arrival, first scheduling, every token, finish.

Generic profilers capture (2) but know nothing about (1) or (3). Serving
metrics know (3) but not why. reqscope records all three with one clock and a
measured bound on how far GPU timestamps can be off.

## Recording

```mermaid
flowchart LR
  subgraph EngineCore process
    A[vLLM plugin<br/>Step + Request records] -->|C ABI| Q[(MPSC queue)]
    C[CUPTI activity buffers] --> Q
    P[clock probe thread] --> Q
    Q --> W[writer thread]
  end
  W --> S[(CRC-framed segments<br/>ring on disk)]
  X[reqscope dump / flush] -.->|control socket| W
```

**The collector** (`collector/`, `libreqscope.so`) is loaded with
`CUDA_INJECTION64_PATH`, so it starts inside CUDA's own initialization,
before the application creates a context.
- It subscribes to CUPTI activity records and registers CLOCK_MONOTONIC as
  CUPTI's timestamp source.
- Records go through a bounded multi-producer queue to one writer thread.
  The writer appends them to CRC-framed segment files and keeps a
  time- and size-bounded ring on disk.
- A control socket (`/tmp/reqscope-PID.sock`) takes `flush`, `dump` (freeze
  the last N seconds into a new directory) and `status`.

Three modes:
- **plugin**: CUPTI is never started; only the vLLM plugin's steps,
  membership and request lifecycles are recorded, and the header has no GPU
  clock bound.
- **light**: one record per CUDA graph launch (graph envelope). Kernels
  launched outside a graph are not recorded one by one.
- **trace**: one record per kernel, including kernels inside graphs. Its
  per-kernel collection changes host timing under load (README, CUPTI-off
  comparison and open-loop overhead).

Both modes record only the driver and runtime APIs that submit work, move
data or synchronize, enabled one CUPTI callback ID at a time.

Torch and Triton surround every launch with query calls (`cudaGetDevice`,
`cuDevicePrimaryCtxGetState`, `cuPointerGetAttribute`). Recording those made
up most of trace mode's volume: in a capture of the README's experiment
with every API enabled, `cudaGetDevice` alone was half of 1.8M API records.
The single writer could not keep up, and CUPTI dropped 1.5M records in
22.7 s (`results/captures.json`, run `interference-s3`).

**What trace mode costs the engine.** Each capture carries the collector's
own accounting (`CollectorStats`: calls, time and longest call per site,
and which ran on an engine thread), and the benchmark samples every
EngineCore thread from `/proc`. One 40 s open-loop run per configuration
at 16 requests/s, read by `bench/trace_cost.py`
([results/trace-cost.json](../results/trace-cost.json)):
- **The collector's code on the engine thread is not the cost.** Buffer
  requests (64/s, under 0.3 µs), step push and pop (350/s each, under
  0.6 µs) and plugin records (190/s, 0.2 µs) add up to under 0.1% of a
  core. CUPTI never returned a buffer on an engine thread.
- **CUPTI's work inside each launch is,** and the collector cannot change
  it short of not tracing every kernel: the decode step's graph launch
  (about 400 kernel nodes) takes 277 µs in trace mode against 72 µs in
  light mode.
- **The writer was 0.33 cores,** 5.1 ms per 256 KB CUPTI buffer: 1.8 µs
  per record to frame it and most of the rest building Json objects. While
  it ran that way, 8% of kernel and copy launches (29,243 in 42 s) took
  over 100 µs, nearly all in runs of three or more consecutive launches of
  about 250 µs each: 7.2 s of launch time. The writer now formats each
  record as text (0.32 µs to frame, 0.10 cores in all) and those runs are
  gone (2,563 launches over 100 µs, 3 in runs; 4.4 s). Why the Json
  writer slowed launches on another thread is not established: slow
  launches overlapped the writer's busy intervals less often than other
  launches (15% against 40%), and CUPTI's own worker thread also fell from
  0.28 to 0.12 cores. The old writer allocated and freed several objects
  per record.
- **Graph launches over 1 ms** (27 in 42 s, median 1.44 ms, 1.3 s apart)
  come partly from CUPTI moving to its next device buffer: in
  `bench/launch_cost.py` the default 3.2 MB buffer gave 21 such stalls in
  5,000 steps and a 32 MB buffer 1 ([results/launch-cost.json](../results/launch-cost.json)).
  The collector now asks for 32 MB buffers, four allocated up front. In
  vLLM 15 remained in 42 s, so not all of them are buffer switches.
- **Not kept:** running the writer, the probe and CUPTI's worker on the
  host's efficiency cores (TPOT p50 5.51 ms against 5.36 ms unpinned, and
  8.4 s of slow launches against 4.4 s); a 1 ms CUPTI flush period (TPOT
  p99 14.3 ms, 150 stalls); pre-zeroed host buffers and a lower writer
  priority, which changed nothing in `bench/launch_cost.py`. 4 MB host
  buffers helped there only at a step rate the Json writer could not
  drain. The engine thread ran on an efficiency core in 1% of samples and
  waited for a CPU under 0.1% of the time, so CPU contention was not the
  cost on this host at its load during these runs (2–5).

**The vLLM plugin** (`adapter/vllm`) is a `vllm.general_plugins` entry point.
It wraps:
- `GPUModelRunner.execute_model` and `sample_tokens`:
  - one **Step** record per engine step, with its members, phases and
    tokens;
  - CUPTI external correlation `CUSTOM0 = step_id` pushed around the step's
    launches, so every API call, and through its correlation id every
    kernel, carries the step id;
  - the step id written onto the `SchedulerOutput`.
- `Scheduler.add_request`, `schedule`, `update_from_output` and
  `_free_request`:
  - one **Request** record per request: arrival, first scheduled step,
    finish, and every token's time and producing step;
  - the producing step is read back from that `SchedulerOutput`;
  - a request's token log is flushed in 1,024-token `RequestTokens` chunks,
    so no record outgrows a queue slot.

The hooks need two details to be right:
- **Token time is recorded before vLLM processes the output.** vLLM frees a
  finished request inside `update_from_output`, so recording it afterwards
  lost every request's last token.
- **The producing step is recorded explicitly.** With async scheduling,
  step k+1 is already launched when step k's tokens are processed, so a
  timestamp cannot tell which step produced a token.

**Failure.** The profiler must not take serving down. If a collector call
fails (the writer hit a full disk, the collector is stopping, an NVTX range
cannot be opened), the plugin closes what it opened, stops collecting for
the rest of the process, records a `Diagnostic` if the collector still
accepts records, and warns; the engine step runs normally. A failed writer
also leaves `collector-error-PID.txt` in the output directory.

**Shutdown.** vLLM's shutdown SIGKILLs EngineCore almost immediately. A
capture is only complete if someone sends the collector `flush` first.
`bench/interference.py` does that for every collector in the server's
process tree.

## Clock bound

CUPTI stamps GPU activity in the collector's CLOCK_MONOTONIC domain, but
nothing says how wrong those stamps can be. A probe thread measures it:

```
every 500 ms:  t0 = now();  cuMemcpyHtoDAsync(4 bytes, private stream);
               cuStreamSynchronize();  t1 = now()
```

The copy's CUPTI `[start, end]` must lie inside `[t0, t1]`. The analyzer
pairs each `ClockProbe` record with that copy by time: a copy on the same
context and stream that lies at most 50 ms outside the bracket, matching
the closest pairs first so that a probe never takes its neighbour's copy.
It then takes the widest bracket plus any violation as the session's bound.
If any probe is unpaired, or the probe stopped (its setup or a copy failed,
which it reports in a `Diagnostic` with `stopped: true`), the bound stays
unknown. A probe that completed
after the last GPU record CUPTI delivered is counted as pending rather than
unpaired, since its copy may still have been in a CUPTI buffer. The probe
copies are excluded from application GPU time.

Two collector bugs kept the bound unknown in early runs:
- The probe bound the driver API with `dlsym(RTLD_DEFAULT)`. The CUDA runtime
  loads `libcuda` with `RTLD_LOCAL`, so the probe never started, and nothing
  said so. It now takes a handle to the loaded copy (`RTLD_NOLOAD`) and
  reports a `Diagnostic` record if it cannot run.
- The CUPTI device-buffer pool was capped at 4 buffers per context. At
  vLLM's launch rate CUPTI then delivered many records with zero timestamps:
  14% of kernels and most copies, probe copies included. The cap is gone.
  The analyzer treats a zero GPU timestamp as missing and counts such
  records.

## Analysis

`reqscope` (`analyzer/`, `cli/`) reads a capture with these joins:

```
GpuActivity --corr--> ApiCall --corr--> ExtCorr(step_id) --> Step --members--> Request
```

All ids are scoped by producer (boot id, pid, process start time). The
chain from GPU activity to request uses only these ids. Two joins do use
time: pairing clock probes with their copies (above), and `blame`'s
fallback for token logs without step ids (below). `analyze` and `explain`
split a timeline, so they also place scheduler and synchronization records
by time.

**`blame`** answers "why were this request's tokens slow":
1. Take the request's token times and the step that produced each token.
   If any chunk of the request's token log lacks `token_steps` (the plugin
   omits them from a chunk with a token of unknown step), fall back to the
   request's steps whose host span ended inside the gap. Under async
   scheduling that can be one step late, so the request's row says
   `producing_steps: host_span_in_gap` and the output carries a warning.
2. A gap more than 1.5× the request's median gap is slow. Its excess over
   the median is what needs explaining.
3. For the step that produced the token after the gap, report:
   - its GPU busy time (the union of its linked kernels and copies, measured
     in the GPU's own clock, so no cross-clock alignment is needed);
   - its co-scheduled members, with their phases and tokens.
4. Split the excess over the prefill tokens co-scheduled in that step. A
   slow gap with no co-scheduled prefill is reported as unexplained.
5. Rank culprits by the excess they caused across all requests.

`blame` also reports how far to trust the answer:
- how many GPU records it linked to steps, and how many had no timestamps;
- `trace`: Complete or Partial, with the reader's and analyzer's issues by
  reason (truncated or corrupt frames, sequence gaps, a missing end of
  session, a `collector-error-PID.txt` left by a failed writer), losses by
  reason, and the `Diagnostic` records of the collector and plugin;
- how many requests used the host-span fallback;
- `warnings`: one sentence for each of the above that limits the answer;
- the clock bound.

### Causes

Each slow gap also gets a `cause`, from its producing step and the gap
`[begin, end]`, with the numbers behind it in `cause_evidence`. The rules
are applied in order; "excess" is the gap minus the request's median gap.

1. **interference**: at least 128 prefill tokens of other requests were
   co-scheduled in the producing step.
2. **undetermined**: the producing step has no GPU records (plugin-only
   capture, or no linked GPU work), or the capture is sampled and the step
   has no `TracedStep` record. Without GPU time, GPU slowness and host
   delay both look like a long gap. In a sampled capture only traced steps
   enter the baselines below, and each slow gap's step says whether it was
   traced (`traced`); `blame`'s summary counts traced steps
   (`sampled_steps`).
3. Otherwise two components are measured against baselines:
   - **GPU extra**: how much the step's GPU busy time, its GPU wait inside
     the step, and its launch wait exceed the medians of comparable steps.
     Comparable steps have the same producer, member count and scheduled
     tokens, no prefill, and step ids within 200 of this one (local,
     because decode GPU work grows with context length). The GPU wait
     inside the step is idle time between the step's own GPU intervals
     while its next kernel had already been launched; the launch wait is
     the shortest time from a launch API's return to its kernel's start.
     Both grow when the GPU is running another process's work. At least 5
     comparable steps are needed, or GPU extra is 0.
   - **host extra**: how much the gap's host time exceeds the request's
     median over all its gaps. Host time is idle time inside the step while
     its next kernel had not been launched yet, time after the step's GPU
     work finished until the token was processed, and time from the gap's
     start until the step's first launch returned.
4. **gpu_slowdown** if GPU extra is at least host extra and at least half
   the excess; **host_delay** if host extra is larger and at least half the
   excess; **unexplained** otherwise. The GPU terms can overlap (a slower
   previous step also lengthens this step's launch wait), so GPU extra can
   exceed the excess; the components are compared, not added up to the gap.

A host delay can be the profiler's own. When the capture records the
writer's busy intervals (`CollectorActivity`), each slow gap's evidence has
`collector_active_ns`, the time inside the gap when the writer was busy or
the clock probe was bracketing a copy, and a host-delay verdict gets the
flag `collector_active` when that time is at least half its host extra.
The flag says the collector was busy for as long as the delay, not that it
caused it. `blame`'s summary counts flagged host delays
(`host_delay_gaps_with_collector_active`).

The rules do not separate a host-device synchronization stall from other
host delay: both appear as host time around the step's GPU work. A long
copy on the step's own streams counts as GPU work of the step.

**`analyze` / `explain`** split each request's wall time into
`gpu_exec`, `gpu_idle_in_step`, `between_steps`, `between_steps_cpu` (with
scheduler evidence), `not_scheduled` and `residual`. GPU intervals are
used only when the clock bound is known and shorter than the interval.
`export-perfetto` writes a timeline.

The decomposition cuts a request's span at every start and end of the
intervals around it (its steps, its GPU activities, all GPU activity of its
process, scheduler states, losses, tokens). Each elementary piece is then
inside or outside every such interval, so one sweep over the pieces keeps
the intervals covering the current one. Tokens are matched to pieces by a
merge, and in-step gaps read per-process indexes of GPU busy time and
synchronization records.

## Format

A segment file starts with the 8 bytes `RQSSEG01`. Each frame is then a
little-endian `uint32 length`, a `uint32 crc32`, and `length` bytes of JSON.
- The first frame is a `Header`: producer identity, clock, mode, CUPTI
  version and the APIs enabled.
- Every other record has `type`, `producer` and a per-producer `seq`
  starting at 1. Gaps in `seq` are losses.

The collector reports its own losses as `Loss` records with a reason and a
count: CUPTI pool exhaustion, CUPTI drops, queue full, oversize records.

The reader verifies each frame's CRC. It fails loudly when a trace exceeds
its memory budget (`REQSCOPE_TRACE_BUDGET_MIB`); it never truncates silently.

## Limits

- One CUPTI subscriber (the V1 API): light and trace mode cannot run
  together with Nsight Systems or `torch.profiler`; plugin-only mode can.
- The record layouts are those of CUPTI API 130202 (CUDA 13.2). Any other
  version is refused rather than parsed with the wrong layout.
- The vLLM plugin supports exactly vLLM 0.31.0 with the V2 model runner,
  the uni executor, TP=1 and PP=1.
- JSON records keep the format easy to inspect, at a cost in size (35 MB/s
  in trace mode under the overhead load) and in parse time (see the
  README's analysis-speed section). A binary encoding of the four record
  types that make up 99% of a trace is the next step.
- `analyze` and `explain` report every elementary piece of every request,
  so their output grows with the GPU activity around each request and is
  much larger than the capture. Long captures need `blame`, or a shorter
  window.
