# Design trade-offs

reqscope runs inside a serving engine and has to produce answers an
operator can check, so most of its design is about what to record and how
to join it without guessing. The sections below go through those choices
one at a time, with the options that lost and the price of the one taken.
Numbers are from the README's experiment (Qwen2.5-1.5B on an RTX 4090,
vLLM 0.31.0) and its 23 s, 0.9 GB trace-mode capture.

## A CUPTI injection library, not a profiler wrapper

**Decision.** The collector is loaded with `CUDA_INJECTION64_PATH` and
subscribes to CUPTI from inside CUDA's own initialization. A vLLM plugin,
loaded as a `general_plugins` entry point, adds steps and request
lifecycles through a C ABI.

**Alternatives.** Nsight Systems or `torch.profiler` with post-hoc joins;
patching vLLM.

**Why.** No change to vLLM, PyTorch or the application, and the collector
is running before the first context exists. A generic profiler records the
GPU side but nothing about which requests a step served, which is the
question here.

**Cost.** CUPTI allows one subscriber, so reqscope cannot run alongside
Nsight Systems or `torch.profiler`. The record layouts are pinned to CUPTI
API 130202, and other versions are refused rather than misparsed.

## Join GPU work to requests by correlation ids, not by time

**Decision.** A kernel reaches its request only through ids: GPU activity
→ API call (CUPTI correlation id) → external correlation (the step id the
plugin pushes around the step's launches) → step → members.

**Alternatives.** Assign kernels to the step whose time window contains
them.

**Why.** With async scheduling, step k + 1 is launched before step k's
tokens are processed, so a time window picks the wrong step; the plugin
writes the step id onto each `SchedulerOutput` and each token records the
step that produced it. Ids are scoped by producer (boot id, pid, start
time), so records from different processes never collide.

**Cost.** Work launched outside the plugin's correlation scope cannot be
attributed, and is reported as unlinked (1% of GPU records in the
experiment) rather than guessed. Two joins still use time. Clock probes are
paired with their copies by time (a copy within 50 ms of the probe's
bracket, closest pairs first). And when a request's token log has no step
ids, `blame` falls back to the request's steps whose host span ended inside
the gap, which is one step late under async scheduling and is not marked in
its output.

## One clock, with a measured bound

**Decision.** CUPTI stamps GPU activity in the collector's
`CLOCK_MONOTONIC`, and a probe thread brackets a 4-byte copy between two
clock reads every 500 ms. The widest bracket, plus any violation, becomes
the session's bound, and GPU intervals shorter than the bound are not used
as evidence.

**Alternatives.** Trust CUPTI's timestamps; align clocks after the fact.

**Why.** Nothing else says how far a GPU timestamp can be off, and a
decomposition that treats a 200 µs kernel as exact when its stamps can be
off by 280 µs is wrong in a way no output reveals.

**Cost.** Two probe records per second and one tiny copy on a private
stream. Without paired probes the bound stays unknown and the affected
intervals are reported as residual.

## Record only the APIs that submit, copy or synchronize

**Decision.** CUPTI callbacks are enabled one callback id at a time, for
the driver and runtime calls that launch work, move data or synchronize.

**Alternatives.** Enable every API callback.

**Why.** Torch and Triton surround every launch with query calls
(`cudaGetDevice`, `cuDevicePrimaryCtxGetState`, `cuPointerGetAttribute`).
Recording them was most of trace mode's volume (`cudaGetDevice` alone was
half of 1.8M API records); the single writer fell behind and CUPTI dropped
1.5M records in a 22.7 s capture.

**Cost.** Calls outside the list are invisible, so a stall inside, say, a
memory query shows up only as time between steps.

## A bounded ring on disk, frozen on demand

**Decision.** Segments rotate into a time- and size-bounded ring, and
`reqscope dump` freezes the last N seconds into a new directory. Rotation
only writes; dump, flush and finalize sync.

**Alternatives.** An unbounded trace; fdatasync on every rotation.

**Why.** A collector that is cheap enough to leave on (light mode costs
1.0% of throughput) must not fill the disk, and the interesting window is
usually the last minute. Syncing each 1 MiB segment on the thread that
drains CUPTI left CUPTI without a free buffer 2.2M times in a 22.7 s
capture, and CUPTI dropped 460k records.

**Cost.** A crash loses whatever the page cache had not yet written, and
vLLM SIGKILLs its engine at shutdown, so a complete capture needs a
`flush` first.

## CRC-framed JSON records

**Decision.** Each record is one JSON document in a frame with its length
and IEEE CRC32. The reader rejects damage and limit violations loudly
instead of truncating.

**Alternatives.** A binary record layout; protobuf.

**Why.** Every record can be read with standard tools, new fields need no
schema change, and the Python tests write fixtures directly. The CRC makes
a torn tail detectable, so partial data is reported as partial.

**Cost.** Size and parse time. A trace-mode capture writes 35 MB/s under
the overhead load, and reading JSON is a large part of `blame`'s time even
with a table-driven CRC32 (the README's analysis-speed section has the
figures). A binary encoding of the four record types that are 99% of a
trace (GPU activity, API calls, external correlations, syncs) is the next
step.

## Blame splits excess linearly over co-scheduled prefill

**Decision.** A token gap above 1.5× the request's median is slow; its
excess over the median is split over the prefill tokens co-scheduled in the
step that produced the token. With no co-scheduled prefill it stays
unexplained.

**Alternatives.** A cost model of each step from its batch shape.

**Why.** The split needs nothing beyond what was recorded, and it can be
checked against ground truth: the six long prompts were the top six
culprits with 92.9% of the excess, 18 of the 22 slow steps inside a prefill
window were blamed on that prompt, and nothing was blamed on a long prompt
outside every prefill window.

**Cost.** It ignores attention cost growing with context length, and it
validates one cause, prefill interference, not others such as a co-running
GPU job. A slow step with no co-scheduled prefill stays unexplained: 11 of
the 30 slow steps in the experiment.

## Decompose on elementary pieces, swept once

**Decision.** `analyze` cuts each request's span at every boundary of the
intervals around it and classifies each elementary piece from the intervals
covering it, kept by one sweep.

**Alternatives.** Classify each piece by scanning every interval; merge
pieces into runs.

**Why.** Every piece lies wholly inside or outside each interval, so the
sweep gives the same answer as a full scan. Scanning every interval for
every piece grows with the square of the trace (the README's
analysis-speed section compares the two on the same capture).

**Cost.** The output still has one entry per piece, much larger than the
capture itself, so long captures need `blame` or a shorter window. Merging
pieces would shrink it but lose per-piece evidence.
