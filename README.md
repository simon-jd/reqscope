# reqscope: why was this vLLM request slow?

A request-level CPU/GPU profiler for vLLM. It records three things on one
clock:
- every engine step and which requests it scheduled;
- every kernel and copy, with the step that launched it;
- every request's token times.

The clock comes with a measured bound on how far GPU timestamps can be off.
Then `reqscope blame` answers the question an operator actually has: *these
tokens were slow, and who caused it?*

- **Collector:** a CUPTI injection library (C++20), loaded with
  `CUDA_INJECTION64_PATH`. No change to vLLM, PyTorch or the application.
- **vLLM plugin:** wraps the model runner and scheduler of vLLM 0.31.0 to
  emit steps, step membership and request lifecycles.
- **Analyzer and CLI:** `blame`, `explain`, `analyze`, `diff`,
  `export-perfetto`, `dump`.

**Experiment.** 16 requests decode continuously on Qwen2.5-1.5B (RTX 4090,
vLLM 0.31.0, chunked prefill, CUDA graphs, async scheduling). Six 6,000-token
prompts arrive, one every 3 s, and their prefill chunks share engine steps
with the decodes. The client records when each long prompt was in prefill
(sent until first token). The analyzer never sees that ground truth.

| What `reqscope blame` reported | |
|---|---|
| Top six culprits by excess latency caused | **exactly the six long prompts** |
| Slow engine steps inside a long prompt's prefill window that were blamed on that prompt | **18 of 22** (288 of 352 slow decode gaps) |
| Slow steps left unexplained (no co-scheduled prefill) | **11 of 30**, 4 of them inside a prefill window |
| Slow steps or gaps blamed on a long prompt outside every prefill window | **0** |
| Share of all excess decode latency attributed to the long prompts | **92.9%** (other requests 0.2%, unexplained 6.9%) |
| GPU busy time of the steps that produced the slow tokens | 43 ms median, against a 5.4 ms typical decode gap |
| CPU/GPU clock bound, measured in the same run | **280 µs** (59 paired probes, no violations) |
| Records captured / lost | 2.06M records, 1.69M of them GPU activities (99.0% linked to steps) / **0 lost** |

The 16 decodes run in lockstep, so one slow engine step shows up as up to
16 slow gaps: the 465 slow decode gaps come from 30 steps
([results/interference-steps.json](results/interference-steps.json) scores
them per step). The 11 unexplained steps had 4.6–6.6 ms of GPU busy time
against 8–15 ms gaps, and the four inside a prefill window are within two
steps of that prompt's prefill chunks. `blame` does not say what slowed
them.

"0 lost": the capture's sequence numbers are contiguous and it holds no
`Loss` records. `blame` still reports `loss_records: 1` with `loss_count: 0`.
That entry is the capture's missing end marker: vLLM SIGKILLs EngineCore
right after the final flush, so no `Finalized` session record is written,
and the analyzer records the open end as a loss of unknown size.

## Getting the measurement right

The first real run produced plausible-looking output that was wrong in
several ways. Each fix below was found by checking the capture against the
client's ground truth or against CUPTI's own counters.

| Symptom | Cause | Fix |
|---|---|---|
| Every request's time was "residual" | No clock bound was ever written for GPU timestamps | A probe thread brackets a 4-byte copy between two clock reads every 500 ms; the analyzer turns the brackets into a session bound |
| The probe never ran, and nothing said so | `dlsym(RTLD_DEFAULT)` cannot see `libcuda`, which the CUDA runtime loads `RTLD_LOCAL` | Take the loaded copy with `dlopen(RTLD_NOLOAD)`, and emit a `Diagnostic` record when the probe cannot run |
| Slow tokens blamed on the wrong step | Under async scheduling, step k+1 is launched before step k's token is processed, so time windows pick the next step | The plugin tags each `SchedulerOutput` with its step id, and each token records the step that produced it |
| In a 22.7 s run, CUPTI found no free buffer 2.2M times and dropped 460k records | The writer fdatasynced every 1 MiB segment on the one thread draining CUPTI buffers | Rotation only writes; dump, flush and finalize sync |
| Still 1.5M records dropped by CUPTI in 22.7 s | Most API records were query calls (`cudaGetDevice` alone was half of 1.8M, then `cuDevicePrimaryCtxGetState` and `cuPointerGetAttribute`) issued by torch and Triton around every launch | Record only APIs that submit work, copy or synchronize, using a per-callback CUPTI whitelist |
| Every request lost its last token, and long decodes lost their whole lifecycle record | vLLM frees a finished request inside `update_from_output`; a 4,000-token log outgrew a 64 KiB queue slot; vLLM SIGKILLs EngineCore at shutdown | Record tokens before the update; flush token logs in 1,024-token chunks; flush the collector over its control socket before stopping the server |
| 14% of kernels and 69% of copies had no timestamps | The CUPTI device-buffer pool was capped at 4 per context, too few at vLLM's launch rate | Keep CUPTI's default pool; the analyzer counts zero timestamps as missing, never as t = 0 |

The counts come from earlier captures of the same experiment, summarized in
[results/captures.json](results/captures.json) (runs `interference-s1`,
`-s3` and `-s4`).

[docs/design.md](docs/design.md) explains the recording path, the clock
bound and the analysis; [docs/design-tradeoffs.md](docs/design-tradeoffs.md)
records why each design decision was made and what it costs, and
[docs/format.md](docs/format.md) specifies the segment format.

## How `blame` decides

```
GpuActivity --corr--> ApiCall --corr--> ExtCorr(step_id) --> Step --members--> Request
```

1. **Find slow gaps.** For each request, every inter-token gap above
   1.5× its median is slow. Its excess over the median needs explaining.
2. **Find the producing step.** The plugin recorded which step produced the
   token after the gap. If any chunk of a request's token log lacks step
   ids (the plugin omits them from a chunk with a token of unknown step),
   `blame` falls back, for that whole request, to its steps whose host span
   ended inside the gap. Under async scheduling that picks the step after
   the producing one, and the output does not say the fallback was used.
3. **Report that step's evidence:**
   - GPU busy time: the union of its linked kernels and copies, measured in
     GPU time;
   - its co-scheduled members, with phase and tokens.
4. **Assign the excess.** It is split over the prefill tokens co-scheduled
   in that step. With no co-scheduled prefill, the gap is reported as
   unexplained, never guessed.
5. **Rank culprits** by the total excess they caused.

```sh
reqscope blame RUN [--request ID]       # slow gaps, their steps, and the culprits
reqscope explain RUN --request ID       # one request's time split into gpu_exec, idle, not_scheduled, ...
reqscope export-perfetto RUN            # timeline for ui.perfetto.dev
```

## Overhead

Measured with `bench/interference.py overhead`: 32 closed-loop clients send
512-token prompts and take 128 output tokens each, for 60 s on the same
engine. Values are medians of three alternating rounds. "Trace written" is
from the first round's captures: the bytes the collector wrote during the
60 s load window ([results/captures.json](results/captures.json)).

| Collector | Output tokens/s | TPOT p50 | Trace written |
|---|---|---|---|
| none | 4,267 (4,254 – 4,274) | 6.69 ms | |
| light: graph envelopes + submission APIs | 4,226 (−1.0%) | 6.80 ms (+1.6%) | 6 MB/s |
| trace: every kernel + submission APIs | 4,078 (−4.4%) | 7.10 ms (+6.1%) | 35 MB/s |

- **Light mode** is cheap enough to leave on. It records each CUDA graph
  launch, but kernels launched outside a graph are not timed one by one.
- **Trace mode** gives per-kernel GPU time, which `blame` needs for step
  busy time. About half its cost appears to be CUPTI collecting a
  timestamp for every kernel: an earlier run with the 4-buffer pool, when
  many kernels went untimed, measured trace mode at 2.3%, but that run's
  data is not in `results/`.
- Both modes keep a bounded ring on disk, and `reqscope dump --pid` freezes
  the last N seconds.

## Analysis speed

On the 23 s, 0.9 GB capture above (2.06M records), with byte-identical
output before and after. These timings, and the analysis figures under
Limitations, were measured on the same host but are not recorded in
`results/`. The one analysis time there, `blame_seconds` (28.9 s) in
`results/interference.json`, was taken by `bench/score.py` with the
analyzer from before these changes.

| | Before | After |
|---|---|---|
| `blame`, whole capture | 28.8 s, 8.6 GB | **9.1 s, 4.2 GB** |
| `analyze`, its last 11 MB | 57 s, 2.9 GB | **2.2 s, 2.1 GB** |
| `analyze`, its last 27 MB | 157 s, 6.9 GB | **5.2 s, 4.8 GB** |

| Cost | Cause | Fix |
|---|---|---|
| 5.5 of the 18 s spent reading the capture | A bitwise CRC32 over every frame | A slicing-by-8 table; the collector writes with it too |
| Another 7 s of reading | One read and one allocation per frame, and a sort whose comparator copied two producer strings on each of 40M comparisons | One read per segment; ordering keys taken while parsing |
| 2.3 KB per record in memory | JSON objects as `std::map`, and values sized by doubling | Members in one sorted vector, values sized exactly |
| A quarter of `blame`'s memory | It built the dependency graph that only `analyze` reports | Built only when reported |
| `analyze` grew with the square of the trace | Every elementary interval of a request rescanned all of its GPU activities, every token all intervals, every step all activities and syncs | One sweep over the intervals, a merge for the tokens, per-producer indexes |

## Run it

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
REQSCOPE_BIN=build/reqscope REQSCOPE_LIBRARY=build/libreqscope.so python3 -m unittest discover -s tests

# The experiment above (vLLM 0.31.0 in this project's environment; holds the GPU lock)
python bench/interference.py interference --mode trace --out results/raw/interference
REQSCOPE_TRACE_BUDGET_MIB=8192 python bench/score.py interference results/raw/interference  # above the reader's 2 GiB default
python bench/score_steps.py results/raw/interference
for m in off light trace; do python bench/interference.py overhead --mode $m --out results/raw/overhead-$m; done
python bench/score.py overhead results/raw/overhead-{off,light,trace}
python bench/capture_stats.py results/raw/interference results/raw/overhead-{light,trace}
```

`bench/interference.py` shows the whole setup:
- `CUDA_INJECTION64_PATH` and `LD_PRELOAD` of the CUPTI library;
- `VLLM_PLUGINS=reqscope`;
- `REQSCOPE_MODE` and `REQSCOPE_OUTPUT`;
- flushing every collector in the server's process tree before shutdown.

Results from the runs above are in `results/`;
[results/README.md](results/README.md) lists which run and script produced
each file.

The 146 tests run on CPU. They cover:
- the format reader;
- the collector's C ABI, against a manual backend;
- the analyzer: linking, clock probes, blame and decomposition;
- the vLLM plugin's contracts, against stubbed runner and scheduler
  classes;
- CUDA DAG fixtures (streams, copies, graph replay and mutation) checked
  against CUPTI recordings.

## Limitations

- **Pinned versions.** CUPTI API 130202 (CUDA 13.2) and vLLM 0.31.0, with
  the V2 model runner, the uni executor and TP = PP = 1. Other versions are
  refused instead of parsed with the wrong layout.
- **One CUPTI subscriber**, so it cannot run together with Nsight Systems
  or `torch.profiler`.
- **`analyze` and `explain` emit every elementary interval.** Their time now
  grows linearly with the trace, but each request's decomposition holds one
  interval per gap between GPU activity boundaries, so the last 59 MB of the
  capture above already makes 1 GB of JSON and 10.6 GB of memory. On a full
  long capture, use `blame`, or analyze a shorter window.
- **JSON records.** Reading the capture above is 3.7 s of `blame`'s 9.1 s.
  A binary encoding of the four record types that make up 99% of a trace
  is the next step.
- **Attribution is linear.** `blame` splits a gap's excess over co-scheduled
  prefill tokens. It does not model attention cost growing with context
  length.
- **One experiment shape.** Prefill interference is validated end to end
  against ground truth. Other causes are not: a co-running GPU job, or
  preemption and recompute.
- **The BPF scheduler probe** (`bpf/`) needs CAP_BPF and was not used here,
  so host-side CPU stalls are reported as `between_steps`, without
  run-queue evidence.
