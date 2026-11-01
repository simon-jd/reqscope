# reqscope: why was this vLLM request slow?

When a request's tokens slow down in a batched LLM server, the cause can be
another request's prefill sharing the engine step, the GPU running slower or
running someone else's work, or the host failing to keep the GPU fed.
Serving metrics show that inter-token latency rose; GPU profilers show
kernels, but not which requests a step served. reqscope records both sides
of vLLM on one clock and answers, for each slow token: which engine step
produced it, what else ran in that step, how its time split between GPU and
host, and what caused the excess.

**How it works.** A vLLM 0.31.0 plugin records every engine step, the
requests it scheduled and their phase, and each token's time and producing
step. A CUPTI injection library (C++20, loaded with `CUDA_INJECTION64_PATH`,
no change to vLLM or PyTorch) tags every kernel and copy with the step that
launched it, on a clock with a measured error bound; it is optional.
`reqscope blame` joins the records by ids, names each slow step's cause
(interference, GPU slowdown or host delay) from step membership and GPU
evidence, and splits interference excess over the co-scheduled prefill.

**Results** (Qwen2.5-1.5B-Instruct on one RTX 4090, vLLM 0.31.0, chunked
prefill, async scheduling):
- **It tells causes apart, blind.** During steady decoding, a seeded
  schedule the analyzer never sees injects 6,000-token prompts, 300 ms
  bursts of matrix multiplies from a separate CUDA process, and 200 ms stops
  of the engine process. Over 3 seeds and 27 events, trace-mode blame named
  the right cause for every event. Per slow step, recall is 0.84 / 0.99 /
  1.00 for interference / GPU slowdown / host delay, and precision inside
  the injection windows 1.00 / 1.00 / 0.60.
- **Without GPU data it cannot.** With CUPTI off, blame still finds all 9
  long prompts but calls every GPU burst and engine stop "undetermined" (0
  of 18). In a single-cause experiment membership alone is enough: six long
  prompts against 16 decodes are found with per-step precision 1.00 and
  recall 0.82, with or without CUPTI.
- **Recording steps is free; tracing every kernel is not.** Under open-loop
  load at 8–24 requests/s, the plugin alone and the light collector stay
  within 1% of no collector on TPOT p50 and p99. Per-kernel trace mode adds
  2–3% at 8 requests/s, 18% (p50) and 27% (p99) at 16, and at 24 the
  server falls behind (median TTFT 26 to 314 ms). Tracing also adds slow
  steps of its own: 415 outside the injection windows across the three
  trace runs, against 39 with CUPTI off.

Limits: one model on one GPU at TP = 1, injected causes rather than
production incidents, synchronization stalls not separated from host delay,
vLLM 0.31.0 and CUDA 13.2 only, and a shared host whose load varied between
runs; see [Limitations](#limitations).

- **vLLM plugin** (`adapter/vllm`): wraps the V2 model runner and the
  scheduler to emit steps, step membership and request lifecycles.
- **Collector** (`collector/`): the CUPTI library, plus a plugin-only mode
  with CUPTI never started.
- **Analyzer and CLI** (`analyzer/`, `cli/`): `blame`, `explain`, `analyze`,
  `diff`, `export-perfetto`, `dump`.

## The multi-cause benchmark

`bench/interference.py multicause`: 16 requests decode 6,000 tokens each.
From 2 s on, an event starts every 1.5–3 s, nine in all, three of each
cause in an order drawn from the seed:
- **interference**: a 6,000-token prompt, whose window runs from sending it
  to its first token;
- **GPU slowdown**: a separate CUDA process, started by the benchmark while
  it holds the GPU lock, runs fp16 8192 × 8192 matrix multiplies for 300 ms;
- **host delay**: the EngineCore process is stopped with SIGSTOP for 200 ms.

The benchmark writes each event's actual window to `client.json`, which
`blame` never reads. `bench/score_causes.py` groups each run's slow decode
gaps by producing step (the decodes run in lockstep) and labels each step
with the event window it overlaps most, extended by 20 ms, or with "no
injected cause". Three seeds ran with trace mode and three with the plugin
alone. The cause rules ([docs/design.md](docs/design.md), "Causes") were
fixed on a pilot run with another seed before these runs. Other projects'
jobs kept the host's load average at 6.5–10.9 during them (per run in
[results/multicause.json](results/multicause.json)).

| Injected cause | Trace: events found | Trace: step recall | Trace: step precision inside windows | Plugin only: events found | Plugin only: step recall |
|---|---|---|---|---|---|
| interference | 9 of 9 | 0.84 (27 of 32) | 1.00 | 9 of 9 | 0.93 (27 of 29) |
| GPU slowdown | 9 of 9 | 0.99 (215 of 216) | 1.00 | 0 of 9 | 0 (all 220 undetermined) |
| host delay | 9 of 9 | 1.00 (9 of 9) | 0.60 | 0 of 9 | 0 (all 9 undetermined) |

Slow steps by true cause (rows) and blame's cause (columns), summed over
the three seeds ([results/multicause.json](results/multicause.json)):

| Trace mode | interference | GPU slowdown | host delay | unexplained | undetermined |
|---|---|---|---|---|---|
| interference | 27 | 0 | 5 | 0 | 0 |
| GPU slowdown | 0 | 215 | 1 | 0 | 0 |
| host delay | 0 | 0 | 9 | 0 | 0 |
| no injected cause | 3 | 0 | 412 | 0 | 0 |

| Plugin only | interference | GPU slowdown | host delay | unexplained | undetermined |
|---|---|---|---|---|---|
| interference | 27 | 0 | 0 | 0 | 2 |
| GPU slowdown | 0 | 0 | 0 | 0 | 220 |
| host delay | 0 | 0 | 0 | 0 | 9 |
| no injected cause | 3 | 0 | 0 | 0 | 36 |

The evidence behind the verdicts (medians over the trace runs' slow steps):
during a GPU burst, a decode step's GPU busy time doubles (10.4 ms against
5.1 ms in nearby decode steps) and its first kernel starts 8.7 ms after its
launch returns instead of 4.1 ms. During an engine stop the producing step's
GPU work is normal (5.6 ms) and the gap holds 198 ms of host time after it.

What does not work:
- **Steps near a long prompt that are not its prefill.** 5 of the 32 slow
  steps inside long-prompt windows had no co-scheduled prefill and were
  called host delay (3.6–12 ms of extra host time), probably the engine
  taking in the 6,000-token request before scheduling its chunks. The
  window-based truth counts them as interference misses, and they are most
  of why host-delay precision inside the windows is 0.60.
- **Tracing's own slow steps.** Outside the windows the trace runs have 415
  slow steps against 39 with CUPTI off under similar host load, and blame
  calls 412 of them host delay. That matches what tracing costs (see "With CUPTI off" below), but
  nothing here injects them, so against the injected truth host-delay
  precision over all steps is 0.02.
- **The 3 "no injected cause" steps called interference** are each run's
  first slow step, when the 16 decodes' own prompts are prefilled together:
  real interference, but not injected.
- **Synchronization stalls** are not a separate cause: the rules count a
  host-device sync as host time, and the benchmark does not inject one.

## The interference experiment

16 requests decode continuously, each to 4,000 tokens. Six 6,000-token
prompts arrive, one every 3 s, and their prefill chunks share engine steps
with the decodes. The client records when each long prompt was in prefill
(sent until first token). The analyzer never sees that ground truth.

![Slow engine steps over the run: the steps blamed on a long prompt sit in
its prefill window; the unexplained ones are 8-15 ms decode
steps](docs/slow-steps.svg)

| What `reqscope blame` reported | |
|---|---|
| Top six culprits by excess latency caused | **exactly the six long prompts** |
| Per slow engine step: precision / recall of "blamed on the long prompt whose prefill window it is in" | **1.00 / 0.82** (18 of 18 / 18 of 22 steps; per slow decode gap 288 of 288 / 288 of 352) |
| Slow steps left unexplained (no co-scheduled prefill) | **11 of 30**, 4 of them inside a prefill window |
| Slow steps or gaps blamed on a long prompt outside every prefill window | **0** |
| Share of all excess decode latency attributed to the long prompts | **92.9%** (other requests 0.2%, unexplained 6.9%) |
| GPU busy time of the steps that produced the slow tokens | 43 ms median, against a 5.4 ms typical decode gap |
| CPU/GPU clock bound, measured in the same run | **280 µs** (59 paired probes, no violations) |
| Records captured / lost | 2.06M records, 1.69M of them GPU activities (99.0% linked to steps) / **0 lost** |

The 16 decodes run in lockstep, so one slow engine step shows up as up to
16 slow gaps: the 465 slow decode gaps come from 30 steps
([results/interference-steps.json](results/interference-steps.json) scores
them per step). The 11 unexplained steps are plain decode steps: the 15
other decodes co-scheduled, no prefill, and 4.6–6.6 ms of GPU busy time,
about an ordinary 5.4 ms decode step. The rest of each 8–15 ms gap, 2–10 ms,
is not GPU work of that step. Three of them have long host spans (5.7–7.4
ms from `execute_model` to the end of `sample_tokens`); the other eight
have short ones (1.0–3.5 ms), so their delay lies between steps, in
scheduling or output processing, which `blame` does not attribute and the
BPF probe (not used here) would observe. The four inside a prefill window
are within two steps of that prompt's prefill chunks.

"0 lost": the capture's sequence numbers are contiguous and it holds no
`Loss` records. `blame` still reports `loss_records: 1` with `loss_count: 0`.
That entry is the capture's missing end marker: vLLM SIGKILLs EngineCore
right after the final flush, so no `Finalized` session record is written,
and the analyzer records the open end as a loss of unknown size.

This run was captured with an earlier build of the collector; the
comparison below repeats the experiment with the current one.

## With CUPTI off: what GPU data adds, and what tracing costs

The same workload ran in back-to-back pairs: trace mode, then the plugin
alone with CUPTI never started (`--mode plugin`). The host is shared with
other projects' jobs, so its 1-minute load average is recorded per run.

| Run | Collector | Host load | Slow steps | In-window steps blamed on that prompt (recall) | Precision | Unexplained steps | Excess on long prompts |
|---|---|---|---|---|---|---|---|
| headline run (earlier build) | trace | not recorded | 30 | 18 of 22 (0.82) | 1.00 | 11 | 92.9% |
| pair 1 | trace | about 9–10 | 129 | 18 of 25 (0.72) | 1.00 | 110 | 58.5% |
| pair 1 | plugin only | about 9–10 | 33 | 18 of 21 (0.86) | 1.00 | 14 | 91.9% |
| pair 2 | trace | 1.7 | 71 | 18 of 26 (0.69) | 1.00 | 52 | 74.9% |
| pair 2 | plugin only | 1.9 | 27 | 18 of 24 (0.75) | 1.00 | 8 | 96.2% |

Pair 1's load comes from `uptime` during the pair (8.8 over 1 minute, 10.4
over 5); the per-run sampler started after it. Per-run details, including
the medians below, are in [results/ablation.json](results/ablation.json).

- **The verdict is the same without GPU data.** Every run blames exactly 18
  steps on a long prompt, the six prompts' three prefill chunks each, all
  inside that prompt's window, and the top six culprits are always the six
  prompts. `blame` assigns excess from the plugin's step membership alone.
- **GPU data says what kind of slow step it was.** With CUPTI, the blamed
  steps show about 49–50 ms of GPU busy time in gaps of the same length: the GPU was doing
  the prefill. The unexplained steps show 6.0–6.3 ms, an ordinary decode
  step, in 9–10 ms gaps (medians per run), so the rest of the gap was host time.
  With CUPTI off, both kinds look only like long gaps.
- **Tracing changes what it measures.** In both pairs trace mode had many
  more slow steps than the plugin alone (129 against 33, 71 against 27).
  The extra ones are decode steps with ordinary GPU time and longer gaps:
  host-side delay. Pair 2 ran on a quiet host (load 1.7–1.9), so other
  jobs do not explain the difference; tracing does, by an amount that
  varies between runs (the headline trace run had 30). The open-loop
  overhead below shows the same cost as inter-token latency. The longer
  host span of the blamed steps in pair 1 (17.7 ms against 5.7 ms) did not
  reproduce on the quiet host (9.4 against 8.4 ms), so it more likely came
  from the load of other jobs.

What each mode is for:
- **plugin only:** always-on attribution of interference between requests.
  No measurable cost, the same culprits, but a host stall and GPU work look
  alike.
- **light:** also within 1% on TPOT; adds GPU time per CUDA graph launch
  (graph envelopes), which this experiment did not validate.
- **trace:** per-kernel GPU time that separates GPU-bound from host-bound
  slow steps. Use it for short captures at moderate load: it adds slow
  steps of its own, and at high load it slows the server down.

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
   the producing one, so each request row says which rule it used
   (`producing_steps`) and the output carries a warning.
3. **Report that step's evidence:**
   - GPU busy time: the union of its linked kernels and copies, measured in
     GPU time (none with CUPTI off);
   - its co-scheduled members, with phase and tokens.
4. **Assign the excess.** It is split over the prefill tokens co-scheduled
   in that step. With no co-scheduled prefill, no request is blamed.
5. **Name the cause:** interference when the step had co-scheduled prefill;
   otherwise GPU slowdown or host delay, whichever of the step's extra GPU
   time and extra host time explains at least half the excess, against
   nearby comparable steps; "unexplained" when neither does, and
   "undetermined" without GPU records
   ([docs/design.md](docs/design.md), "Causes").
6. **Rank culprits** by the total excess they caused.

```sh
reqscope blame RUN [--request ID]       # slow gaps, their steps, and the culprits
reqscope explain RUN --request ID       # one request's time split into gpu_exec, idle, not_scheduled, ...
reqscope export-perfetto RUN            # timeline for ui.perfetto.dev
```

## Overhead

**Open loop.** `bench/interference.py openloop`: Poisson arrivals of
512-token prompts with 128 output tokens at 8, 16 and 24 requests/s, 40 s
per rate on one server, against a closed-loop capacity of about 33
requests/s. Five rounds with the mode order rotated each round; each cell
is the median over rounds with the range. The host's load average during
the runs was 1.1–4.7 (per run in [results/openloop.json](results/openloop.json)).

| Rate | Collector | TTFT p50, ms | TTFT p99, ms | TPOT p50, ms | TPOT p99, ms | Output tokens/s | TPOT p50 / p99 vs none |
|---|---|---|---|---|---|---|---|
| 8/s | none | 23.3 (22.9–23.4) | 37.5 (37.2–41.2) | 5.12 (5.11–5.13) | 5.45 (5.44–5.60) | 967 (967–968) |  |
| 8/s | plugin only | 23.1 (22.8–23.3) | 37.0 (36.2–39.3) | 5.11 (5.11–5.12) | 5.44 (5.41–5.46) | 968 (968–968) | -0.2% / -0.2% |
| 8/s | light | 22.7 (22.6–22.8) | 37.6 (35.4–38.5) | 5.13 (5.12–5.13) | 5.46 (5.44–5.47) | 968 (967–968) | +0.2% / +0.2% |
| 8/s | trace | 22.9 (22.4–23.3) | 38.2 (37.3–38.5) | 5.21 (5.19–5.22) | 5.62 (5.53–5.70) | 967 (967–967) | +1.8% / +3.1% |
| 16/s | none | 23.8 (23.6–24.1) | 44.4 (41.2–50.5) | 5.29 (5.28–5.29) | 6.09 (6.06–6.10) | 1,918 (1,918–1,919) |  |
| 16/s | plugin only | 23.8 (23.4–23.9) | 44.0 (43.4–45.8) | 5.29 (5.27–5.29) | 6.10 (6.08–6.15) | 1,918 (1,918–1,919) | +0.0% / +0.2% |
| 16/s | light | 23.5 (23.4–23.6) | 43.5 (40.2–44.5) | 5.30 (5.29–5.30) | 6.13 (6.11–6.14) | 1,919 (1,918–1,919) | +0.2% / +0.7% |
| 16/s | trace | 26.4 (26.3–27.1) | 47.1 (45.6–50.4) | 6.25 (6.18–6.35) | 7.76 (7.69–10.3) | 1,911 (1,910–1,912) | +18.1% / +27.4% |
| 24/s | none | 26.4 (26.2–27.1) | 130 (126–134) | 6.19 (6.18–6.19) | 7.32 (7.29–7.33) | 3,155 (3,154–3,155) |  |
| 24/s | plugin only | 26.2 (25.2–26.3) | 128 (125–137) | 6.18 (6.17–6.19) | 7.34 (7.31–7.36) | 3,155 (3,155–3,155) | -0.2% / +0.3% |
| 24/s | light | 25.9 (25.8–26.2) | 135 (133–152) | 6.22 (6.21–6.23) | 7.36 (7.34–7.41) | 3,155 (3,155–3,155) | +0.5% / +0.5% |
| 24/s | trace | 314 (275–417) | 2,049 (1,814–2,574) | 9.87 (9.75–9.87) | 11.1 (10.5–12.1) | 3,003 (2,977–3,028) | +59.5% / +51.0% |

- **Plugin only and light** are within 1% of no collector on TPOT at every
  rate. Their TTFT medians are within 3%, except light's p99 at 24/s (+4%,
  with overlapping ranges).
- **Trace** costs 2–3% at 8 requests/s and 18% (TPOT p50) to 27% (p99) at
  16. At 24 the engine no longer keeps up: throughput falls 4.8%, requests
  queue, and median TTFT rises from 26 to 314 ms.

**Closed loop.** `bench/interference.py overhead`: 32 closed-loop clients
send 512-token prompts and take 128 output tokens each, for 60 s on the
same engine. Medians of three alternating rounds ([results/overhead.json](results/overhead.json)).
"Trace written" is from the first round's captures: the bytes the
collector wrote during the 60 s load window ([results/captures.json](results/captures.json)).

| Collector | Output tokens/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Trace written |
|---|---|---|---|---|---|---|
| none | 4,267 (4,254 – 4,274) | 108 ms | 233 ms | 6.69 ms | 7.39 ms | |
| light: graph envelopes + submission APIs | 4,226 (−1.0%) | 99 ms | 222 ms | 6.80 ms (+1.6%) | 7.43 ms (+0.5%) | 6 MB/s |
| trace: every kernel + submission APIs | 4,078 (−4.4%) | 99 ms | 201 ms | 7.10 ms (+6.1%) | 8.01 ms (+8.4%) | 35 MB/s |

In closed loop, median TTFT is lower with a collector on (108 to 99 ms),
probably because with a fixed number of clients a slower engine also
receives new prompts at a lower rate. Closed-loop TTFT therefore does not
measure the collector's cost; the open-loop numbers above do.

- **Light mode** records each CUDA graph launch, but kernels launched
  outside a graph are not timed one by one.
- **Trace mode** gives per-kernel GPU time, which `blame` needs for step
  busy time. About half its cost appears to be CUPTI collecting a
  timestamp for every kernel: an earlier run with the 4-buffer pool, when
  many kernels went untimed, measured trace mode at 2.3%, but that run's
  data is not in `results/`.
- All modes keep a bounded ring on disk, and `reqscope dump --pid` freezes
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

## Reproduce

Needs Linux, one NVIDIA GPU, the CUDA 13.2 toolkit (for CUPTI API 130202),
CMake 3.20 or newer, a C++20 compiler and Python 3.12. The versions behind
the committed results are in
[results/environment.json](results/environment.json).

```sh
git clone https://github.com/simon-jd/reqscope.git && cd reqscope
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure        # C++ and Python tests, CPU only

python3.12 -m venv .venv && . .venv/bin/activate
pip install -r bench/requirements.txt && pip install -e adapter/vllm

# Blind multi-cause benchmark: three seeds with trace mode and with the plugin
# alone; each run holds the GPU lock and starts its own GPU burner
for s in 1 2 3; do for m in trace plugin; do
  python bench/interference.py multicause --mode $m --seed $s --out results/raw/multicause-$m-s$s
done; done
python bench/score_causes.py results/raw/multicause-*-s?

# Interference experiment: one vLLM server, a few minutes on the GPU.
# The model is Qwen/Qwen2.5-1.5B-Instruct at a pinned revision.
python bench/interference.py interference --mode trace --out results/raw/interference
python bench/score.py interference results/raw/interference
python bench/score_steps.py results/raw/interference

# The same workload with CUPTI off, compared with the trace run
python bench/interference.py interference --mode plugin --out results/raw/ablation-plugin
python bench/score.py interference results/raw/ablation-plugin
python bench/compare_runs.py results/raw/interference results/raw/ablation-plugin

# Open-loop overhead: 5 rounds, mode order rotated each round
modes=(off plugin light trace)
for r in 1 2 3 4 5; do for k in 0 1 2 3; do m=${modes[$(( (k + r - 1) % 4 ))]}
  python bench/interference.py openloop --mode $m --round $r --rates 8,16,24 --seconds 40 --out results/raw/openloop-$m-r$r
done; done
python bench/score.py openloop results/raw/openloop-*

# Closed-loop overhead, one server per collector mode
for m in off light trace; do python bench/interference.py overhead --mode $m --out results/raw/overhead-$m; done
python bench/score.py overhead results/raw/overhead-{off,light,trace}
python bench/capture_stats.py results/raw/interference results/raw/overhead-{light,trace}
```

`bench/interference.py --help` lists every setting, each with a default
that needs no local path: `--cuda` (toolkit root, default `$CUDA_HOME` or
`/usr/local/cuda`), `--build` (default `build/`), `--python` (the
environment with vLLM and the plugin, default the running interpreter),
`--model` and `--revision` (a Hub id, or a local model directory),
`--hf-home`, `--offline`, `--cache-root`, `--gpu-memory-utilization`
(default 0.6) and `--lock FILE`, which holds that file's flock from server
start to shutdown so runs on a shared GPU do not overlap. Each run records
the host's load average in its `client.json`. `bench/score.py`
raises the trace reader's budget to 8 GiB, which a 0.9 GB capture needs.

The script also shows the whole setup:
- `CUDA_INJECTION64_PATH` and `LD_PRELOAD` of the CUPTI library;
- `VLLM_PLUGINS=reqscope`;
- `REQSCOPE_MODE` and `REQSCOPE_OUTPUT`;
- flushing every collector in the server's process tree before shutdown.

Results from the runs above are in `results/`;
[results/README.md](results/README.md) lists which run and script produced
each file.

The tests run on CPU. They cover:
- the format reader;
- the collector's C ABI, against a manual backend, including a writer
  failure under a file-size limit;
- the clock probe, against a fake CUDA driver;
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
- **Injected causes, one model, one GPU.** Prefill interference, a
  co-running CUDA process and a stopped engine are validated against ground
  truth, on one model and one GPU at TP = 1. A real CPU contention, a GPU
  clock drop, preemption and recompute, or two causes in one step are not.
  Each slow step gets one cause.
- **Host delay is one class.** A host-device synchronization stall, a
  garbage-collection pause and CPU contention all look like host time
  around the step's GPU work; `blame` does not tell them apart, and the
  BPF probe that could was not used.
- **Trace mode perturbs what it measures.** It added slow decode steps of
  its own in the interference runs, by an amount that varied between runs
  (30, 71 and 129 slow steps against 27 and 33 with CUPTI off), 415 against
  39 outside the injection windows of the multi-cause runs, and it slows a
  loaded server (open-loop table). Its per-kernel evidence is for short
  captures at moderate load.
- **A shared host.** The runs shared the machine with other projects' jobs.
  Every run held the GPU lock those jobs share, but host load varied
  (1-minute load average 1.1–4.7 in the open-loop runs, 6.5–10.9 in the
  multi-cause runs, about 9–10 in the first CUPTI-off pair), and host load
  changes host-side timings.
- **The BPF scheduler probe** (`bpf/`) needs CAP_BPF and was not used here,
  so host-side CPU stalls are reported as `between_steps`, without
  run-queue evidence.
