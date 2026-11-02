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
  the right cause for every event. Per slow step, recall is 0.77 / 1.00 /
  1.00 for interference / GPU slowdown / host delay, and precision over all
  slow steps 0.90 / 1.00 / **0.04**: trace mode adds slow steps of its own
  (231 outside the injection windows), and blame calls almost all of them
  host delay. Counting only steps inside the windows, host-delay precision
  is 0.56.
- **The GPU data's whole contribution is GPU slowdowns.** On the same
  runs, a trivial rule (any co-scheduled prefill means interference,
  anything else host delay) gets every interference verdict and every
  verdict on trace mode's own slow steps the same as blame; with GPU data,
  blame changes only the 223 steps during GPU bursts, from host delay to
  GPU slowdown.
- **Nsight Systems does as well.** In three runs under Nsight Systems,
  with the plugin's NVTX step ranges, a simple GPU-time rule also named
  every event's cause, and those runs had 98 slow steps of their own
  against trace mode's 231. vLLM's own per-step log separates GPU bursts
  but calls 6 of 9 engine-stop steps GPU slowdown.
- **Without GPU data it cannot.** With CUPTI off, blame still finds all 9
  long prompts but calls every GPU burst and engine stop "undetermined" (0
  of 18). In a single-cause experiment membership alone is enough: six long
  prompts against 16 decodes are the top six culprits in every run, with
  per-step precision 1.00 and recall 0.69–0.86 across four paired runs
  with and without CUPTI (0.82 in an earlier single run).
- **Recording steps is free; tracing every kernel is not.** Under open-loop
  load at 8–24 requests/s, the plugin alone and the light collector stay
  within 1% of no collector on TPOT p50 and p99 (earlier build). Per-kernel
  trace mode adds 1.6% / 3.7% to TPOT p50 / p99 at 8 requests/s and 17% /
  28% at 16 (current build; an earlier build fell behind at 24). It also
  adds slow steps of its own: 231 outside the injection windows across
  three trace runs, against 6 with CUPTI off. A faster writer and larger
  CUPTI device buffers halved those (from 503 in the same session); their
  effect on latency is within the noise of five rounds; see
  [Trace mode's own cost](#trace-modes-own-cost).

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
injected cause". The cause rules ([docs/design.md](docs/design.md),
"Causes") were fixed on a pilot run with another seed. The runs below used
the current collector, three seeds with trace mode and three with the
plugin alone, interleaved with three trace runs of the collector before
its writer and device-buffer changes (commit 3ee45a6, "before"); host load
averaged 2.3–3.9 ([results/multicause-after.json](results/multicause-after.json),
[results/multicause-before.json](results/multicause-before.json)). An
earlier round on a busier host (load 6.5–10.9, commit 60fca57) is in
[results/multicause.json](results/multicause.json).

| Injected cause | Trace: events found | Trace: step recall | Trace: step precision, all steps | Trace: step precision inside windows | Plugin only: events found | Plugin only: step recall |
|---|---|---|---|---|---|---|
| interference | 9 of 9 | 0.77 (27 of 35) | 0.90 (27 of 30) | 1.00 | 9 of 9 | 0.84 (27 of 32) |
| GPU slowdown | 9 of 9 | 1.00 (223 of 223) | 1.00 | 1.00 | 0 of 9 | 0 (all 221 undetermined) |
| host delay | 9 of 9 | 1.00 (10 of 10) | **0.04** (10 of 246) | 0.56 | 0 of 9 | 0 (all 9 undetermined) |

Slow steps by true cause (rows) and blame's cause (columns), summed over
the three seeds:

| Trace mode | interference | GPU slowdown | host delay | unexplained | undetermined |
|---|---|---|---|---|---|
| interference | 27 | 0 | 8 | 0 | 0 |
| GPU slowdown | 0 | 223 | 0 | 0 | 0 |
| host delay | 0 | 0 | 10 | 0 | 0 |
| no injected cause | 3 | 0 | 228 | 0 | 0 |

| Plugin only | interference | GPU slowdown | host delay | unexplained | undetermined |
|---|---|---|---|---|---|
| interference | 27 | 0 | 0 | 0 | 5 |
| GPU slowdown | 0 | 0 | 0 | 0 | 221 |
| host delay | 0 | 0 | 0 | 0 | 9 |
| no injected cause | 3 | 0 | 0 | 0 | 3 |

**Against a trivial rule.** The rows below score the same slow steps of
the same runs in the same way; only the cause each step gets differs
(`bench/score_causes.py --rule`,
[results/multicause-baselines.json](results/multicause-baselines.json)).
The trivial rule calls a slow step interference if it had any co-scheduled
prefill, and host delay otherwise. Precision is over all slow steps.

| Rule | Runs | Interference recall / precision | GPU slowdown recall / precision | Host delay recall / precision | Events found |
|---|---|---|---|---|---|
| blame | trace | 0.77 / 0.90 | 1.00 / 1.00 | 1.00 / 0.04 | 27 of 27 |
| trivial | trace | 0.77 / 0.90 | 0 / — | 1.00 / 0.02 | 18 of 27 |
| blame | plugin only | 0.84 / 0.90 | 0 / — | 0 / — | 9 of 27 |
| trivial | plugin only | 0.84 / 0.90 | 0 / — | 1.00 / 0.04 | 18 of 27 |

What the GPU data adds over the trivial rule, on the same trace runs, is
exactly one thing: the 223 slow steps during GPU bursts, which the trivial
rule calls host delay, are called GPU slowdown, all of them correctly.
That is why host-delay precision inside the windows rises from 0.04 to
0.56. It changes no other verdict:
- **Interference** is the same step for step (27 found, 8 missed, 3 false).
  The 8 misses next to a long prompt have no co-scheduled prefill, and GPU
  data does not change that. On plugin-only runs, blame gives the same
  interference verdicts and calls everything else undetermined.
- **Trace mode's own slow steps**: blame calls all 228 of them host delay,
  as the trivial rule does. Over all steps, host-delay precision with GPU
  data (0.04) is no better than the trivial rule without it (0.04 on the
  plugin-only runs, which do not have those steps).

**Against Nsight Systems and vLLM's own step log.** Three more runs with the
same seeds used the plugin alone, without reqscope's CUPTI library, under
Nsight Systems 2025.6. It traced CUDA with one row per graph launch, and
the plugin marked each step with an NVTX range; vLLM's own per-step log was
on (`--nsys --vllm-iteration-log`,
[results/multicause-nsys.json](results/multicause-nsys.json)).
`checker/nsys_projection.py` matched every GPU row of each run (about
149,000) to a step, and 6,006 of vLLM's 6,009 log lines per run agreed with
the plugin's steps. Both kinds of data get one simple rule: interference if
the step had prefill; otherwise GPU slowdown if the step's time (GPU busy
time from Nsight Systems, or vLLM's "iteration elapsed time") exceeds the
run's median by at least half the step's excess; host delay otherwise. This
is not blame's rule, which compares with nearby comparable steps and also
measures host time. All rows are the same three runs, scored as above. The
host was busier than during the trace runs (load average 5.8–8.9 against
2.3–3.9).

| Rule, on the Nsight Systems runs | Interference recall / precision | GPU slowdown recall / precision | Host delay recall / precision | Events found |
|---|---|---|---|---|
| Nsight Systems GPU busy time | 0.84 / 0.90 | 0.98 / 1.00 | 1.00 / 0.08 | 27 of 27 |
| vLLM iteration elapsed time | 0.84 / 0.90 | 0.95 / 0.95 | 0.33 / 0.03 | 21 of 27 |
| trivial | 0.84 / 0.90 | 0 / — | 1.00 / 0.03 | 18 of 27 |
| blame, plugin only | 0.84 / 0.90 | 0 / — | 0 / — | 9 of 27 |

- **Nsight Systems' GPU data does as well as trace mode here.** Matched to
  steps through the plugin's NVTX ranges, it names the cause of every
  event, as trace-mode blame does, and misses 4 of the 218 GPU-burst steps.
  It added fewer slow steps of its own: 98 outside the windows over three
  runs, against trace mode's 231 on a quieter host and 6 with the plugin
  alone. So over all steps its host-delay precision is 0.08, against
  blame's 0.04 in trace mode. On this benchmark reqscope's own CUPTI
  tracing gives no better verdicts than Nsight Systems; both depend on the
  plugin's step and token records.
- **vLLM's per-step log** separates GPU bursts almost as well: its elapsed
  time is the engine's wait for the step's output, which grows when
  another process shares the GPU. But it calls 6 of the 9 engine-stop steps
  GPU slowdown, because a stop that falls inside that wait lengthens it the
  same way.

The evidence behind the correct verdicts (medians over the steps of the
three trace runs, `verdict_evidence_medians` in
[results/multicause-after.json](results/multicause-after.json)): during a
GPU burst, a decode step's GPU busy time doubles (10.4 ms against 5.1 ms
in comparable decode steps) and its shortest launch-to-kernel wait grows
from 3.5 to 9.2 ms. During an engine stop the producing step's GPU work is
normal (5.5 ms against 5.5 ms) and the gap holds 200 ms of extra host
time.

What does not work:
- **Steps near a long prompt that are not its prefill.** 8 of the 35 slow
  steps inside long-prompt windows had no co-scheduled prefill and were
  called host delay, probably the engine taking in the 6,000-token request
  before scheduling its chunks. The window-based truth counts them as
  interference misses, and they are why host-delay precision inside the
  windows is 0.56.
- **Tracing's own slow steps.** Outside the windows the trace runs have 231
  slow steps against 6 with CUPTI off, and blame calls 228 of them host
  delay, so against the injected truth host-delay precision over all steps
  is 0.04. The collector of 3ee45a6 had 503 such steps in the same session
  (490 called host delay, precision 0.03). What is left is CUPTI's own
  work inside launches ([Trace mode's own cost](#trace-modes-own-cost)).
- **The `collector_active` flag does not separate them.** blame flags a
  host delay when the collector was busy for at least half of its host
  extra. It flagged 22 of the 228 spurious host delays and also 2 of the
  10 real ones. What delays the engine now happens inside CUDA launch
  calls, where CUPTI does its work, and the collector does not see what
  CUPTI's own threads are doing.
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
scheduling or output processing, which `blame` does not attribute. The
four inside a prefill window
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

This section covers the plugin alone and light mode, measured with an
earlier build (commit 2c85655); trace mode's cost with the current build
is in [Trace mode's own cost](#trace-modes-own-cost).

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
| 16/s | none | 23.8 (23.6–24.1) | 44.4 (41.2–50.5) | 5.29 (5.28–5.29) | 6.09 (6.06–6.10) | 1,918 (1,918–1,919) |  |
| 16/s | plugin only | 23.8 (23.4–23.9) | 44.0 (43.4–45.8) | 5.29 (5.27–5.29) | 6.10 (6.08–6.15) | 1,918 (1,918–1,919) | +0.0% / +0.2% |
| 16/s | light | 23.5 (23.4–23.6) | 43.5 (40.2–44.5) | 5.30 (5.29–5.30) | 6.13 (6.11–6.14) | 1,919 (1,918–1,919) | +0.2% / +0.7% |
| 24/s | none | 26.4 (26.2–27.1) | 130 (126–134) | 6.19 (6.18–6.19) | 7.32 (7.29–7.33) | 3,155 (3,154–3,155) |  |
| 24/s | plugin only | 26.2 (25.2–26.3) | 128 (125–137) | 6.18 (6.17–6.19) | 7.34 (7.31–7.36) | 3,155 (3,155–3,155) | -0.2% / +0.3% |
| 24/s | light | 25.9 (25.8–26.2) | 135 (133–152) | 6.22 (6.21–6.23) | 7.36 (7.34–7.41) | 3,155 (3,155–3,155) | +0.5% / +0.5% |

- **Plugin only and light** are within 1% of no collector on TPOT at every
  rate. Their TTFT medians are within 3%, except light's p99 at 24/s (+4%,
  with overlapping ranges).
- **Trace mode of that build** (rows in the same file) fell behind at 24
  requests/s: throughput fell 4.8%, requests queued, and median TTFT rose
  from 26 to 314 ms. The current build was not run at 24 requests/s.

**Closed loop.** `bench/interference.py overhead`: 32 closed-loop clients
send 512-token prompts and take 128 output tokens each, for 60 s on the
same engine. Medians of three alternating rounds ([results/overhead.json](results/overhead.json)).
"Trace written" is from the first round's captures: the bytes the
collector wrote during the 60 s load window ([results/captures.json](results/captures.json)).

| Collector | Output tokens/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | Trace written |
|---|---|---|---|---|---|---|
| none | 4,267 (4,254 – 4,274) | 108 ms | 233 ms | 6.69 ms | 7.39 ms | |
| light: graph envelopes + submission APIs | 4,226 (−1.0%) | 99 ms | 222 ms | 6.80 ms (+1.6%) | 7.43 ms (+0.5%) | 6 MB/s |

Trace mode of that build wrote 35 MB/s and cost 4.4% of throughput here
(the trace row of the same files). In closed loop, median TTFT is lower
with light mode on (108 to 99 ms),
probably because with a fixed number of clients a slower engine also
receives new prompts at a lower rate. Closed-loop TTFT therefore does not
measure the collector's cost; the open-loop numbers above do.

- **Light mode** records each CUDA graph launch, but kernels launched
  outside a graph are not timed one by one.
- **Trace mode** gives per-kernel GPU time, which `blame` needs for step
  busy time.
- All modes keep a bounded ring on disk, and `reqscope dump --pid` freezes
  the last N seconds.

## Trace mode's own cost

Trace mode slows the engine it measures and adds slow steps that blame
calls host delay. Where that time goes was measured from inside: every
capture now has the collector's own timings per site (`CollectorStats`),
the benchmark samples every EngineCore thread from `/proc`, and
`bench/trace_cost.py` sets the CUDA API durations of light and trace runs
side by side (docs/design.md, "What trace mode costs the engine";
[results/trace-cost.json](results/trace-cost.json)). In one 40 s run at
16 requests/s:
- the collector's code on the engine thread (buffer requests, step push
  and pop, plugin records) is under 0.1% of a core;
- CUPTI's work inside launches is not: the decode graph's launch takes
  277 µs against 72 µs in light mode;
- the writer took 0.33 cores, and while it built `Json` objects, runs of
  launches on the engine thread took about 250 µs each (29,243 launches
  over 100 µs, 7.2 s in 42 s); writing text instead, it takes 0.10 cores
  and the runs are gone;
- moving to CUPTI's next 3.2 MB device buffer stalled a launch for 1–2 ms
  every few hundred steps; the collector now asks for 32 MB buffers.

Before and after, in the same session: five rounds of the open-loop
experiment at 8 and 16 requests/s, with no collector, trace mode as of
commit 3ee45a6 and the current trace mode in an order rotated each round
([results/openloop-before.json](results/openloop-before.json),
[results/openloop-after.json](results/openloop-after.json); host load
2.0–3.6), and the multi-cause runs above.

| Rate | Collector | TTFT p50, ms | TPOT p50, ms | TPOT p99, ms | TPOT p50 / p99 vs none |
|---|---|---|---|---|---|
| 8/s | none | 23.2 (22.9–23.3) | 5.12 (5.11–5.12) | 5.44 (5.40–5.44) | |
| 8/s | trace, before | 22.6 (22.6–22.9) | 5.21 (5.18–5.21) | 5.65 (5.50–5.69) | +1.8% / +3.9% |
| 8/s | trace, now | 22.7 (22.6–22.8) | 5.20 (5.19–5.21) | 5.64 (5.57–5.67) | +1.6% / +3.7% |
| 16/s | none | 23.7 (23.7–24.1) | 5.29 (5.27–5.29) | 6.08 (6.07–6.10) | |
| 16/s | trace, before | 26.4 (25.9–26.7) | 6.22 (6.15–6.29) | 8.26 (7.77–8.40) | +17.6% / +35.9% |
| 16/s | trace, now | 25.2 (24.8–25.9) | 6.19 (6.11–6.24) | 7.77 (7.34–7.93) | +17.0% / +27.8% |

| Multi-cause runs, 3 seeds | Slow steps outside the windows | Called host delay | Host-delay precision, all steps |
|---|---|---|---|
| trace, before | 503 | 490 | 0.03 |
| trace, now | 231 | 228 | 0.04 |
| plugin only | 6 | 0 | — |

What improved and what did not:
- **Spurious slow steps halved** (503 to 231).
- **The latency cost did not measurably change.** At 16 requests/s the
  median TPOT p99 went from 8.26 to 7.77 ms (+36% to +28% over no
  collector) and TTFT p50 from 26.4 to 25.2 ms, but in both the ranges of
  the five rounds overlap (7.77–8.40 against 7.34–7.93 ms; 25.9–26.7
  against 24.8–25.9 ms), so these changes are within noise; so is TPOT
  p50 (+17.6% to +17.0%). Host-delay precision over all steps is still
  0.04. In the open-loop runs, launches
  over 100 µs fell from 104,247 to 11,277 in round 1, but their total time
  only from 25.1 s to 23.1 s: the long waits left are fewer and longer
  (median about 2 ms), all inside CUDA launch calls
  ([results/openloop-trace-cost.json](results/openloop-trace-cost.json)).
  They come with per-kernel tracing (light mode had 11 launches over
  100 µs in 42 s), and the collector has no handle on them short of not
  tracing every kernel.
- **Tried and dropped:** pinning the collector's and CUPTI's threads to
  the efficiency cores and a 1 ms CUPTI flush period both made trace mode
  slower (docs/design-tradeoffs.md).

## Analysis speed

`reqscope blame` reads the 23 s, 0.9 GB capture of the interference
experiment (2.06M records) and reports on it in 10.5 s with 4.1 GB of
memory: the median of three runs with the current build, on the shared
host at a load average of 8.8
([results/analysis-speed.json](results/analysis-speed.json), from
`bench/analysis_speed.py`). The one earlier timing in `results/`,
`blame_seconds` (28.9 s) in `results/interference.json`, was taken by
`bench/score.py` with the analyzer from before the changes below; other
timings of that analyzer were not recorded.

| Cause | Fix |
|---|---|
| A bitwise CRC32 over every frame | A slicing-by-8 table; the collector writes with it too |
| One read and one allocation per frame, and a sort whose comparator copied two producer strings on each comparison | One read per segment; ordering keys taken while parsing |
| JSON objects as `std::map`, and values sized by doubling | Members in one sorted vector, values sized exactly |
| `blame` built the dependency graph that only `analyze` reports | Built only when reported |
| `analyze` grew with the square of the trace: every elementary interval of a request rescanned all of its GPU activities, every token all intervals, every step all activities and syncs | One sweep over the intervals, a merge for the tokens, per-producer indexes |

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
# alone, and trace mode of commit 3ee45a6 (built in build-before/); each run
# holds the GPU lock and starts its own GPU burner
for s in 1 2 3; do
  for m in trace plugin; do
    python bench/interference.py multicause --mode $m --seed $s --out results/raw/multicause-after-$m-s$s
  done
  python bench/interference.py multicause --mode trace --build build-before --seed $s \
    --out results/raw/multicause-before-trace-s$s
done
python bench/score_causes.py results/raw/multicause-after-*-s?
python bench/score_causes.py --rule trivial results/raw/multicause-after-*-s?
python bench/score_causes.py results/raw/multicause-before-trace-s?

# The same seeds under Nsight Systems: plugin only, NVTX step ranges,
# vLLM's per-step log
for s in 1 2 3; do
  r=results/raw/multicause-nsys-s$s
  python bench/interference.py multicause --mode plugin --nsys --vllm-iteration-log --seed $s --out $r
  nsys export --type sqlite --output $r/nsys.sqlite $r/nsys.nsys-rep
  python -m checker.nsys_projection $r/nsys.sqlite --steps --max-bytes $((8 << 30)) \
    --max-work 200000000 --output $r/nsys-steps.json
  python bench/vllm_iterations.py $r
done
for rule in blame trivial prefill_only vllm nsys; do
  python bench/score_causes.py --rule $rule results/raw/multicause-nsys-s?
done

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

# Trace mode before and after its writer and device-buffer changes: a
# build of commit 3ee45a6 in build-before/, rounds rotated as above
for r in 1 2 3 4 5; do for m in off before trace; do
  b=build; mode=$m; [ $m = before ] && { b=build-before; mode=trace; }
  python bench/interference.py openloop --mode $mode --build $b --round $r --rates 8,16 --seconds 40 \
    --out results/raw/openloop-$([ $m = before ] && echo before-trace || echo after-$m)-r$r
done; done
python bench/score.py openloop results/raw/openloop-after-off-r* results/raw/openloop-before-trace-r*
python bench/score.py openloop results/raw/openloop-after-off-r* results/raw/openloop-after-trace-r*
python bench/trace_cost.py results/raw/openloop-before-trace-r1 results/raw/openloop-after-trace-r1

# Where trace mode's time goes: one 40 s run at 16 requests/s per mode, and
# decode-like steps without vLLM
for m in off plugin light trace; do
  python bench/interference.py openloop --mode $m --rates 16 --seconds 40 --out results/raw/tracecost-$m
done
python bench/trace_cost.py results/raw/tracecost-*
python bench/launch_cost.py --mode trace --out results/raw/launch-cost/decode-trace-1.json

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
(default 0.6), `--lock FILE`, which holds that file's flock from server
start to shutdown so runs on a shared GPU do not overlap, and
`--collector-env REQSCOPE_KEY=VALUE` for extra collector settings. Each run
records the host's load average and samples of every EngineCore thread
(CPU, run-queue wait, last CPU) in its `client.json`. `bench/score.py`
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
- **One CUPTI subscriber**, so light and trace mode cannot run together
  with Nsight Systems or `torch.profiler`. The plugin alone can; the
  Nsight Systems runs above used it.
- **`analyze` and `explain` emit every elementary interval.** Each
  request's decomposition holds one interval per gap between GPU activity
  boundaries, so their output and memory grow with the GPU activity around
  each request and soon exceed the capture's size. On a full long capture,
  use `blame`, or analyze a shorter window.
- **JSON records.** The analyzer parses every record from JSON text. A
  binary encoding of the four record types that make up 99% of a trace
  ([results/captures.json](results/captures.json)) is the next step.
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
  around the step's GPU work; `blame` does not tell them apart.
- **Trace mode perturbs what it measures.** It added slow decode steps of
  its own in the interference runs (30, 71 and 129 against 27 and 33 with
  CUPTI off, earlier build) and 231 against 6 outside the injection windows
  of the multi-cause runs, which blame calls host delay; it costs 17% of
  TPOT p50 at 16 requests/s. Most of what is left happens inside CUDA
  launch calls during per-kernel tracing, which the collector cannot
  change ([Trace mode's own cost](#trace-modes-own-cost)). The
  `collector_active` flag does not pick these steps out. Its per-kernel
  evidence is for short captures at moderate load.
- **A shared host.** The runs shared the machine with other projects' jobs.
  Every run held the GPU lock those jobs share, but host load varied
  (1-minute load average 1.1–4.7 in the first open-loop round, 2.0–3.6 in
  the before/after round, 6.5–10.9 in the first multi-cause round and
  2.3–3.9 in the second, about 9–10 in the first CUPTI-off pair), and host
  load changes host-side timings.
- **No CPU scheduling evidence.** Host-side CPU stalls are reported as
  `between_steps`, without run-queue evidence.
