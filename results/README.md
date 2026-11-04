# Results

All runs used `bench/interference.py` on one RTX 4090 with vLLM 0.31.0 and
Qwen2.5-1.5B-Instruct. The raw run directories (capture, `client.json`,
`server.log`) stay out of the repository under `results/raw/`; the files
here are derived from them.

| File | Runs | Produced by |
|---|---|---|
| `interference.json` | `interference-s5` | `bench/score.py interference` |
| `interference-blame.json` | `interference-s5` | `reqscope blame` on its capture (`score.py` saves it as `blame.json`) |
| `interference-steps.json` | `interference-s5` | `bench/score_steps.py`, from `blame.json` and the client's ground truth; it includes the six long prompts' windows |
| `overhead.json` | `overhead-{off,light,trace}`, then `-r2` and `-r3` | `bench/score.py overhead`; one row per run, rounds 1 to 3 in order |
| `../docs/slow-steps.svg` | `interference-s5` | `bench/plot_steps.py`, from `interference-blame.json` and `interference-steps.json` |
| `environment.json` | the host | `bench/environment.py`: Python, vLLM, torch, CUDA toolkit, CUPTI, driver, GPU, kernel, CPU |
| `ablation.json` | `interference-s5`, then `ablation-trace` and `ablation-plugin` (pair 1) and `ablation-trace-2` and `ablation-plugin-2` (pair 2) | `bench/compare_runs.py --load`, after `bench/score.py interference` on each |
| `openloop.json` | `openloop-{off,plugin,light,trace}-r1` to `-r5` | `bench/score.py openloop --load` |
| `multicause.json` | `multicause-trace-s1` to `-s3` and `multicause-plugin-s1` to `-s3` | `bench/score_causes.py`, which runs `reqscope blame` on each capture and compares it with the injected events in `client.json` |
| `openloop-before.json`, `openloop-after.json` | `openloop-after-off-r1` to `-r5` with `openloop-before-trace-r1` to `-r5`, and with `openloop-after-trace-r1` to `-r5` | `bench/score.py openloop`, once per pair of sets: both files share the same no-collector runs |
| `multicause-before.json`, `multicause-after.json` | `multicause-before-trace-s1` to `-s3`; `multicause-after-trace-s1` to `-s3` and `multicause-after-plugin-s1` to `-s3` | `bench/score_causes.py`, with the current `reqscope blame` (it adds the `collector_active` flags; the before runs have no writer activity records, so none are flagged) |
| `multicause-baselines.json` | `multicause-after-trace-s1` to `-s3` and `multicause-after-plugin-s1` to `-s3` | `bench/score_causes.py --rule trivial` and `--rule prefill_only` on the same runs as `multicause-after.json` |
| `multicause-nsys.json` | `multicause-nsys-s1` to `-s3` (plugin only under Nsight Systems, with vLLM's per-step log) | `bench/score_causes.py` with each `--rule`, after `checker/nsys_projection.py --steps` and `bench/vllm_iterations.py` on each run; `sources` has each run's matching coverage |
| `openloop-trace-cost.json` | `openloop-before-trace-r1`, `openloop-after-trace-r1` | `bench/trace_cost.py` (both rates of round 1) |
| `trace-cost.json` | `tracecost-{off,plugin,light,trace}`, then `tracecost-trace-text-dev32`, `-ecores` and `-flush1` | `bench/trace_cost.py`: one 40 s open-loop run at 16 requests/s each; the collector's CollectorStats, EngineCore thread samples, CUDA API durations, slow launches and graph-launch stalls |
| `launch-cost.json` | `results/raw/launch-cost/*.json` | `bench/launch_cost.py`, one file per run, gathered with a `config` and `collector` label per run |
| `analysis-speed.json` | `interference-s5` | `bench/analysis_speed.py`: three runs of `reqscope blame` on its capture, wall time and peak memory |
| `captures.json` | `interference-s5`, round 1 of `overhead-light` and `overhead-trace`, and `interference-s1` to `-s4` | `bench/capture_stats.py` |
| `multicause-sampled.json` | `multicause-sampled-{trace,s4,s16,trig,plugin}-s1` to `-s3` | `bench/score_causes.py`, which ran `reqscope blame` on each capture (the trace reader's budget at 8 GiB) |
| `openloop-sampled.json` | `openloop-sampled-{off,light,s16,s4,trig,trace}-r1` to `-r5` | `bench/score.py openloop` |
| `sampled-captures.json` | `multicause-sampled-{s4,s16,trig}-s1` and `openloop-sampled-{light,s16,s4,trig}-r1` | `bench/sampled_captures.py` |
| `multicause-sampled-guard.json` | `multicause-guard-{s4,s16,trig}-s1` to `-s3` | `bench/score_causes.py`, as for `multicause-sampled.json` |
| `openloop-sampled-guard.json` | `openloop-guard-{off,s16,s4,trig}-r1` to `-r5` | `bench/score.py openloop` |
| `sampled-captures-guard.json` | `multicause-guard-{s4,s16,trig}-s1` and `openloop-guard-{s16,s4,trig}-r1` | `bench/sampled_captures.py` of commit a21d637 (guard steps in their own group) |

Notes:

- `blame_seconds` in `interference.json` (28.9 s) is the wall time of
  `reqscope blame` as built when the run was scored, before the reader and
  decomposition changes in the README's analysis-speed section. The
  current build's time is in `analysis-speed.json`.
- `loss_records: 1` with `loss_count: 0` is not a dropped record. The
  capture has no `Finalized` session record, because vLLM SIGKILLs
  EngineCore right after the final flush, and the analyzer records the open
  end as a loss of unknown size. `captures.json` shows the capture's
  sequence numbers are contiguous and that it has no `Loss` records.
- `interference-s1` to `-s4` are earlier captures of the same experiment,
  made while the collector still had problems listed in the README's
  "Getting the measurement right" table. Their headers record the API
  policy (`all-driver-runtime` for s1 to s3, the submission/copy/sync
  whitelist for s4), and s1's segments are 1 MiB where the later runs'
  reach 16 MiB. Their scores are not committed.
- The CUPTI-off pairs and the open-loop runs used the collector and bench
  scripts of commit 2c85655, with `--gpu-memory-utilization 0.6`; each run
  held the host's GPU lock. All but `interference-s5` (which keeps its
  original `blame.json`) were scored with the current analyzer. Their host load comes from a CSV in
  `bench/sample_load.py`'s format, sampled every 5 s through the session, kept
  with the raw runs as `results/raw/loadavg-2026-10-31.csv` (`client.json`
  records load itself in later runs).
  The sampler started after pair 1, whose load (8.8 over 1 minute, 10.4
  over 5) was read with `uptime` during the pair.
- The plugin-only captures predate the fix that leaves `clock_error_ns`
  absent in plugin mode, so their headers still claim 0. They have no GPU
  records, and `ablation.json` reports no clock values.
- The multi-cause runs used commit 60fca57, `--gpu-memory-utilization 0.6`
  and the GPU lock for each run (the GPU burner ran inside that hold). The
  cause rules were fixed on `multicause-pilot-trace` (seed 100), which is
  not scored. Each run's `client.json` has its host load samples.
- The before/after runs (`openloop-before-*`, `openloop-after-*`,
  `multicause-before-*`, `multicause-after-*`) ran in one session, each
  holding the GPU lock, with the bench scripts of commit 3218aeb and
  `--gpu-memory-utilization 0.6`. "after" is the collector of 3218aeb
  (the same as 983884e); "before" is a build of 3ee45a6. Open-loop rounds
  rotated the order of no collector, before and after. Each run directory
  has `nvidia-smi.csv` with the GPU's state before and after it: at the
  start of the 24 runs the GPU was idle in P0 or P2 at a 2.5–2.8 GHz SM
  clock (clocks cannot be locked on this host).
- The `tracecost-*` runs measured where trace mode's time goes. The first
  four (off, plugin, light, trace) used the collector of commit 8b4670e,
  built with switches for the experiments below left unset, and the bench
  scripts of afb0fed. The other three used `--collector-env` (3218aeb)
  and builds with the text writer and 32 MB device buffers:
  `tracecost-trace-text-dev32` with an option to pin the collector's
  threads set to none, which is what 983884e does; `-ecores` with the
  writer, probe and CUPTI's worker pinned to the efficiency cores; and
  `-flush1` with a 1 ms CUPTI flush period. The last two were not kept.
  All held the GPU lock, with `--gpu-memory-utilization 0.6`.
- In `launch-cost.json`, `fast_pace` runs used an earlier revision of
  `bench/launch_cost.py` whose kernels touched 256 floats, as `--size 256
  --steps 5000` does now; at that pace the Json writer could not drain
  CUPTI's buffers, so those runs measure a saturated writer. `decode_pace`
  runs used the committed script. The `env` of each run lists the
  experiment switches set; the `collector` label says which build ran.
- Overhead rounds 2 and 3 kept only client timings, so the write rates in
  `captures.json` come from round 1.
- The sampled-mode runs (`multicause-sampled-*`, `openloop-sampled-*`) ran
  in one session with the collector, analyzer and bench scripts of commit
  a14b088 and `--gpu-memory-utilization 0.6`. Each run held the GPU lock
  and sampled `nvidia-smi` every 5 s inside it (`nvidia-smi.csv` in the run
  directory). In the run names, `s4` is `--mode sampled` with
  `REQSCOPE_TRACE_EVERY=4`, `s16` the same with 16, and `trig` is
  `REQSCOPE_TRACE_TRIGGER_PERCENT=150` with `REQSCOPE_TRACE_WINDOW=32` and
  no periodic steps; the other arms are the modes of the same name, and
  `off` is no collector. The multi-cause seeds rotated the order of their
  five arms, and the five open-loop rounds (40 s at 8 and then 16
  requests/s per run) the order of their six. Host load (1-minute average): 2.05–3.32
  during the multi-cause runs, 1.85–6.57 during the open-loop rate windows.
  Their sampled captures have the late kernel end times that commit 6fd8ae3
  fixed (the README's "Sampled trace mode"); the three files keep them as
  measured.
- The guard-step runs (`multicause-guard-*`, `openloop-guard-*`) ran in a
  second session, after the collector change of commit 6fd8ae3 (built from
  1600976, which also has the analyzer change), with the same settings,
  model, lock and `nvidia-smi` sampling as the first; `bench.stdout` in
  each run directory is the benchmark's own output. Only the sampled arms
  were rerun, with no collector (`off`) as the open-loop reference; trace
  mode, light and the plugin alone were not. The multi-cause seeds rotated
  the order of their three arms, the open-loop rounds that of their four.
  Host load: 3.70–7.44 during the multi-cause runs and 2.38–13.14 during
  the open-loop rate windows (round 5 at 6.2–13.1), higher than in the
  first session. While the GPU was busy, SM clocks were 2520–2805 MHz and
  21 of 395 samples showed the software power cap.
