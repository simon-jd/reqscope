# Results

All runs used `bench/interference.py` on one RTX 4090 with vLLM 0.31.0 and
Qwen2.5-1.5B-Instruct. The raw run directories (capture, `client.json`,
`server.log`) stay out of the repository under `results/raw/`; the files
here are derived from them.

## Which file backs which claim

"Run date" is the day the runs were made (host time); "Committed" is the
commit that added the file, then any that changed it. Section names are
the README's.

| File | README claim it backs | Run date, collector build | Committed |
|---|---|---|---|
| `multicause-after.json` | Results table: blame's column of rows 1–4 and both columns of row 5; "Without GPU data"; the multi-cause benchmark's tables, confusion matrices, verdict evidence medians and `collector_active` counts; "trace, now" in Trace mode's own cost | 2026-11-01, 3218aeb | 3f236f2 (2026-11-01), af9bf54 (2026-11-02) |
| `multicause-baselines.json` | Results table: the trivial rule's column of rows 1–4; "Against a trivial rule" | the runs of `multicause-after.json`, rescored | 6e1bf24 (2026-11-02) |
| `openloop-after.json` | Results table row 6; Limitations: trace mode's +28% on TPOT p99 and +17% on p50 at 16 requests/s; "trace, now" and no-collector rows in Trace mode's own cost | 2026-11-01, 3218aeb | 3f236f2 (2026-11-01) |
| `multicause-before.json` | Trace mode's own cost: "trace, before", 503 slow steps outside the windows, precision 0.03 (15 of 511) | 2026-11-01, 3ee45a6 | 3f236f2 (2026-11-01) |
| `openloop-before.json` | Trace mode's own cost: the "trace, before" rows | 2026-11-01, 3ee45a6 | 3f236f2 (2026-11-01) |
| `openloop-trace-cost.json` | Trace mode's own cost: launches over 100 µs in round 1, 104,247 before and 11,277 after | 2026-11-01, 3ee45a6 and 3218aeb | 3f236f2 (2026-11-01) |
| `multicause-nsys.json` | Limitations: "Nsight Systems does as well"; Results from other sessions; "Against Nsight Systems and vLLM's own step log" | 2026-11-02, plugin only (build not recorded) | 7972521 (2026-11-02) |
| `multicause.json` | The multi-cause benchmark: the earlier round on a busier host, cited but not tabulated | 2026-10-31, 60fca57 | 3ee45a6 (2026-11-01) |
| `multicause-sampled.json` | Sampled trace mode, first session: the multi-cause rows of "What it changed", left of the arrows; trace mode's 220 of 220 burst steps and 9 right of 19 host-delay verdicts; its 13 slow steps outside the windows | 2026-11-03, a14b088 | 4d62d64 (2026-11-03) |
| `openloop-sampled.json` | Sampled trace mode, first session: the light and trace arms (+15.5% / +37.1% at 16 requests/s) and the TPOT row of "What it changed", left of the arrows | 2026-11-03, a14b088 | 4d62d64 (2026-11-03) |
| `sampled-captures.json` | Sampled trace mode, first session: kernel spans and steps traced before the guard step, light mode's 49 µs graph launch, copies without timestamps (2.3–3.4%), and why the trigger ignores intervals after a prefill (752 of 799, 792 of 811) | 2026-11-03, a14b088 | 4d62d64 (2026-11-03) |
| `multicause-sampled-guard.json` | Limitations: "Sampled mode misses most host delays"; the multi-cause table of the second session and the multi-cause rows of "What it changed", right of the arrows | 2026-11-04, built from 1600976 | eba36e3 (2026-11-04) |
| `openloop-sampled-guard.json` | Results from other sessions: sampled mode's TPOT cost; the open-loop table of the second session and the TPOT row of "What it changed", right of the arrows | 2026-11-04, built from 1600976 | eba36e3 (2026-11-04) |
| `sampled-captures-guard.json` | Sampled trace mode, second session: kernel spans after the guard step, launch times on traced and guard steps (213–254 µs), late end times where the guard step runs other graphs (10 of 1,498; 119 of 3,600), copies without timestamps (2.4–3.5%) | 2026-11-04, built from 1600976 | eba36e3, 99c2990 (both 2026-11-04) |
| `openloop.json` | Overhead, open loop: the plugin alone and light mode within 1% of no collector; trace mode of that build fell behind at 24 requests/s | 2026-10-31, 2c85655 | 78dbd41 (2026-10-31) |
| `ablation.json` | With CUPTI off: the paired-run table; Results from other sessions: "Single-cause interference" | 2026-10-28 (earlier build) and 2026-10-31 (2c85655, the pairs) | 0d0ad6c (2026-10-31) |
| `interference.json` | The interference experiment's table: top six culprits, excess shares, clock bound, records linked and lost | 2026-10-28, a build from before the first commit | 43734aa (2026-10-30) |
| `interference-blame.json` | blame's own report on that run, plotted in `docs/slow-steps.svg` | 2026-10-28, as above | 43734aa (2026-10-30) |
| `interference-steps.json` | The interference experiment: per-step precision 1.00 (18 of 18) and recall 0.82 (18 of 22) | 2026-10-28, as above | 43734aa (2026-10-30), 0cbf3dd (2026-10-31) |
| `captures.json` | The interference experiment: 2.06M records, 0 lost; Overhead, closed loop: trace written per mode; Limitations in full: four record types make up 99% of a trace | 2026-10-28, as above | 43734aa (2026-10-30) |
| `overhead.json` | Overhead, closed loop: throughput, TTFT and TPOT per mode | 2026-10-28, as above | 43734aa (2026-10-30) |
| `trace-cost.json` | Trace mode's own cost: the collector's share of the engine thread, 277 against 72 µs decode-graph launches, the writer's 0.33 and 0.10 cores, device-buffer stalls | 2026-11-01, 8b4670e and the text-writer builds | 1e8fbcf (2026-11-01) |
| `launch-cost.json` | docs/design.md: device-buffer stalls in decode-like steps without vLLM, 21 with 3.2 MB buffers and 1 with 32 MB in 5,000 steps | 2026-11-01, per run (`collector` label) | 1e8fbcf (2026-11-01) |
| `analysis-speed.json` | Analysis speed: 10.5 s and 4.1 GB for blame on the 0.9 GB capture | 2026-11-02, analyzer of fabf15e | 5f26db6 (2026-11-02) |
| `environment.json` | Reproduce: the versions behind the committed results | the host | a5360c7 (2026-10-31) |
| `../docs/slow-steps.svg` | The interference experiment's figure | from `interference-s5` | 30405a4 (2026-10-31) |

## Host load per session

| Run date | Runs | 1-minute load average |
|---|---|---|
| 2026-10-28 | `interference-s5` | not recorded |
| 2026-10-31 | CUPTI-off pair 1 / pair 2 | about 9–10 / 1.7–1.9 |
| 2026-10-31 | `openloop-*` (2c85655) | 1.1–4.7 |
| 2026-10-31 | `multicause-*` (60fca57) | 6.5–10.9 |
| 2026-11-01 | `openloop-before-*`, `openloop-after-*` | 2.0–3.6 |
| 2026-11-01 | `multicause-before-*`, `multicause-after-*` | 2.3–3.9 |
| 2026-11-02 | `multicause-nsys-*` | 5.8–8.9 |
| 2026-11-02 | `analysis-speed.json` | 8.8 |
| 2026-11-03 | `multicause-sampled-*` / `openloop-sampled-*` | 2.05–3.32 / 1.85–6.57 |
| 2026-11-04 | `multicause-guard-*` / `openloop-guard-*` | 3.70–7.44 / 2.38–13.14 |

## How each file was produced

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
| `sampled-captures-guard.json` | `multicause-guard-{s4,s16,trig}-s1` and `openloop-guard-{s16,s4,trig}-r1` | `bench/sampled_captures.py` of commit 99c2990 (guard steps in their own group, and long spans counted) |

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
