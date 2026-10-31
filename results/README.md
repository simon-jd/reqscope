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
| `captures.json` | `interference-s5`, round 1 of `overhead-light` and `overhead-trace`, and `interference-s1` to `-s4` | `bench/capture_stats.py` |

Notes:

- `blame_seconds` in `interference.json` (28.9 s) is the wall time of
  `reqscope blame` as built when the run was scored, before the reader and
  decomposition changes in the README's analysis-speed section. The
  current build's time is not recorded here.
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
  original `blame.json`) were scored with the current analyzer. Their host load comes from a
  `bench/sample_load.py` CSV sampled every 5 s through the session (not
  committed; `client.json` records load itself from the next commit on).
  The sampler started after pair 1, whose load (8.8 over 1 minute, 10.4
  over 5) was read with `uptime` during the pair.
- The plugin-only captures predate the fix that leaves `clock_error_ns`
  absent in plugin mode, so their headers still claim 0. They have no GPU
  records, and `ablation.json` reports no clock values.
- Overhead rounds 2 and 3 kept only client timings, so the write rates in
  `captures.json` come from round 1.
