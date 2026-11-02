# Independent recording checks

The checker uses only Python's standard library and imports no C++ analyzer,
collector, or adapter implementation. It shares the documented wire format.

```sh
python3 -m fixtures.fake_engine.generate /tmp/replay
build/reqscope analyze /tmp/replay > /tmp/replay-analysis.json
python3 -m checker.oracle /tmp/replay --analysis /tmp/replay-analysis.json --truth /tmp/replay/truth.json
python3 -m unittest discover -s tests -v
```

`raw.py` checks bounds before reading frames, validates CRC and UTF-8 JSON,
rejects duplicate keys and non-finite numbers, and detects producer identity,
sequence, clock, and incomplete capture boundary issues. Valid prefixes remain
available after corruption; malformed records cannot enter the membership oracle.
`oracle.py` independently joins producer-scoped step/external/API evidence,
compares exact replay membership, and checks non-overlapping request wall-clock
intervals, E2E conservation, and token-weighted GPU activity estimates. Fixture
truth contains literal answers written separately from those computations.
The test suite also deliberately mutates analyzer output to prove the oracle
rejects wrong membership, invented edge provenance, and broken conservation.

The fake engine records are synthetic CPU files. They exercise graph replay
identity and padding semantics but do not execute kernels or validate CUPTI,
vLLM, tracing overhead, or any GPU hardware behavior.

The CUPTI recordings of the CUDA DAG fixtures (`fixtures/dag/`) are checked
in under `fixtures/recordings/`. `tests/test_real_dag_edges.py` reads them
with `raw.py`, without importing analyzer code, and compares the analyzer's
operation-DAG edges with handwritten expected edges:

```sh
python3 -m tests.test_real_dag_edges --output dag-edges.json
```

Graph replay identity alone does not verify graph DAG edges; replays whose
edges the recordings cannot prove are listed as unresolved.

`nsys_projection.py` independently projects reqscope NVTX membership onto an
Nsight Systems SQLite export, for comparing reqscope with an Nsight Systems
run of the plugin (`REQSCOPE_NVTX=1` marks each step with an NVTX range in
the `reqscope` domain). No such comparison is included in `results/`.

```sh
python3 -m checker.nsys_projection recording.sqlite --output new-projection.json
python3 -m unittest discover -s tests -p test_nsys_projection.py -v
```

The input is opened read-only. This checker requires Python 3.11 or newer to
enforce SQLite's value/row length limit before allocation. A CUDA runtime API
must be completely contained
in one reqscope NVTX range on the same `globalTid`; GPU records then require a
unique runtime `correlationId` in the same masked `globalPid`. This uses the
Nsight export schema's runtime association contract/comment, not an enforced
SQLite foreign key. A numeric collision in the separate DRIVER table is not a
runtime identity candidate, and a driver-only match remains unknown. GPU times
never select identities, and request IDs are preserved exactly.

The report retains source row IDs, NVTX payload fields and member JSON paths.
Requests are scoped by process. Reused Step phases or conflicting members
invalidate their ranges; ambiguous and unknown GPU records stay in the
coverage denominator. Kernel, copy and memset counts and duration distributions
are reported separately. Durations sum observed activity, including overlap
and work shared by multiple requests; they do not measure exclusive request
GPU time. Bounds cover combined input rows (`--max-rows`, default 2,000,000),
SQLite value/row bytes before Python allocation (`--max-value-bytes`, 2 MiB),
cumulative input and separately estimated membership expansion (`--max-bytes`,
256 MiB each), and containment/correlation/member-projection work
(`--max-work`, 10,000,000). Overflow fails explicitly without a completeness
result. These are explicit work/data limits, not an exact Python RSS guarantee.
Export metadata is retained when available; the fixed runtime association
contract is assumed, without claiming compatibility with arbitrary future
Nsight schemas. `--output` must name a new file.
