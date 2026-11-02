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

`nsys_projection.py` maps an Nsight Systems run of the plugin onto engine
steps, without reqscope's collector or analyzer: with `REQSCOPE_NVTX=1` the
plugin marks each step with an NVTX range in the `reqscope` domain whose
text holds the step id, the phase and the step's members. The multi-cause benchmark uses it
for its Nsight Systems runs (`bench/interference.py multicause --nsys`).

```sh
nsys export --type sqlite --output run.sqlite run.nsys-rep
python3 -m checker.nsys_projection run.sqlite --steps --output steps.json   # GPU busy time per step
python3 -m checker.nsys_projection run.sqlite --output projection.json       # every row and how it was matched
python3 -m unittest discover -s tests -p test_nsys_projection.py -v
```

How records are matched. A CUDA runtime call belongs to a step when it lies
entirely inside one reqscope NVTX range on the same thread. A kernel, copy,
memset or whole-graph row belongs to that call when it has the same correlation id and comes
from the same process. Nsight Systems documents this link between its
tables, but SQLite does not enforce it, so the script assumes it. Driver API
calls are not used for matching: a GPU record whose correlation id matches
only a driver call stays unmatched. GPU timestamps never decide which step a
record belongs to, and request ids are copied unchanged.

What it reports. Every row keeps its SQLite row id, so each match can be
checked against the export. A range is invalid if its payload is malformed,
if it is not closed on the thread that opened it, or if a step's two phases
disagree on its members. A call inside two nested ranges, and a GPU record
that matches no call or more than one, is counted as ambiguous or unknown
rather than dropped. `--steps` gives each step's GPU busy time:
the union of its matched records' intervals, so overlapping work counts
once. Without `--steps`, durations are plain sums per request and count
work shared by several requests once for each of them.

Limits. The export is opened read-only, and Python 3.11 or newer is needed.
The script stops with an error, instead of returning a partial result, if
the input exceeds any of these: 2,000,000 rows read (`--max-rows`), 2 MiB in
one SQLite value or row (`--max-value-bytes`), 256 MiB of input or of member
lists copied onto matched rows (`--max-bytes`), or 10,000,000 comparisons while
matching (`--max-work`). These limit the work done, not memory use exactly.
The Nsight Systems runs of the multi-cause benchmark need
`--max-bytes 8589934592 --max-work 200000000`. The table
layout is that of the Nsight Systems version used here (2025.6); other
versions may differ. `--output` must name a new file.
