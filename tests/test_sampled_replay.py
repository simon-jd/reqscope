"""Sampled mode on a real CUDA graph: a traced step keeps its own kernel end times.

Runs fixtures/dag/sampled_replay.cu (one graph of eight 100 us spin kernels,
launched once per step while the previous step's still runs) under the
collector in sampled mode. Needs a GPU, the collector built with CUPTI, and a
build of the CUDA fixtures (cmake -DREQSCOPE_BUILD_CUDA_FIXTURES=ON) named by
REQSCOPE_CUDA_FIXTURES; skipped otherwise.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from checker.raw import read_run

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("REQSCOPE_LIBRARY", ROOT / "build/libreqscope.so"))
FIXTURES = os.environ.get("REQSCOPE_CUDA_FIXTURES")
FIXTURE = Path(FIXTURES) / "dag_sampled_replay" if FIXTURES else None
STEPS = 48


@unittest.skipUnless(FIXTURE and FIXTURE.is_file() and LIBRARY.is_file(),
                     "set REQSCOPE_CUDA_FIXTURES to a build of the CUDA fixtures; needs a GPU")
class SampledReplayTests(unittest.TestCase):
    def capture(self, every):
        directory = tempfile.TemporaryDirectory(prefix="rqs-sampled-replay-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        # No forced CUPTI flush during the run: one returns the kernels still
        # running without timestamps (docs/design.md, "Sampled tracing").
        env = {**os.environ, "CUDA_INJECTION64_PATH": str(LIBRARY), "REQSCOPE_LIBRARY": str(LIBRARY),
               "REQSCOPE_MODE": "sampled", "REQSCOPE_TRACE_EVERY": str(every), "REQSCOPE_CUPTI_FLUSH_MS": "60000",
               "REQSCOPE_OUTPUT": str(root / "capture"), "REQSCOPE_SOCKET": str(root / "control.sock")}
        result = subprocess.run([str(FIXTURE), str(STEPS)], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        truth = json.loads(result.stdout.splitlines()[-1])
        records = read_run(root / "capture").records
        traced = {r["step_id"] for r in records if r["type"] == "TracedStep"}
        step_of = {r["corr"]: r["step_id"] for r in records if r["type"] == "ExtCorr"}
        kernels = defaultdict(list)
        for r in records:
            if r["type"] == "GpuActivity" and r["kind"] == "kernel":
                kernels[step_of.get(r["corr"])].append((r.get("start") or 0, r.get("end") or 0))
        return truth, traced, kernels

    def test_traced_steps_keep_their_kernel_end_times(self):
        for every in (4, 16):
            with self.subTest(every=every):
                truth, traced, kernels = self.capture(every)
                self.assertEqual(sorted(traced), list(range(1, STEPS + 1, every)))
                for step in sorted(traced):
                    spans = kernels[step]
                    self.assertEqual(len(spans), truth["nodes_per_replay"], f"step {step}")
                    for start, end in spans:
                        self.assertTrue(start and end, f"step {step} has an untimed kernel")
                        # Each kernel spins 100 us; the same node of the next
                        # launch ends at least one launch (800 us) later.
                        self.assertLess(end - start, 3 * truth["spin_ns"], f"step {step}")
                # Kernel tracing stays on for the step after each traced one
                # only; that guard step's records are not evidence.
                self.assertEqual(set(kernels) - traced, {step + 1 for step in traced})


if __name__ == "__main__":
    unittest.main()
