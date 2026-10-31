"""Exercise the real shared library in isolated subprocesses, without CUDA."""

import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import unittest

from checker.raw import read_run

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("REQSCOPE_LIBRARY", ROOT / "build/libreqscope.so"))
BIN = Path(os.environ.get("REQSCOPE_BIN", ROOT / "build/reqscope"))
BOOTSTRAP = '''
import ctypes, json, os, sys
lib = ctypes.CDLL(os.environ["REQSCOPE_LIBRARY"])
lib.InitializeInjection.restype = ctypes.c_int
lib.reqscope_emit_json.argtypes = [ctypes.c_char_p]
lib.reqscope_emit_json.restype = ctypes.c_int
lib.reqscope_flush.restype = ctypes.c_int
assert lib.InitializeInjection() == 1
def emit(name):
    row = {"type":"Request", "req_id":name, "start":1, "end":2, "prompt_tokens":1}
    return lib.reqscope_emit_json(json.dumps(row).encode())
'''


@unittest.skipUnless(LIBRARY.is_file(), "build libreqscope.so first")
class CollectorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rqs-collector-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "capture"
        self.socket = self.root / "control.sock"
        self.env = {**os.environ, "REQSCOPE_LIBRARY": str(LIBRARY), "REQSCOPE_MODE": "synthetic",
                    "REQSCOPE_BACKEND": "manual", "REQSCOPE_OUTPUT": str(self.output),
                    "REQSCOPE_SOCKET": str(self.socket), "REQSCOPE_QUEUE_SLOTS": "128"}

    def child(self, body):
        result = subprocess.run([sys.executable, "-c", BOOTSTRAP + body], env=self.env,
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def waiting_child(self, body):
        child = subprocess.Popen([sys.executable, "-c", BOOTSTRAP + body], env=self.env,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def cleanup():
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        self.addCleanup(cleanup)
        ready, _, _ = select.select([child.stdout], [], [], 10)
        self.assertTrue(ready, "collector child did not initialize in time")
        line = child.stdout.readline().strip()
        if line != "ready":
            _, stderr = child.communicate(timeout=5)
            self.fail(f"collector did not become ready: {line!r}; {stderr}")
        return child

    def control(self, request):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(10)
            sock.connect(str(self.socket))
            sock.sendall(json.dumps(request).encode() + b"\n")
            chunks = bytearray()
            while b"\n" not in chunks:
                data = sock.recv(8192)
                self.assertTrue(data, "control socket disconnected without response")
                chunks.extend(data)
            return json.loads(chunks.split(b"\n", 1)[0])

    def test_emit_flush_and_normal_shutdown_preserve_raw_records(self):
        self.child('''
assert emit("first") == 0
assert emit("second") == 0
assert lib.reqscope_flush() == 0
''')
        recording = read_run(self.output)
        self.assertFalse(recording.issues, recording.issues)
        rows = [r for r in recording.records if r["type"] == "Request"]
        self.assertEqual([r["req_id"] for r in rows], ["first", "second"])
        self.assertEqual(len(recording.headers), 1)
        self.assertFalse(self.socket.exists())

    def test_malformed_record_becomes_explicit_loss(self):
        self.child('''
lib.reqscope_emit_json(b"{invalid json")
assert lib.reqscope_flush() == 0
''')
        recording = read_run(self.output)
        losses = [r for r in recording.records if r["type"] == "Loss"]
        self.assertTrue(losses)
        self.assertTrue(any(r["reason"] == "malformed_record" and r["count"] >= 1 for r in losses))

    def test_queue_exhaustion_is_accounted_and_nonblocking(self):
        self.env["REQSCOPE_QUEUE_SLOTS"] = "2"
        result = self.child('''
payload = json.dumps({"type":"Request", "req_id":"load", "start":1, "end":2, "padding":"x"*4096}).encode()
attempts = 3000
accepted = sum(lib.reqscope_emit_json(payload) == 0 for _ in range(attempts))
assert lib.reqscope_flush() == 0
print(json.dumps({"attempts":attempts, "accepted":accepted}))
''')
        counts = json.loads(result.stdout)
        self.assertLess(counts["accepted"], counts["attempts"], "stress did not exercise queue exhaustion")
        recording = read_run(self.output)
        rows = [r for r in recording.records if r["type"] == "Request"]
        losses = [r for r in recording.records if r["type"] == "Loss" and r["reason"] == "queue_full"]
        self.assertEqual(len(rows), counts["accepted"])
        self.assertEqual(sum(r["count"] for r in losses), counts["attempts"] - counts["accepted"])
        self.assertTrue(all(len(r["seq_range"]) == 2 and len(r["time_range"]) == 2 for r in losses))
        self.assertTrue(all(r["count_unit"] == "records" for r in losses))

    def test_dump_is_a_stable_snapshot_and_refuses_overwrite(self):
        child = self.waiting_child('''
assert emit("before") == 0
assert lib.reqscope_flush() == 0
print("ready", flush=True)
sys.stdin.readline()
assert emit("after") == 0
assert lib.reqscope_flush() == 0
''')
        destination = self.root / "snapshot"
        answer = self.control({"command": "dump", "output": str(destination), "last_ns": 30_000_000_000})
        self.assertTrue(answer["ok"], answer)
        again = self.control({"command": "dump", "output": str(destination), "last_ns": 30_000_000_000})
        self.assertFalse(again["ok"])
        _, stderr = child.communicate("continue\n", timeout=10)
        self.assertEqual(child.returncode, 0, stderr)
        snapshot_rows = [r["req_id"] for r in read_run(destination).records if r["type"] == "Request"]
        live_rows = [r["req_id"] for r in read_run(self.output).records if r["type"] == "Request"]
        self.assertEqual(snapshot_rows, ["before"])
        self.assertEqual(live_rows, ["before", "after"])

    def test_sigterm_flushes_accepted_records(self):
        child = self.waiting_child('''
assert emit("last-record") == 0
print("ready", flush=True)
sys.stdin.readline()
''')
        child.send_signal(signal.SIGTERM)
        _, stderr = child.communicate(timeout=10)
        self.assertIn(child.returncode, (-signal.SIGTERM, 128 + signal.SIGTERM), stderr)
        rows = [r["req_id"] for r in read_run(self.output).records if r["type"] == "Request"]
        self.assertEqual(rows, ["last-record"])

    @unittest.skipUnless(BIN.is_file(), "build reqscope first")
    def test_run_cli_preserves_child_exit_and_capture_state(self):
        script = BOOTSTRAP + '\nassert emit("via-cli") == 0\nassert lib.reqscope_flush() == 0\n'
        result = subprocess.run([str(BIN), "run", "--manual", "--mode", "light", "--output", str(self.output),
                                 "--library", str(LIBRARY), "--", sys.executable, "-c", script],
                                env=self.env, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.output / "session.json").read_text())
        self.assertEqual(manifest["state"], "Finalized")
        rows = [r["req_id"] for r in read_run(self.output).records if r["type"] == "Request"]
        self.assertEqual(rows, ["via-cli"])
        failed = self.root / "failed"
        result = subprocess.run([str(BIN), "run", "--manual", "--output", str(failed), "--library", str(LIBRARY),
                                 "--", sys.executable, "-c", "raise SystemExit(37)"],
                                env=self.env, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 37)
        self.assertEqual(json.loads((failed / "session.json").read_text())["state"], "Partial")

    def test_small_segments_rotate_per_record_and_retention_stays_bounded(self):
        self.env.update(REQSCOPE_SEGMENT_BYTES="1024", REQSCOPE_RING_BYTES="65536")
        self.child('''
for identity in range(600):
    row = {"type":"Request", "req_id":str(identity), "start":1, "end":2, "padding":"x"*128}
    assert lib.reqscope_emit_json(json.dumps(row).encode()) == 0
    if identity % 8 == 0:
        assert lib.reqscope_flush() == 0
assert lib.reqscope_flush() == 0
''')
        import struct
        files = list(self.output.glob("*.rqs"))
        self.assertGreater(len(files), 1)
        self.assertLessEqual(len(files), 128)
        # A segment may pass its target by a header plus one complete record.
        for path in files:
            raw = path.read_bytes()
            header_size = struct.unpack_from("<I", raw, 8)[0] + 16
            self.assertLessEqual(len(raw), 1024 + header_size + 2048)
        self.assertLessEqual(sum(path.stat().st_size for path in files), 65536 + 8192)
        recording = read_run(self.output)
        self.assertTrue(any(row["type"] == "Loss" and row.get("reason") == "ring_overwrite"
                            for row in recording.records), "retention overwrite needs an explicit Loss")
        self.assertTrue(all(r["count_unit"] == "sequence_positions" for r in recording.records
                            if r["type"] == "Loss" and r.get("reason") == "ring_overwrite"))
        self.assertFalse(any("CRC" in issue or "truncated" in issue or "invalid JSON" in issue
                             for issue in recording.issues), recording.issues)

    def test_slow_step_trigger_creates_a_bounded_valid_snapshot(self):
        self.env["REQSCOPE_SLOW_STEP_NS"] = "100"
        self.env["REQSCOPE_MAX_TRIGGERS"] = "1"
        self.child('''
for identity in range(1, 5):
    row = {"type":"Step", "step_id":identity, "t_begin":100, "t_end":300,
           "tid":1, "members":[{"req_id":"slow", "tokens":1, "phase":"decode"}],
           "preempted":[], "finished":[]}
    assert lib.reqscope_emit_json(json.dumps(row).encode()) == 0
assert lib.reqscope_flush() == 0
''')
        snapshots = list((self.output / "triggers").iterdir())
        self.assertEqual(len(snapshots), 1)
        snapshot = read_run(snapshots[0])
        self.assertFalse(snapshot.issues, snapshot.issues)
        self.assertTrue(any(r["type"] == "Step" for r in snapshot.records))
        terminal = [r for r in snapshot.records if r["type"] == "Session"][-1]
        self.assertEqual(terminal["state"], "Snapshot")
        manifest = json.loads((snapshots[0] / "manifest.json").read_text())
        self.assertEqual(manifest["session_state"], "Snapshot")


    @unittest.skipUnless(BIN.is_file(), "build reqscope first")
    def test_plugin_only_capture_has_no_gpu_clock_bound(self):
        self.env.pop("REQSCOPE_BACKEND")
        self.env["REQSCOPE_MODE"] = "plugin"
        self.child('''
row = {"type":"Request", "req_id":"r", "start":1, "end":9, "prompt_tokens":1, "token_times":[2, 3, 4]}
assert lib.reqscope_emit_json(json.dumps(row).encode()) == 0
assert lib.reqscope_flush() == 0
''')
        recording = read_run(self.output)
        [header] = recording.headers.values()
        self.assertNotIn("clock_error_ns", header)
        self.assertFalse(header["clock_calibrated"])
        self.assertEqual(header["data_kind"], "plugin")
        [calibration] = [r for r in recording.records if r["type"] == "ClockCalibration"]
        self.assertFalse(calibration["calibrated"])
        blame = subprocess.run([str(BIN), "blame", str(self.output)], text=True, capture_output=True, timeout=20)
        self.assertEqual(blame.returncode, 0, blame.stderr)
        self.assertIsNone(json.loads(blame.stdout)["clocks"][0]["clock_error_ns"])

    def test_writer_failure_stops_the_plugin_without_raising_into_the_engine(self):
        # A file-size limit makes the writer's first drain fail with EFBIG, the
        # way a full disk fails a write.
        result = subprocess.run([sys.executable, "-c", WRITER_FAILURE], env={**self.env, "ROOT": str(ROOT)},
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        outcome = json.loads(result.stdout)
        self.assertTrue(outcome["stopped"], outcome)
        self.assertEqual(outcome["results"], ["served"])               # every step was served
        self.assertGreaterEqual(outcome["steps_after_stop"], 3)
        self.assertTrue(any("stopped collecting" in w for w in outcome["warnings"]), outcome["warnings"])
        [error] = list(self.output.glob("collector-error-*.txt"))
        self.assertIn("segment write failed", error.read_text())
        if BIN.is_file():
            blame = subprocess.run([str(BIN), "blame", str(self.output)], text=True, capture_output=True, timeout=20)
            self.assertEqual(blame.returncode, 0, blame.stderr)
            trace = json.loads(blame.stdout)["trace"]
            self.assertEqual(trace["status"], "Partial")
            self.assertIn("collector_failed", trace["issues_by_reason"])


WRITER_FAILURE = '''
import importlib.util, json, os, resource, signal, time, warnings
from types import SimpleNamespace
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (8192, 8192))
spec = importlib.util.spec_from_file_location(
    "adapter", os.path.join(os.environ["ROOT"], "adapter/vllm/reqscope_adapter/__init__.py"))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
class Runner:
    def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
        return "served"
    def sample_tokens(self, grammar=None):
        return "tokens"
collector = adapter.Collector()
adapter.install(Runner, collector)
runner = Runner()
step = SimpleNamespace(num_scheduled_tokens={f"request-{i}": 1 for i in range(64)},
                       scheduled_new_reqs=[], preempted_req_ids=set(), finished_req_ids=set())
results, after = set(), 0
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and after < 3:
        results.add(runner.execute_model(step))
        if not adapter.collecting(collector):
            after += 1
        time.sleep(0.01)
print(json.dumps({"stopped": not adapter.collecting(collector), "results": sorted(results),
                  "steps_after_stop": after, "warnings": [str(w.message) for w in caught]}))
'''


if __name__ == "__main__":
    unittest.main()
