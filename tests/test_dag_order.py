"""Black-box DAG and client token checks with handwritten expected outcomes."""

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from fixtures.fake_engine.generate import A, member, request, step, write_segment

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("REQSCOPE_BIN", ROOT / "build" / "reqscope"))


def api(corr, start, end, **extra):
    return dict(type="ApiCall", corr=corr, start=start, end=end, tid=43,
                tid_domain="system", return_value=0, **extra)


def activity(corr, start, end, stream=13, **extra):
    return dict(type="GpuActivity", corr=corr, start=start, end=end,
                ctx=1, stream=stream, kind="kernel", graph_id=0, graph_node_id=0, **extra)


@unittest.skipUnless(BIN.is_file(), "build reqscope first")
class DagOrderingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rqs-dag-order-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "recording.rqs"

    def report(self, rows, **kwargs):
        write_segment(self.path, rows, **kwargs)
        completed = subprocess.run([str(BIN), "analyze", str(self.path)],
                                   capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def stream_edges(self, report):
        return [edge for edge in report["edges"] if edge["source"] == "stream-order"]

    def test_same_thread_host_submissions_prove_order_without_gpu_clock_bound(self):
        # GPU timestamps deliberately appear reversed. Submission order is
        # established by APIs in the same host clock and CUDA stream semantics.
        rows = [api(1, 10, 20), activity(1, 200, 210),
                api(2, 30, 40), activity(2, 100, 110)]
        report = self.report(rows, clock_error_ns=None)
        edges = self.stream_edges(report)
        self.assertEqual([(e["from"], e["to"]) for e in edges], [(f"{A}:2", f"{A}:4")])
        self.assertEqual(edges[0]["ordering_basis"], "serialized-host-api")
        self.assertFalse(edges[0]["gpu_timestamps_used"])
        self.assertEqual(set(edges[0]["evidence_record_ids"]), {f"{A}:{i}" for i in (1, 2, 3, 4)})

    def test_stream_event_wait_dag_matches_handwritten_edge_set(self):
        rows = [api(1, 10, 20), activity(1, 100, 110),
                api(2, 30, 40), dict(type="CudaEvent", corr=2, ctx=1, stream=13, event=3, event_sync_id=4),
                api(3, 50, 60), dict(type="Sync", corr=3, ctx=1, stream=14, event=3, event_sync_id=4,
                                    sync_type=2, start=50, end=60, return_value=0),
                api(4, 70, 80), activity(4, 130, 140, stream=14)]
        rows += [request("x", 0, 200), step(1, 0, 200, [member("x")])]
        rows += [dict(type="ExtCorr", corr=corr, step_id=1) for corr in (1, 2, 3, 4)]
        report = self.report(rows)
        operation_ids = {f"{A}:{i}" for i in (2, 4, 6, 8)}
        edges = {(e["from"], e["to"], e["source"]) for e in report["edges"]
                 if e["from"] in operation_ids and e["to"] in operation_ids}
        self.assertEqual(edges, {(f"{A}:2", f"{A}:4", "stream-order"),
                                 (f"{A}:4", f"{A}:6", "sync-record"),
                                 (f"{A}:6", f"{A}:8", "stream-order")})
        completed = subprocess.run([str(BIN), "explain", str(self.path), "--request", "x"],
                                   capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        explanation = json.loads(completed.stdout)
        self.assertEqual({(e["from"], e["to"], e["source"]) for e in explanation["edges"]
                          if e["from"] in operation_ids and e["to"] in operation_ids}, edges)
        self.assertEqual(explanation["request"]["activity_record_ids"], [f"{A}:2", f"{A}:8"])

    def test_ambiguous_or_unordered_submission_never_becomes_stream_edge(self):
        rows = [api(1, 10, 20), activity(1, 100, 110), api(2, 30, 40), activity(2, 120, 130)]
        for change in ("overlap", "equal_boundary", "thread", "domain", "failure", "graph", "context", "stream", "missing_tid"):
            changed = copy.deepcopy(rows)
            if change == "overlap": changed[2]["start"] = 15
            if change == "equal_boundary": changed[2]["start"] = 20
            if change == "thread": changed[2]["tid"] = 44
            if change == "domain": changed[2]["tid_domain"] = "pthread"
            if change == "failure": changed[2]["return_value"] = 1
            if change == "graph": changed[3]["graph_id"] = 7
            if change == "context": changed[3]["ctx"] = 2
            if change == "stream": changed[3]["stream"] = 14
            if change == "missing_tid": del changed[2]["tid"]
            with self.subTest(change=change):
                self.assertEqual(self.stream_edges(self.report(changed)), [])

    def test_missing_or_ambiguous_correlation_is_not_replaced_by_gpu_order(self):
        for rows in ([activity(1, 100, 110), activity(2, 120, 130)],
                     [api(1, 10, 20), api(1, 11, 19), activity(1, 100, 110), api(2, 30, 40), activity(2, 120, 130)]):
            with self.subTest(rows=rows):
                self.assertEqual(self.stream_edges(self.report(rows)), [])

    def test_loss_covering_host_submission_prevents_stream_edge(self):
        rows = [api(1, 10, 20), activity(1, 100, 110), api(2, 30, 40), activity(2, 120, 130),
                dict(type="Loss", count=1, reason="buffer_exhausted", time_range=[15, 35])]
        self.assertEqual(self.stream_edges(self.report(rows)), [])

    def test_graph_node_activity_times_cannot_invent_graph_topology(self):
        rows = [api(1, 10, 20), activity(1, 100, 110), activity(1, 120, 130)]
        rows[1].update(graph_id=7, graph_node_id=1)
        rows[2].update(graph_id=7, graph_node_id=2)
        self.assertEqual(self.stream_edges(self.report(rows)), [])

    def test_repeated_arrival_timestamps_preserve_token_count_and_zero_intervals(self):
        req = {**request("x", 10, 100), "token_times": [20, 20, 40],
               "token_time_semantics": "client-arrival"}
        result = self.report([req, step(1, 10, 100, [member("x")])])["requests"][0]
        self.assertEqual(result["status"], "Complete")
        self.assertEqual(result["token_time_semantics"], "client-arrival")
        self.assertEqual([r["token_index"] for r in result["token_intervals"]], [0, 1, 2])
        self.assertEqual([r["duration_ns"] for r in result["token_intervals"]], [10, 0, 20])
        self.assertEqual(sum(result["token_intervals"][1]["components_ns"].values()), 0)
        self.assertEqual(result["after_last_token_ns"], 60)

    def test_any_malformed_token_timestamp_invalidates_entire_optional_sequence(self):
        for value in (None, "20", 20, [20, "30", 40], [20, True], [20, -1], [20, 30.5],
                      [20, 9], [20, 101], [40, 20], [20, {}]):
            with self.subTest(value=value):
                req = {**request("x", 10, 100), "token_times": value}
                result = self.report([req, step(1, 10, 100, [member("x")])])["requests"][0]
                self.assertEqual(result["status"], "Partial")
                self.assertIn("invalid_token_times", result["reasons"])
                self.assertEqual(result["token_intervals"], [])
                self.assertIsNone(result["after_last_token_ns"])
                self.assertEqual(sum(result["components_ns"].values()), 90)

    def test_explicit_external_alias_resolves_one_internal_identity(self):
        req = {**request("public-0-a1b2c3d4", 10, 100), "external_request_id": "public-0",
               "identity_source": "otel-reqscope.internal_request_id"}
        report = self.report([req, step(1, 10, 100, [member("public-0-a1b2c3d4")])])
        self.assertEqual(report["requests"][0]["external_request_id"], "public-0")
        self.assertEqual(report["requests"][0]["identity_source"], "otel-reqscope.internal_request_id")
        completed = subprocess.run([str(BIN), "explain", str(self.path), "--request", "public-0"],
                                   capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        selected = json.loads(completed.stdout)["request"]
        self.assertEqual(selected["request_id"], "public-0-a1b2c3d4")
        self.assertEqual(selected["step_ids"], [1])

    def test_reused_external_alias_is_ambiguous_and_never_merges_requests(self):
        rows = [{**request("internal-a", 10, 40), "external_request_id": "public"},
                {**request("internal-b", 50, 100), "external_request_id": "public"},
                step(1, 10, 40, [member("internal-a")]), step(2, 50, 100, [member("internal-b")])]
        report = self.report(rows)
        self.assertEqual([(r["request_id"], r["step_ids"]) for r in report["requests"]],
                         [("internal-a", [1]), ("internal-b", [2])])
        completed = subprocess.run([str(BIN), "explain", str(self.path), "--request", "public"],
                                   capture_output=True, text=True, timeout=10)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ambiguous", completed.stderr)
        completed = subprocess.run([str(BIN), "explain", str(self.path), "--request", f"request:{A}:internal-b"],
                                   capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["request"]["step_ids"], [2])

    def test_no_alias_never_strips_random_suffix_or_guesses_external_join(self):
        self.report([request("public-0-a1b2c3d4", 10, 100), step(1, 10, 100, [member("public-0-a1b2c3d4")])])
        completed = subprocess.run([str(BIN), "explain", str(self.path), "--request", "public-0"],
                                   capture_output=True, text=True, timeout=10)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not found", completed.stderr)


if __name__ == "__main__":
    unittest.main()
