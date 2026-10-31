"""Black-box CLI checks. No imports from the analyzer/collector implementation."""

import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib

from checker.raw import read_run
from checker.oracle import check
from fixtures.fake_engine.generate import A, B, TRUTH, frame_bytes, generate, gpu, header, launch, member, replay_records, request, step, write_segment

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("REQSCOPE_BIN", ROOT / "build" / "reqscope"))


@unittest.skipUnless(BIN.is_file(), "build reqscope first: cmake -S . -B build && cmake --build build")
class AnalyzerBlackBoxTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="reqscope-analyzer-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def cli(self, *args, success=True):
        result = subprocess.run([str(BIN), *map(str, args)], capture_output=True, text=True, timeout=10)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def analyze(self, path):
        return json.loads(self.cli("analyze", path).stdout)

    def test_host_observation_survives_unresolved_gpu_clock_but_rejects_bad_step_boundary(self):
        for end, expected in ((140, "Observed"), (240, "Partial")):
            with self.subTest(end=end):
                rows = [request("x", 0, 300), step(1, 100, 200, [member("x")]),
                        dict(type="HostSubmitWait", step_id=1, tid=43, start=110, end=end,
                             clock="CLOCK_MONOTONIC")]
                path = write_segment(self.root / f"host-{end}.rqs", rows)
                report = json.loads(self.cli("explain", path, "--request", "x").stdout)
                wait = report["host_submit_waits"][0]
                self.assertEqual(wait["status"], expected)
                self.assertEqual(wait["duration_ns"], end - 110)
                self.assertNotIn("injected", wait["classification"])
                self.assertIn("unresolved", wait["scope"])

    def probe_trace(self, name, probes):
        """One request, one step with one kernel; ClockProbe brackets around probe copies."""
        rows = [request("x", 0, 1000), step(1, 100, 400, [member("x")]), *launch(5, 1, 110), gpu(5, 150, 350)]
        for i, (t0, t1, copy) in enumerate(probes, 1):
            rows.append(dict(type="ClockProbe", probe=i, t0=t0, t1=t1, ctx=9, stream=99))
            if copy:
                rows.append(dict(type="GpuActivity", corr=900 + i, kind="memcpy", ctx=9, stream=99,
                                 start=copy[0], end=copy[1], graph_id=0, graph_node_id=0, bytes=4))
        return write_segment(self.root / name, rows, omit_clock_error=True)

    def test_clock_probes_bound_the_session_and_unlock_gpu_attribution(self):
        report = self.analyze(self.probe_trace("probes.rqs", [(10, 30, (15, 25)), (600, 640, (610, 620))]))
        clock = report["clocks"][0]
        self.assertEqual(clock["clock_error_ns"], 40)                 # widest bracket
        self.assertEqual(clock["probe"]["paired"], 2)
        self.assertEqual(clock["probe"]["max_violation_ns"], 0)
        self.assertEqual(report["summary"]["gpu_activities"], 1)      # probe copies are not app work
        components = report["requests"][0]["components_ns"]
        self.assertEqual(components["gpu_exec"], 200)
        self.assertGreater(components["between_steps"] + components["gpu_idle_in_step"], 0)

    def test_clock_probe_outside_its_bracket_widens_the_bound(self):
        report = self.analyze(self.probe_trace("violation.rqs", [(10, 30, (5, 25))]))
        self.assertEqual(report["clocks"][0]["clock_error_ns"], 25)   # width 20 + 5 outside
        self.assertEqual(report["clocks"][0]["probe"]["max_violation_ns"], 5)

    def test_probe_after_the_last_delivered_gpu_record_is_pending_not_unpaired(self):
        # The kernel ends at 350; the second probe completes at 640 and its
        # copy was not delivered before the capture ended.
        report = self.analyze(self.probe_trace("pending.rqs", [(10, 30, (15, 25)), (600, 640, None)]))
        clock = report["clocks"][0]
        self.assertEqual(clock["probe"]["pending_at_capture_end"], 1)
        self.assertEqual(clock["probe"]["unpaired"], 0)
        self.assertEqual(clock["clock_error_ns"], 20)

    def test_unpaired_clock_probe_leaves_the_bound_unknown(self):
        # The first probe's copy is missing although later GPU work was delivered.
        report = self.analyze(self.probe_trace("unpaired.rqs", [(10, 30, None), (600, 640, (610, 620))]))
        self.assertIsNone(report["clocks"][0]["clock_error_ns"])
        self.assertEqual(report["requests"][0]["components_ns"]["gpu_exec"], 0)

    def test_blame_attributes_a_slow_token_to_the_co_scheduled_prefill(self):
        a = dict(request("a", 0, 1000), token_times=[100, 200, 300, 800, 900])
        b = dict(request("b", 250, 1000, prompt_tokens=2048), token_times=[790, 900, 990])
        rows = [a, b,
                step(1, 110, 190, [member("a")]), step(2, 210, 290, [member("a")]),
                step(3, 310, 790, [member("a"), member("b", tokens=2048, phase="prefill")]),
                *launch(30, 3, 315), gpu(30, 320, 780),
                step(4, 810, 890, [member("a"), member("b")])]
        path = write_segment(self.root / "blame.rqs", rows)
        report = json.loads(self.cli("blame", path, "--request", "a").stdout)
        [row] = report["requests"]
        self.assertEqual(row["median_gap_ns"], 100)
        [gap] = row["slow_gaps"]
        self.assertEqual((gap["token_index"], gap["gap_ns"], gap["excess_ns"]), (3, 500, 400))
        self.assertEqual((gap["begin_ns"], gap["end_ns"]), (300, 800))
        [produced_by] = gap["steps"]
        self.assertEqual(produced_by["step_id"], 3)
        self.assertEqual(produced_by["gpu_busy_ns"], 460)
        self.assertEqual(gap["blamed"], [{"req_id": "b", "prefill_tokens": 2048, "excess_ns": 400}])
        self.assertEqual(report["culprits"], [{"req_id": "b", "excess_caused_ns": 400}])
        self.assertEqual(row["unexplained_ns"], 0)

    def test_blame_follows_token_steps_when_the_next_step_is_already_launched(self):
        # Async scheduling: step k+1 is launched before step k's token is
        # processed. Step 4 carries b's prefill; step 5's host span is the one
        # that ends inside the slow gap (300, 800].
        steps = [step(1, 10, 40, [member("a")]), step(2, 60, 90, [member("a")]),
                 step(3, 150, 190, [member("a")]),
                 step(4, 250, 290, [member("a"), member("b", tokens=2048, phase="prefill")]),
                 step(5, 310, 330, [member("a"), member("b")])]
        b = dict(request("b", 200, 1000, prompt_tokens=2048), token_times=[790, 900, 990])
        for token_steps, producer, culprit in (([1, 2, 3, 4, 5], 4, "b"), (None, 5, None)):
            with self.subTest(token_steps=token_steps):
                a = dict(request("a", 0, 1000), token_times=[100, 200, 300, 800, 900])
                if token_steps:
                    a["token_steps"] = token_steps
                path = write_segment(self.root / f"async-{producer}.rqs", [a, b, *steps])
                [gap] = json.loads(self.cli("blame", path, "--request", "a").stdout)["requests"][0]["slow_gaps"]
                self.assertEqual([s["step_id"] for s in gap["steps"]], [producer])
                self.assertEqual([x["req_id"] for x in gap["blamed"]], [culprit] if culprit else [])

    def test_blame_reassembles_tokens_flushed_in_chunks(self):
        chunk = dict(type="RequestTokens", req_id="a", first=0, token_times=[100, 200], token_steps=[1, 2])
        a = dict(request("a", 0, 1000), token_times=[300, 800, 900], token_steps=[3, 4, 5], token_offset=2)
        rows = [chunk, a, dict(request("b", 250, 1000, prompt_tokens=2048), token_times=[790, 900, 990]),
                step(1, 50, 90, [member("a")]), step(2, 150, 190, [member("a")]), step(3, 210, 290, [member("a")]),
                step(4, 310, 330, [member("a"), member("b", tokens=2048, phase="prefill")]),
                step(5, 810, 830, [member("a"), member("b")])]
        report = json.loads(self.cli("blame", write_segment(self.root / "chunks.rqs", rows), "--request", "a").stdout)
        [row] = report["requests"]
        self.assertEqual(row["tokens"], 5)
        [gap] = row["slow_gaps"]
        self.assertEqual(([s["step_id"] for s in gap["steps"]], gap["blamed"][0]["req_id"]), ([4], "b"))

    def test_blame_reports_slow_gaps_without_prefill_as_unexplained(self):
        a = dict(request("a", 0, 1000), token_times=[100, 200, 300, 800, 900])
        rows = [a, step(1, 110, 190, [member("a")]), step(3, 310, 790, [member("a")])]
        report = json.loads(self.cli("blame", write_segment(self.root / "lone.rqs", rows)).stdout)
        self.assertEqual(report["requests"][0]["unexplained_ns"], 400)
        self.assertEqual(report["culprits"], [])

    def test_gpu_records_without_timestamps_are_counted_not_timed(self):
        a = dict(request("a", 0, 1000), token_times=[100, 200, 300])
        rows = [a, step(1, 110, 190, [member("a")]), *launch(5, 1, 120), gpu(5, 130, 170),
                *launch(6, 1, 125), gpu(6, 0, 0)]
        report = json.loads(self.cli("blame", write_segment(self.root / "untimed.rqs", rows)).stdout)
        self.assertEqual(report["summary"]["untimed_gpu_activities"], 1)

    def test_trace_over_the_parser_budget_fails_instead_of_truncating(self):
        rows = [step(i, 10 * i, 10 * i + 5, [member("x")]) for i in range(1, 20000)]
        path = write_segment(self.root / "big.rqs", rows)
        result = subprocess.run([str(BIN), "analyze", str(path)], capture_output=True, text=True, timeout=30,
                                env={**os.environ, "REQSCOPE_TRACE_BUDGET_MIB": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("memory budget", result.stderr)

    def test_replays_padding_pid_reuse_overlap_and_preemption(self):
        path = generate(self.root / "replay")
        report = self.analyze(path)
        self.assertEqual(check(read_run(path), report, TRUTH), [])
        self.assertEqual(report["summary"]["gpu_activities"], 6)
        self.assertEqual(report["summary"]["linked_gpu_activities"], 5)
        estimates = sum(r["gpu_share_estimate_ns"] for r in report["requests"])
        self.assertEqual(estimates, 270)

    def test_independent_oracle_rejects_mutated_results(self):
        path = generate(self.root / "mutation")
        report = self.analyze(path)
        recording = read_run(path)
        for mutation in ("membership", "provenance", "conservation", "interval"):
            bad = copy.deepcopy(report)
            if mutation == "membership":
                bad["activities"][0]["request_ids"] = ["invented-request"]
            elif mutation == "provenance":
                bad["edges"][0]["source"] = "timestamp-nearby"
            elif mutation == "conservation":
                bad["requests"][0]["components_ns"]["residual"] += 1
            else:
                bad["requests"][0]["intervals"][0]["end_ns"] += 1
            self.assertTrue(check(recording, bad, TRUTH), mutation)

    def test_missing_external_correlation_never_links_by_proximity(self):
        rows = [request("x", 0, 100), step(1, 10, 90, [member("x")]),
                dict(type="ApiCall", corr=9, tid=43, start=11, end=12, cbid="cudaLaunchKernel"), gpu(9, 12, 80)]
        path = write_segment(self.root / "missing.rqs", rows)
        report = self.analyze(path)
        self.assertEqual(report["activities"][0]["request_ids"], [])
        self.assertEqual(report["summary"]["unlinked_gpu_activities"], 1)
        self.assertEqual(report["requests"][0]["components_ns"]["gpu_exec"], 0)

    def test_conflicting_external_correlation_is_unknown(self):
        path = generate(self.root / "ambiguous", "ambiguous")
        report = self.analyze(path)
        ambiguous = [r for r in report["activities"] if r["producer"] == A and r["corr"] == 1]
        self.assertEqual(len(ambiguous), 2)
        self.assertTrue(all(not r["request_ids"] for r in ambiguous))
        self.assertEqual(check(read_run(path), report, json.loads((path / "truth.json").read_text())), [])

    def test_loss_propagates_to_request_and_gap(self):
        path = generate(self.root / "loss", "loss")
        report = self.analyze(path)
        self.assertEqual(report["status"].lower(), "partial")
        alpha = next(r for r in report["requests"] if r["request_id"] == "alpha")
        self.assertEqual(alpha["status"].lower(), "partial")
        affected = [g for g in report["gaps"] if g["producer"] == A and g["start_ns"] < 200 and g["end_ns"] > 100]
        self.assertTrue(affected)
        self.assertTrue(all(g["category"] == "unknown" for g in affected))
        self.assertEqual(check(read_run(path), report, TRUTH), [])

    def test_clock_bound_prevents_short_gap_attribution(self):
        path = generate(self.root / "clock", "clock")
        report = self.analyze(path)
        short_gaps = [g for g in report["gaps"] if g["end_ns"] - g["start_ns"] < 50]
        self.assertTrue(short_gaps)
        self.assertTrue(all(g["category"] == "unknown" for g in short_gaps), short_gaps)
        self.assertEqual(check(read_run(path), report, TRUTH), [])

    def test_explicit_scheduler_and_sync_evidence(self):
        rows = [request("x", 0, 400), step(1, 100, 300, [member("x")]),
                *launch(1, 1, 145), gpu(1, 150, 200),
                dict(type="Sched", tid=43, start=100, end=150, state="runnable"),
                dict(type="Sched", tid=43, start=200, end=300, state="sleeping"),
                dict(type="ApiCall", corr=2, tid=43, start=220, end=280, cbid="cudaStreamSynchronize"),
                dict(type="Sync", corr=2, tid=43, stream=1, event=1, start=220, end=280, sync_type="stream_synchronize")]
        path = write_segment(self.root / "scheduler.rqs", rows)
        categories = {g["category"] for g in self.analyze(path)["gaps"]}
        self.assertIn("launch_thread_runnable", categories)
        self.assertIn("blocked_on_sync", categories)

    def test_stream_wait_event_does_not_imply_cpu_blocking(self):
        rows = [request("x", 0, 400), step(1, 100, 300, [member("x")]),
                *launch(1, 1, 145), gpu(1, 150, 200),
                dict(type="Sched", tid=43, start=200, end=300, state="on_cpu"),
                dict(type="ApiCall", corr=2, tid=43, start=200, end=300, cbid="cudaStreamWaitEvent"),
                dict(type="Sync", corr=2, tid=43, stream=1, event=1, start=200, end=300, sync_type="stream_wait_event")]
        path = write_segment(self.root / "stream-wait.rqs", rows)
        categories = {g["category"] for g in self.analyze(path)["gaps"]}
        self.assertNotIn("blocked_on_sync", categories)
        self.assertIn("launch_thread_on_cpu", categories)

    def test_missing_request_span_does_not_fabricate_e2e(self):
        path = write_segment(self.root / "no-span.rqs", [step(1, 10, 90, [member("x")]), *launch(1, 1, 15), gpu(1, 20, 80)])
        report = self.analyze(path)
        row = next(r for r in report["requests"] if r["request_id"] == "x")
        self.assertIsNone(row["e2e_ns"])
        self.assertEqual(row["status"].lower(), "partial")

    def test_explain_filters_request(self):
        path = generate(self.root / "replay")
        result = json.loads(self.cli("explain", path, "--request", "alpha").stdout)
        self.assertTrue(result)
        serialized = json.dumps(result)
        self.assertIn("alpha", serialized)

    def test_perfetto_flow_pairs_and_time_units(self):
        path = generate(self.root / "replay")
        report = json.loads(self.cli("export-perfetto", path).stdout)
        events = report["traceEvents"]
        begins = {e["id"] for e in events if e.get("ph") == "s"}
        ends = {e["id"] for e in events if e.get("ph") == "f"}
        self.assertTrue(begins)
        self.assertEqual(begins, ends)
        self.assertTrue(all(e["dur"] >= 0 for e in events if "dur" in e))
        self.assertTrue(all(e.get("ts", 0) < 1 for e in events))  # fixture is < 1 us

    def test_diff_identical_recordings(self):
        path = generate(self.root / "replay")
        result = json.loads(self.cli("diff", path, path, "--by", "prompt-bucket").stdout)
        self.assertEqual(len(result["groups"]), 1)
        group = result["groups"][0]
        self.assertEqual(group["before"]["count"], 4)
        self.assertEqual(group["before"]["mean_e2e_ns"], 430)
        self.assertEqual(group["before"]["mean_components_ns"]["gpu_exec"], 90)
        self.assertEqual(group["delta_mean_e2e_ns"], 0)
        self.assertTrue(all(value == 0 for value in group["delta_mean_components_ns"].values()))

    def test_diff_reports_handwritten_delta_and_withholds_ci_for_loss(self):
        before = generate(self.root / "before")
        after = self.root / "after"
        rows = replay_records()
        rows[A][0]["end"] = 700  # Exactly 100 ns additional alpha residual; four requests.
        for index, (producer, records) in enumerate(rows.items()):
            write_segment(after / f"producer-{index}.rqs", records, producer)
        report = json.loads(self.cli("diff", before, after).stdout)
        group = report["groups"][0]
        self.assertEqual(group["delta_mean_e2e_ns"], 25)
        self.assertEqual(group["delta_mean_components_ns"]["residual"], 25)
        self.assertEqual(group["delta_mean_components_ns"]["gpu_exec"], 0)
        lost = generate(self.root / "lost", "loss")
        report = json.loads(self.cli("diff", before, lost).stdout)
        self.assertEqual(report["status"], "Partial")
        self.assertTrue(all(value is None for value in report["groups"][0]["bootstrap_95pct_ci_ns"].values()))

    def test_corruption_is_rejected_or_reported_partial(self):
        base = b"RQSSEG01" + frame_bytes(header())
        valid = frame_bytes(dict(type="Request", producer=A, seq=1, req_id="x", start=1, end=2))
        damaged = bytearray(valid)
        damaged[-1] ^= 1
        payload = b'{"type":"Request","producer":"' + A.encode() + b'","seq":1,"seq":2}'
        duplicate_keys = struct.pack("<II", len(payload), zlib.crc32(payload)) + payload
        cases = {"truncated": base + valid[:-3], "crc": base + damaged,
                 "length-bomb": base + struct.pack("<II", 0xFFFFFFFF, 0),
                 "duplicate-keys": base + duplicate_keys, "bad-magic": b"NOTTRACE",
                 "negative-time": base + frame_bytes(dict(type="GpuActivity", producer=A, seq=1, corr=1, start=-1, end=2)),
                 "reversed-time": base + frame_bytes(dict(type="GpuActivity", producer=A, seq=1, corr=1, start=20, end=2))}
        for name, data in cases.items():
            with self.subTest(case=name):
                path = self.root / f"{name}.rqs"
                path.write_bytes(data)
                result = self.cli("analyze", path, success=False)
                self.assertGreaterEqual(result.returncode, 0, "parser crashed")
                if result.returncode == 0:
                    report = json.loads(result.stdout)
                    self.assertEqual(report["status"].lower(), "partial", result.stdout)

    def test_header_identity_and_clock_are_validated(self):
        variants = [{**header(), "pid": 999}, {**header(), "clock_error_ns": -1},
                    {**header(), "clock": "CLOCK_REALTIME"}]
        for index, invalid in enumerate(variants):
            with self.subTest(header=invalid):
                path = self.root / f"header-{index}.rqs"
                path.write_bytes(b"RQSSEG01" + frame_bytes(invalid))
                result = self.cli("analyze", path, success=False)
                self.assertGreaterEqual(result.returncode, 0)
                if result.returncode == 0:
                    self.assertEqual(json.loads(result.stdout)["status"].lower(), "partial")

    def test_inconsistent_producer_clock_across_segments_is_partial(self):
        directory = self.root / "clock-change"
        directory.mkdir()
        write_segment(directory / "a.rqs", [request("x", 0, 100)], clock_error_ns=1)
        write_segment(directory / "b.rqs", [{**request("y", 0, 100), "seq": 2}], clock_error_ns=100)
        report = self.analyze(directory)
        self.assertEqual(report["status"].lower(), "partial")

    def test_session_boundary_distinguishes_snapshot_from_abrupt_exit(self):
        work = [request("x", 0, 100), step(1, 10, 90, [member("x")]), *launch(1, 1, 10), gpu(1, 10, 90)]
        cases = {
            "abrupt": (work, True),
            "snapshot": (work + [{"type": "Session", "state": "Snapshot"}], False),
            "stale-snapshot": ([{"type": "Session", "state": "Snapshot"}] + work, True),
            "resumed-snapshot": (work + [{"type": "Session", "state": "Snapshot"},
                                         {"type": "Session", "state": "Capturing"}], True),
            "partial-stays-partial": ([{"type": "Session", "state": "Partial"}] + work +
                                      [{"type": "Session", "state": "Finalized"}], True),
        }
        for name, (rows, partial) in cases.items():
            with self.subTest(case=name):
                path = self.root / f"{name}.rqs"
                data = b"RQSSEG01" + frame_bytes({**header(), "session_state": "Capturing"})
                data += b"".join(frame_bytes({"producer": A, "seq": i, **row}) for i, row in enumerate(rows, 1))
                path.write_bytes(data)
                report = self.analyze(path)
                self.assertEqual(report["status"].lower() == "partial", partial, report)

    def test_scheduler_source_clock_bound_cannot_be_ignored(self):
        directory = self.root / "bpf-clock"
        directory.mkdir()
        write_segment(directory / "engine.rqs", [request("x", 0, 400), step(1, 100, 300, [member("x")]),
                                                  *launch(1, 1, 145), gpu(1, 150, 200)])
        write_segment(directory / "bpf.rqs", [{"type": "Sched", "target_producer": A, "tid": 43,
                                                "start": 0, "end": 1000, "state": "runnable"}], B, 100)
        report = self.analyze(directory)
        early = [g for g in report["gaps"] if g["producer"] == A and g["start_ns"] == 100]
        self.assertTrue(early)
        self.assertTrue(all(g["category"] == "unknown" for g in early))

    def test_schedstat_aggregate_does_not_invent_runnable_interval(self):
        path = write_segment(self.root / "schedstat.rqs", [request("x", 0, 400), step(1, 100, 300, [member("x")]),
                                                          *launch(1, 1, 145), gpu(1, 150, 200),
                                                          {"type": "SchedSample", "tid": 43, "start": 100, "end": 300,
                                                           "runnable_ns": 200, "state": "runnable"}])
        self.assertTrue(all(g["category"] == "unknown" for g in self.analyze(path)["gaps"]))

    def test_graph_nodes_do_not_gain_dependencies_from_timestamp_order(self):
        rows = [request("x", 0, 400), step(1, 50, 300, [member("x")]), *launch(1, 1, 90),
                {**gpu(1, 100, 120, 1), "stream": 1}, {**gpu(1, 150, 180, 2), "stream": 1}]
        report = self.analyze(write_segment(self.root / "graph-dag.rqs", rows))
        self.assertFalse(any(edge["source"] == "stream-order" for edge in report["edges"]))

    def test_empty_later_step_cannot_explain_earlier_gpu_gap(self):
        rows = [request("x", 0, 100), step(1, 10, 20, [member("x")]),
                step(2, 40, 50, []), *launch(1, 1, 10), gpu(1, 12, 18)]
        report = self.analyze(write_segment(self.root / "empty-later.rqs", rows))
        between = [g for g in report["gaps"] if g["start_ns"] < 40 and g["end_ns"] > 20]
        self.assertTrue(between)
        self.assertTrue(all(g["category"] != "no_work_scheduled" for g in between), between)

    def test_request_without_any_step_is_partial(self):
        report = self.analyze(write_segment(self.root / "no-step.rqs", [request("x", 0, 100)]))
        self.assertEqual(report["requests"][0]["status"].lower(), "partial")

    def event_rows(self, event_sync_id=7):
        return [dict(type="ApiCall", corr=1, tid=43, start=1, end=2, name="cudaEventRecord"),
                dict(type="CudaEvent", corr=1, ctx=1, stream=1, event=9, event_sync_id=event_sync_id, device_timestamp=0),
                dict(type="ApiCall", corr=2, tid=43, start=3, end=4, name="cudaStreamWaitEvent"),
                dict(type="Sync", corr=2, ctx=1, stream=2, event=9, event_sync_id=event_sync_id,
                     start=3, end=4, sync_type=2, return_value=0)]

    def test_explicit_event_identity_works_without_temporal_certainty(self):
        report = self.analyze(write_segment(self.root / "event.rqs", self.event_rows(), clock_error_ns=100))
        links = {(e["from"], e["to"], e["source"]) for e in report["edges"]}
        self.assertIn((f"{A}:1", f"{A}:2", "cupti-correlation"), links)
        self.assertIn((f"{A}:2", f"{A}:4", "sync-record"), links)
        self.assertEqual(report["synchronization"][0]["event_record_id"], f"{A}:2")

    def test_event_sentinels_duplicates_context_and_failed_wait_never_link(self):
        variants = [(str(value), self.event_rows(value)) for value in (0, 2 ** 32 - 1, 2 ** 64 - 1)]
        duplicate = self.event_rows()
        duplicate.append(dict(duplicate[1]))
        variants.append(("duplicate", duplicate))
        for field, value in (("ctx", 2), ("event", 8), ("return_value", 1)):
            rows = self.event_rows()
            rows[-1][field] = value
            variants.append((field, rows))
        for name, rows in variants:
            with self.subTest(case=name):
                report = self.analyze(write_segment(self.root / f"event-{name}.rqs", rows))
                self.assertIsNone(report["synchronization"][0]["event_record_id"])

    def test_event_identity_is_scoped_to_producer_even_with_reused_pid(self):
        directory = self.root / "event-pid-reuse"
        directory.mkdir()
        rows = self.event_rows()
        write_segment(directory / "old.rqs", rows[:2], A)
        write_segment(directory / "new.rqs", rows[2:], B)
        report = self.analyze(directory)
        self.assertIsNone(report["synchronization"][0]["event_record_id"])

    def causal_rows(self, sync_end):
        # Handwritten window: launch [110, 120], GPU [200, 400], sync API ends at sync_end.
        # Error is the farthest observed GPU timestamp from [110, sync_end].
        return [request("x", 0, 1000),
                step(1, 100, 900, [member("x")]),
                dict(type="ExtCorr", corr=9, step_id=1),
                dict(type="ApiCall", corr=9, tid=43, tid_domain="system", start=110, end=120,
                     return_value=0, name="cudaLaunchKernel"),
                dict(type="GpuActivity", corr=9, kind="kernel", ctx=1, stream=7, start=200, end=400,
                     graph_id=0, graph_node_id=0),
                dict(type="ApiCall", corr=10, tid=43, tid_domain="system", start=500, end=sync_end,
                     return_value=0, name="cudaStreamSynchronize"),
                dict(type="Sync", corr=10, sync_type=3, ctx=1, stream=7, start=500, end=sync_end, return_value=0)]

    def test_missing_header_clock_uses_per_activity_causal_window(self):
        path = write_segment(self.root / "bracket.rqs", self.causal_rows(530), omit_clock_error=True,
                             header_extra={"thread_id_type": "system"})
        report = self.analyze(path)
        activity = report["activities"][0]
        basis = activity["clock_basis"]
        self.assertEqual(basis["method"], "per-activity-host-causal-window")
        self.assertEqual(basis["lower_ns"], 110)
        self.assertEqual(basis["upper_ns"], 530)
        self.assertEqual(basis["per_activity_timestamp_error_bound_ns"], 330)
        self.assertTrue(basis["timestamps_inside_bracket"])
        self.assertFalse(basis["global_offset_calibrated"])
        self.assertFalse(activity["timing_reliable"])
        request_row = report["requests"][0]
        self.assertNotIn("missing_clock_calibration", request_row["reasons"])
        self.assertEqual(request_row["clock_error_basis_ns"], 330)
        self.assertEqual(request_row["clock_basis_method"], "per-activity-host-causal-window")
        clock = report["clocks"][0]
        self.assertIsNone(clock["clock_error_ns"])
        self.assertFalse(clock["global_offset_calibrated"])
        self.assertEqual(clock["bracket_covered"], 1)
        self.assertEqual(clock["bracket_max_error_ns"], 330)

    def test_uncovered_activity_stays_missing_clock_calibration(self):
        rows = self.causal_rows(530)[:-2]
        path = write_segment(self.root / "no-sync.rqs", rows, omit_clock_error=True,
                             header_extra={"thread_id_type": "system"})
        report = self.analyze(path)
        self.assertNotIn("clock_basis", report["activities"][0])
        self.assertIn("missing_clock_calibration", report["requests"][0]["reasons"])
        self.assertIsNone(report["clocks"][0]["clock_error_ns"])
        self.assertEqual(report["clocks"][0]["bracket_covered"], 0)

    def test_causal_window_wider_than_request_stays_unresolved(self):
        path = write_segment(self.root / "wide.rqs", self.causal_rows(50000), omit_clock_error=True,
                             header_extra={"thread_id_type": "system"})
        report = self.analyze(path)
        reasons = report["requests"][0]["reasons"]
        self.assertNotIn("missing_clock_calibration", reasons)
        self.assertIn("clock_window_wider_than_request", reasons)
        self.assertEqual(report["requests"][0]["status"], "Partial")
        self.assertEqual(report["activities"][0]["clock_basis"]["per_activity_timestamp_error_bound_ns"], 49800)


if __name__ == "__main__":
    unittest.main()
