import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from checker.raw import read_run
from fixtures.fake_engine.generate import A, frame_bytes, gpu, header, launch, member, request, step, write_segment

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("REQSCOPE_BIN", ROOT / "build/reqscope"))


def span(identity="x", begin="1000000000000", end="1000000000100"):
    return {"name": "llm_request", "traceId": "1" * 32, "spanId": "2" * 16,
            "startTimeUnixNano": begin, "endTimeUnixNano": end,
            "attributes": [{"key": "gen_ai.request.id", "value": {"stringValue": identity}},
                           {"key": "gen_ai.usage.prompt_tokens", "value": {"intValue": "16"}},
                           {"key": "gen_ai.latency.time_in_queue", "value": {"doubleValue": 0.00000001}}]}


def envelope(spans):
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


def client_record():
    return {"clock": "CLOCK_MONOTONIC", "boot_id": A.split(":")[0],
            "engine_request_id": "x", "response_id": "cmpl-x", "http_status": 200,
            "request_start_monotonic_ns": 0, "request_end_monotonic_ns": 120,
            "usage": {"completion_tokens": 3},
            "chunks": [{"arrival_monotonic_ns": 80, "data": {"id": "cmpl-x",
                        "choices": [{"index": 0, "text": "one", "token_ids": [10]}]}},
                       {"arrival_monotonic_ns": 110, "data": {"id": "cmpl-x",
                        "choices": [{"index": 0, "text": "two three", "token_ids": [11, 12]}]}}]}


@unittest.skipUnless(BIN.is_file(), "build reqscope first")
class OTelImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rqs-otel-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = write_segment(self.root / "source.rqs", [step(1, 10, 30, [member("x")]),
                                                              *launch(1, 1, 15), gpu(1, 20, 25)])
        self.input = self.root / "otel.json"
        self.output = self.root / "imported"

    def call(self, *extra):
        return subprocess.run([str(BIN), "import-otel", str(self.source), "--input", str(self.input),
                               "--output", str(self.output), "--producer", A,
                               "--unix-minus-monotonic-ns", "1000000000000", "--clock-error-ns", "1", *extra],
                              text=True, capture_output=True, timeout=10)

    def test_explicit_offset_and_queue_import_without_changing_source(self):
        original = self.source.read_bytes()
        self.input.write_text(json.dumps(envelope([span()])))
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.source.read_bytes(), original)
        rows = [r for r in read_run(self.output).records if r["type"] == "Request"]
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["start"], rows[0]["end"], rows[0]["queued_until"]), (0, 100, 10))
        self.assertEqual(rows[0]["source"], "otel-request-span")
        analyzed = subprocess.run([str(BIN), "analyze", str(self.output)], text=True, capture_output=True, timeout=10)
        self.assertEqual(analyzed.returncode, 0, analyzed.stderr)
        request_result = json.loads(analyzed.stdout)["requests"][0]
        self.assertEqual(request_result["e2e_ns"], 100)
        self.assertEqual(request_result["components_ns"]["gpu_exec"], 5)
        self.assertEqual(request_result["components_ns"]["not_scheduled"], 10)

    def test_json_lines_spans_are_supported(self):
        self.input.write_text(json.dumps(envelope([span("x")])) + "\n" + json.dumps(envelope([span("y")])) + "\n")
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["requests_imported"], 2)

    def test_duplicate_request_span_is_rejected_before_output(self):
        self.input.write_text(json.dumps(envelope([span("x"), span("x")])))
        result = self.call()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate", result.stderr.lower())
        self.assertFalse(self.output.exists())

    def test_invalid_clock_and_duration_inputs_do_not_create_output(self):
        for item in (span(begin="999999999999"), span(begin="1000000000200"), span(end="not-a-number")):
            with self.subTest(span=item):
                self.input.write_text(json.dumps(envelope([item])))
                result = self.call()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("preserve")
        self.input.write_text(json.dumps(envelope([span()])))
        result = self.call()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "preserve")

    def test_import_preserves_finalization_and_cannot_close_killed_capture(self):
        base_records = read_run(self.source).records
        self.input.write_text(json.dumps(envelope([span()])))
        for closed in (False, True):
            with self.subTest(source_finalized=closed):
                data = b"RQSSEG01" + frame_bytes({**header(), "session_state": "Capturing"})
                data += b"".join(frame_bytes(row) for row in base_records)
                if closed:
                    data += frame_bytes({"type": "Session", "producer": A, "seq": len(base_records) + 1,
                                         "state": "Finalized"})
                self.source.write_bytes(data)
                self.output = self.root / f"imported-{closed}"
                result = self.call()
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run([str(BIN), "analyze", str(self.output)], text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["status"].lower() == "partial", not closed, report)
                boundaries = [r for r in read_run(self.output).records if r["type"] == "Session" and r["state"] == "Snapshot"]
                self.assertEqual(bool(boundaries), closed)

    def test_client_token_identity_preserves_chunk_arrivals_and_both_spans(self):
        self.input.write_text(json.dumps(envelope([span(begin="1000000000005")])) )
        client = self.root / "client.json"
        client.write_text(json.dumps(client_record()))
        result = self.call("--client-log", str(client))
        self.assertEqual(result.returncode, 0, result.stderr)
        row = next(r for r in read_run(self.output).records if r["type"] == "Request")
        self.assertEqual((row["start"], row["end"], row["otel_start"], row["otel_end"]), (0, 120, 5, 100))
        self.assertEqual(row["token_times"], [80, 110, 110])
        self.assertEqual(row["token_ids"], [10, 11, 12])
        self.assertEqual(row["not_scheduled_intervals"], [[5, 15]])
        self.assertNotIn("queued_until", row)

    def test_client_identity_clock_and_token_count_errors_reject_before_output(self):
        self.input.write_text(json.dumps(envelope([span()])))
        client = self.root / "client.json"
        mutations = [lambda r: r.update(boot_id="another-boot"),
                     lambda r: r.update(engine_request_id="unmatched"),
                     lambda r: r.update(request_end_monotonic_ns=90),
                     lambda r: r["usage"].update(completion_tokens=4),
                     lambda r: r["chunks"][1].update(arrival_monotonic_ns=70),
                     lambda r: r["chunks"][0]["data"].update(id="mixed-id")]
        for mutate in mutations:
            row = client_record()
            mutate(row)
            client.write_text(json.dumps(row))
            result = self.call("--client-log", str(client))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.output.exists())

    def test_client_text_without_token_ids_never_implies_token_boundaries(self):
        self.input.write_text(json.dumps(envelope([span()])))
        row = client_record()
        del row["chunks"][0]["data"]["choices"][0]["token_ids"]
        client = self.root / "client.json"
        client.write_text(json.dumps(row))
        result = self.call("--client-log", str(client))
        self.assertEqual(result.returncode, 0, result.stderr)
        row = next(r for r in read_run(self.output).records if r["type"] == "Request")
        self.assertFalse(row["client_token_times_available"])
        self.assertNotIn("token_times", row)

    def test_explicit_internal_identity_is_used_without_guessing_suffixes(self):
        item = span()
        item["attributes"].append({"key": "reqscope.internal_request_id", "value": {"stringValue": "x-ab12cd34"}})
        self.input.write_text(json.dumps(envelope([item])))
        client = self.root / "client.json"
        client.write_text(json.dumps(client_record()))
        result = self.call("--client-log", str(client))
        self.assertEqual(result.returncode, 0, result.stderr)
        row = next(r for r in read_run(self.output).records if r["type"] == "Request")
        self.assertEqual(row["req_id"], "x-ab12cd34")
        self.assertEqual(row["external_request_id"], "x")
        self.assertEqual(row["identity_source"], "otel-explicit-internal-request-id")
        self.assertEqual(row["token_times"], [80, 110, 110])


if __name__ == "__main__":
    unittest.main()
