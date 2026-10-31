import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from checker.raw import MAX_FRAME_BYTES, read_run
from checker.oracle import expected_membership, interval_union, union_duration
from fixtures.fake_engine.generate import A, B, TRUTH, frame_bytes, generate, header, write_segment


class IndependentReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="reqscope-reader-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def raw(self, suffix):
        path = self.root / "bad.rqs"
        path.write_bytes(b"RQSSEG01" + frame_bytes(header()) + suffix)
        return read_run(path)

    def test_handwritten_replay_truth(self):
        run = read_run(generate(self.root / "normal"))
        self.assertFalse(run.issues, run.issues)
        activities = expected_membership(run)
        answer = {(a["producer"], a["seq"]): a["request_ids"] for a in activities}
        for row in TRUTH["activities"]:
            self.assertEqual(answer[(row["producer"], row["seq"])], row["request_ids"])
        self.assertEqual(len(run.headers), 2)
        self.assertEqual(run.headers[A]["pid"], run.headers[B]["pid"])

    def test_interval_union_not_sum(self):
        self.assertEqual(interval_union([(150, 230), (130, 210), (230, 240), (500, 510)]),
                         [(130, 240), (500, 510)])
        self.assertEqual(union_duration([(130, 210), (150, 230)]), 100)
        self.assertEqual(union_duration([(1, 1), (2, 1)]), 0)

    def test_truncation_at_every_byte_never_invents_records(self):
        row = dict(type="Request", producer=A, seq=1, req_id="unicode-请求", start=1, end=2)
        data = frame_bytes(row)
        for boundary in range(1, len(data)):
            with self.subTest(boundary=boundary):
                run = self.raw(data[:boundary])
                self.assertTrue(run.partial)
                self.assertEqual(run.records, [])
        self.assertEqual(len(self.raw(data).records), 1)

    def test_bad_crc_never_enters_oracle(self):
        data = bytearray(frame_bytes(dict(type="Request", producer=A, seq=1, req_id="bad", start=1, end=2)))
        data[-2] ^= 1
        run = self.raw(data)
        self.assertTrue(any("CRC" in issue for issue in run.issues))
        self.assertEqual(run.records, [])

    def test_length_bombs_stop_before_allocating_payload(self):
        for length in (0, MAX_FRAME_BYTES + 1, 0xFFFFFFFF):
            run = self.raw(struct.pack("<II", length, 0))
            self.assertTrue(any("length" in issue for issue in run.issues))

    def test_duplicate_keys_nan_and_bad_utf8(self):
        for data in (b'{"seq":1,"seq":2}', b'{"x":NaN}', b'{"x":"\xff"}', b'{"x":"\\ud800"}', b'[]'):
            suffix = struct.pack("<II", len(data), zlib.crc32(data)) + data
            self.assertTrue(self.raw(suffix).partial)

    def test_duplicate_and_missing_sequence(self):
        for seqs in ((1, 1), (1, 3), (2, 3)):
            rows = [dict(type="Request", producer=A, seq=n, req_id=str(n), start=1, end=2) for n in seqs]
            run = self.raw(b"".join(frame_bytes(row) for row in rows))
            self.assertTrue(any("sequence" in issue for issue in run.issues))

    def test_sequence_order_is_global_across_segment_files(self):
        row = dict(type="Request", req_id="x", start=1, end=2)
        write_segment(self.root / "z.rqs", [{**row, "seq": 1}])
        write_segment(self.root / "a.rqs", [{**row, "seq": 2}])
        self.assertFalse(read_run(self.root).partial)

    def test_foreign_producer_not_accepted(self):
        run = self.raw(frame_bytes(dict(type="Request", producer=B, seq=1, req_id="x", start=0, end=10)))
        self.assertTrue(run.partial)
        self.assertEqual(run.records, [])

    def test_budget_applies_before_read(self):
        path = write_segment(self.root / "budget.rqs", [])
        run = read_run(path, max_run_bytes=8)
        self.assertTrue(any("byte limit" in issue for issue in run.issues))

    def test_empty_magic_and_invalid_time_are_partial(self):
        path = self.root / "empty.rqs"
        path.write_bytes(b"RQSSEG01")
        self.assertTrue(read_run(path).partial)
        path.write_bytes(b"RQSSEG01" + frame_bytes({**header(), "clock_error_ns": None}))
        self.assertFalse(read_run(path).partial)  # Null is unknown, not malformed.
        for begin, end in ((-1, 10), (10, 1), (True, 10), ("1", 10)):
            run = self.raw(frame_bytes(dict(type="Request", producer=A, seq=1, req_id="x", start=begin, end=end)))
            self.assertTrue(run.partial)
            self.assertEqual(run.records, [])

    def test_explicit_loss_marks_reader_partial(self):
        run = read_run(generate(self.root / "loss", "loss"))
        self.assertTrue(run.partial)
        self.assertTrue(any("explicit loss" in issue for issue in run.issues))

    def test_invalid_record_shape_never_reaches_membership_oracle(self):
        for row in ({"type": "GpuActivity", "start": 0, "end": 1},
                    {"type": "Step", "step_id": 1, "members": "wrong"},
                    {"type": "Step", "step_id": 1, "members": [{"req_id": "x", "tokens": True}]},
                    {"type": "Request", "start": 0, "end": 1}, {}):
            run = self.raw(frame_bytes({"producer": A, "seq": 1, **row}))
            self.assertTrue(run.partial)
            self.assertEqual(expected_membership(run), [])


if __name__ == "__main__":
    unittest.main()
