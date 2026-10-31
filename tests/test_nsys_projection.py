"""Small SQLite counterexamples, independent of the running GPU benchmark."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from checker.nsys_projection import GLOBAL_PID_MASK, ProjectionError, project_sqlite


class NsysProjectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nsys.sqlite"
        self.db = sqlite3.connect(self.path)
        self.addCleanup(self.db.close)
        self.db.executescript('''
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE ENUM_NSYS_EVENT_TYPE(id INTEGER PRIMARY KEY, name TEXT);
            INSERT INTO ENUM_NSYS_EVENT_TYPE VALUES (75,'NvtxDomainCreate'),
                (76,'NvtxDomainDestroy'), (59,'NvtxPushPopRange'), (60,'NvtxStartEndRange');
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,eventType INTEGER,text TEXT,
                globalTid INTEGER,endGlobalTid INTEGER,textId INTEGER,domainId INTEGER,
                jsonText TEXT,jsonTextId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,
                globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,
                globalPid INTEGER,
                correlationId INTEGER, -- Nsight export contract: REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
                demangledName INTEGER,
                deviceId INTEGER,contextId INTEGER,streamId INTEGER,graphId INTEGER,graphNodeId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,
                globalPid INTEGER,correlationId INTEGER,bytes INTEGER);
            INSERT INTO StringIds VALUES(1,'cudaGraphLaunch'),(2,'known_kernel');
        ''')
        self.pid = 123 << 24
        self.tid = self.pid | 456
        self.domain()

    def domain(self, name="reqscope", pid=None, domain_id=2, start=-100):
        self.db.execute("INSERT INTO NVTX_EVENTS(start,eventType,text,globalTid,domainId) VALUES(?,75,?,?,?)",
                        (start, name, (pid or self.pid) | 1, domain_id))

    def nvtx(self, start=10, end=50, step=1, members=("cmpl-exact-0-abcd1234",),
             tid=None, phase="execute_model", payload=None, domain=2, end_tid=None, registered=False):
        if payload is None:
            payload = json.dumps({"step_id": step, "phase": phase,
                                  "members": [{"req_id": x, "tokens": 1, "phase": "decode"} for x in members]})
        text_id = None
        if registered:
            text_id = self.db.execute("SELECT COALESCE(MAX(id),0)+1 FROM StringIds").fetchone()[0]
            self.db.execute("INSERT INTO StringIds VALUES(?,?)", (text_id, payload))
        return self.db.execute('''INSERT INTO NVTX_EVENTS(start,end,eventType,text,globalTid,
            endGlobalTid,textId,domainId) VALUES(?,?,59,?,?,?,?,?)''',
            (start, end, None if registered else payload, tid or self.tid, end_tid, text_id, domain)).lastrowid

    def api(self, corr=7, start=20, end=30, tid=None):
        return self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,1)",
                               (start, end, tid or self.tid, corr)).lastrowid

    def kernel(self, corr=7, start=100, end=140, pid=None):
        return self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,2,0,1,7,99,1001)",
                               (start, end, self.pid if pid is None else pid, corr)).lastrowid

    def report(self, **kwargs):
        self.db.commit()
        return project_sqlite(self.path, **kwargs)

    def test_exact_ids_and_outliving_gpu_are_linked_with_full_row_provenance(self):
        nv = self.nvtx(members=("cmpl-exact-0-abcd1234", "request-B"), registered=True)
        api = self.api()
        gpu = self.kernel(start=1000, end=1040)
        self.kernel(corr=999)
        report = self.report()
        linked = report["gpu_activities"][0]
        self.assertEqual(linked["status"], "linked")
        self.assertEqual(linked["rowid"], gpu)
        self.assertEqual(linked["runtime_rowids"], [api])
        self.assertEqual(linked["nvtx_rowids"], [nv])
        self.assertEqual(linked["request_ids"], ["cmpl-exact-0-abcd1234", "request-B"])
        self.assertEqual(linked["membership_sources"][0]["json_path"], "$.members[0].req_id")
        self.assertEqual(linked["membership_sources"][0]["payload_field"], "textId")
        self.assertFalse(report["method"]["gpu_time_used_for_identity"])
        self.assertEqual(report["gpu_summary"]["denominator"], 2)
        self.assertEqual(report["gpu_summary"]["linked_fraction"], .5)
        self.assertEqual(report["gpu_summary"]["unknown"], 1)

    def test_replayed_graph_membership_changes_per_runtime_and_range(self):
        self.nvtx(step=1, members=("A",))
        self.nvtx(start=60, end=90, step=2, members=("B",), phase="sample_tokens")
        self.api(corr=7)
        self.api(corr=8, start=65, end=70)
        self.kernel(corr=7, start=100, end=120)
        self.kernel(corr=8, start=121, end=141)
        activities = self.report()["gpu_activities"]
        self.assertEqual([x["request_ids"] for x in activities], [["A"], ["B"]])
        self.assertEqual([x["step_id"] for x in activities], [1, 2])
        self.assertEqual({x["graph_id"] for x in activities}, {99})

    def test_identical_request_ids_in_other_process_are_separate(self):
        other_pid = 124 << 24
        other_tid = other_pid | 456
        self.domain(pid=other_pid)
        self.nvtx(members=("same-id",))
        self.nvtx(members=("same-id",), tid=other_tid)
        self.api()
        self.api(tid=other_tid)
        self.kernel()
        self.kernel(pid=other_pid)
        requests = self.report()["requests"]
        self.assertEqual([(r["global_pid"], r["request_id"]) for r in requests],
                         [(self.pid, "same-id"), (other_pid, "same-id")])

    def test_same_step_two_phases_require_same_membership(self):
        self.nvtx(members=("A",))
        self.nvtx(start=60, end=90, members=("B",), phase="sample_tokens")
        self.api(corr=7)
        self.api(corr=8, start=65, end=70)
        self.kernel(corr=7)
        self.kernel(corr=8)
        report = self.report()
        self.assertEqual(report["gpu_summary"]["linked"], 0)
        self.assertTrue(all("conflicting_step_membership" in r["reasons"] for r in report["nvtx_ranges"]))

    def test_same_step_duplicate_disjoint_phase_is_invalid(self):
        self.nvtx()
        self.nvtx(start=60, end=90)
        self.api()
        self.kernel()
        report = self.report()
        self.assertEqual(report["gpu_summary"]["unknown"], 1)
        self.assertTrue(all("duplicate_step_phase" in r["reasons"] for r in report["nvtx_ranges"]))

    def test_same_step_distinct_phases_consistent_members_are_allowed(self):
        self.nvtx(members=("A", "B"))
        self.nvtx(start=60, end=90, members=("B", "A"), phase="sample_tokens")
        self.api(corr=7)
        self.api(corr=8, start=65, end=70)
        self.kernel(corr=7)
        self.kernel(corr=8)
        self.assertEqual(self.report()["gpu_summary"]["linked"], 2)

    def test_same_corr_other_process_and_other_thread_do_not_link(self):
        self.nvtx()
        self.api(corr=7, tid=(124 << 24) | 456)
        self.api(corr=8, tid=self.pid | 999)
        self.kernel(corr=7)
        self.kernel(corr=8)
        report = self.report()
        self.assertEqual(report["gpu_summary"]["unknown"], 2)
        self.assertTrue(all(not x["request_ids"] for x in report["gpu_activities"]))
        self.assertEqual(self.tid & GLOBAL_PID_MASK, self.pid)

    def test_duplicate_runtime_corr_is_ambiguous_without_gpu_time_tiebreak(self):
        self.nvtx()
        a = self.api()
        b = self.api(start=200, end=210)
        self.kernel(start=31, end=32)
        row = self.report()["gpu_activities"][0]
        self.assertEqual(row["status"], "ambiguous")
        self.assertEqual(row["runtime_rowids"], [a, b])
        self.assertFalse(row["request_ids"])

    def test_driver_numeric_collision_is_not_the_runtime_association_namespace(self):
        self.db.execute('''CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,
            globalTid INTEGER,correlationId INTEGER,nameId INTEGER)''')
        self.nvtx()
        runtime_row = self.api(corr=7)
        self.db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(20,30,?,?,1)",
                            [(self.tid, 7), (self.tid, 8)])
        self.kernel(corr=7)
        self.kernel(corr=8)
        report = self.report()
        self.assertEqual(report["gpu_activities"][0]["status"], "linked")
        self.assertEqual(report["gpu_activities"][0]["runtime_rowids"], [runtime_row])
        self.assertEqual(report["gpu_activities"][1]["status"], "unknown")
        self.assertEqual(report["gpu_activities"][1]["runtime_rowids"], [])
        self.assertIn("not an enforced SQLite FK", report["method"]["correlation_namespace"])

    def test_nested_nvtx_ranges_are_ambiguous_even_with_identical_members(self):
        a = self.nvtx(start=0, end=100)
        b = self.nvtx(start=10, end=50)
        self.api()
        self.kernel()
        report = self.report()
        self.assertEqual(report["gpu_activities"][0]["status"], "ambiguous")
        self.assertEqual(report["gpu_activities"][0]["nvtx_rowids"], [a, b])
        self.assertEqual(report["gpu_summary"]["ambiguous"], 1)

    def test_api_partial_overlap_is_not_complete_containment(self):
        self.nvtx(start=20, end=50)
        self.api(start=19, end=30)
        self.kernel(start=21, end=22)
        self.assertEqual(self.report()["gpu_activities"][0]["status"], "unknown")

    def test_unclosed_cross_thread_and_malformed_ranges_remain_unknown(self):
        for options in ({"end": None}, {"end_tid": self.tid + 1}, {"payload": '{"step_id":1,"step_id":2}'},
                        {"members": ()}):
            with self.subTest(options=options):
                self.db.execute("DELETE FROM NVTX_EVENTS WHERE eventType=59")
                self.db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME")
                self.db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
                self.nvtx(**options)
                self.api()
                self.kernel()
                row = self.report()["gpu_activities"][0]
                self.assertEqual(row["status"], "unknown")
                self.assertFalse(row["request_ids"])

    def test_exact_domain_name_required_and_unknown_domain_is_not_guessed(self):
        self.domain(name="reqscope-copy", domain_id=3)
        self.nvtx(domain=3)
        self.api()
        self.kernel()
        report = self.report()
        self.assertEqual(report["nvtx_summary"]["reqscope_ranges"], 0)
        self.assertEqual(report["gpu_summary"]["unknown"], 1)

    def test_conflicting_domain_registration_is_not_resolved_by_name(self):
        self.domain(name="another-tool", domain_id=2)
        self.nvtx()
        self.api()
        self.kernel()
        report = self.report()
        self.assertEqual(report["nvtx_summary"]["invalid_reqscope_ranges"], 1)
        self.assertEqual(report["gpu_summary"]["unknown"], 1)

    def test_invalid_member_phase_and_overflow_float_do_not_link(self):
        for extra in ('"phase":"unsupported"', '"extra":1e999'):
            with self.subTest(extra=extra):
                self.db.execute("DELETE FROM NVTX_EVENTS WHERE eventType=59")
                self.db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME")
                self.db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
                payload = '{"step_id":1,"phase":"execute_model","members":[{"req_id":"A",' + extra + '} ]}'
                self.nvtx(payload=payload)
                self.api()
                self.kernel()
                report = self.report()
                self.assertEqual(report["gpu_summary"]["unknown"], 1)
                json.dumps(report, allow_nan=False)

    def test_string_table_blob_fails_explicitly(self):
        self.db.execute("UPDATE StringIds SET value=? WHERE id=2", (b"binary name",))
        with self.assertRaisesRegex(ProjectionError, "StringIds.*binary"):
            self.report()

    def test_invalid_range_end_remains_an_ambiguity_candidate(self):
        self.nvtx(step=1, members=("A",))
        self.nvtx(start=0, end="broken", step=2, members=("B",))
        self.api()
        self.kernel()
        report = self.report()
        self.assertEqual(report["gpu_activities"][0]["status"], "ambiguous")
        self.assertEqual(len(report["gpu_activities"][0]["nvtx_rowids"]), 2)

    def test_unlocatable_range_start_fails_explicitly(self):
        self.nvtx(start="broken")
        with self.assertRaisesRegex(ProjectionError, "integer start"):
            self.report()

    def test_memcpy_duration_and_kernel_distribution_have_distinct_denominators(self):
        self.nvtx()
        self.api()
        self.kernel(start=100, end=110)
        self.kernel(start=101, end=131)
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(150,155,?,?,4096)", (self.pid, 7))
        report = self.report()
        self.assertEqual(report["gpu_summary"]["denominator"], 3)
        kernel = report["gpu_by_kind"]["kernel"]
        self.assertEqual(kernel["denominator"], 2)
        self.assertEqual(kernel["duration_ns"]["sum_ns"], 40)
        self.assertEqual(kernel["duration_ns"]["p50_ns"], 20)
        self.assertEqual(report["gpu_by_kind"]["memcpy"]["duration_ns"]["sum_ns"], 5)

    def test_invalid_gpu_duration_does_not_change_identity_but_is_not_summed(self):
        self.nvtx()
        self.api()
        self.kernel(start=50, end=20)
        report = self.report()
        self.assertEqual(report["gpu_summary"]["linked"], 1)
        self.assertEqual(report["gpu_summary"]["duration_valid_denominator"], 0)
        self.assertIsNone(report["gpu_summary"]["duration_ns"]["p50_ns"])

    def test_input_budget_fails_instead_of_reporting_truncated_completeness(self):
        self.nvtx()
        self.api()
        self.kernel()
        with self.assertRaisesRegex(ProjectionError, "budget"):
            self.report(max_rows=1)

    def test_sqlite_length_limit_rejects_large_value_before_projection(self):
        self.db.execute("INSERT INTO StringIds VALUES(3,?)", ("a" * 5000,))
        with self.assertRaisesRegex(ProjectionError, "length limit"):
            self.report(max_value_bytes=1024)

    def test_cumulative_input_byte_budget_fails(self):
        with self.assertRaisesRegex(ProjectionError, "input byte budget"):
            self.report(max_bytes=16)

    def test_containment_scan_has_a_global_work_budget(self):
        for step in range(1, 21):
            self.nvtx(start=0, end=1000, step=step)
        self.api()
        self.kernel()
        with self.assertRaisesRegex(ProjectionError, "join work budget"):
            self.report(max_work=5)

    def test_member_projection_counts_toward_work_budget(self):
        self.nvtx(members=tuple(f"request-{i}" for i in range(20)))
        self.api()
        self.kernel()
        with self.assertRaisesRegex(ProjectionError, "join work budget"):
            self.report(max_work=30)

    def test_nested_nvtx_references_in_correlation_join_are_budgeted(self):
        for step in range(1, 11):
            self.nvtx(start=0, end=1000, step=step)
        for _ in range(5):
            self.api()
        self.kernel()
        with self.assertRaisesRegex(ProjectionError, "join work budget"):
            self.report(max_work=70)

    def test_repeated_registered_payload_parsing_is_budgeted(self):
        source = self.nvtx(members=("A" * 1000,), registered=True)
        for _ in range(3):
            self.db.execute("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE rowid=?", (source,))
        with self.assertRaisesRegex(ProjectionError, "association byte budget"):
            self.report(max_bytes=3000)

    def test_repeated_member_output_has_a_separate_byte_budget(self):
        self.nvtx(members=("A" * 1000,))
        self.api()
        self.kernel()
        self.kernel()
        with self.assertRaisesRegex(ProjectionError, "association byte budget"):
            self.report(max_bytes=6000)

    def test_export_metadata_is_preserved_without_forward_schema_claim(self):
        self.db.execute("CREATE TABLE EXPORT_META_DATA(name TEXT,value TEXT)")
        self.db.execute("INSERT INTO EXPORT_META_DATA VALUES('EXPORT_SCHEMA_VERSION','fixture-version')")
        source = self.report()["source"]
        self.assertEqual(source["export_metadata"]["EXPORT_META_DATA"][0]["value"], "fixture-version")
        self.assertIn("assumed", source["schema_contract_status"])

    def test_missing_required_schema_is_explicit(self):
        self.db.execute("DROP TABLE CUPTI_ACTIVITY_KIND_RUNTIME")
        with self.assertRaisesRegex(ProjectionError, "missing required tables"):
            self.report()

    def test_cli_refuses_to_overwrite_existing_evidence(self):
        self.db.commit()
        output = Path(self.directory.name) / "existing.json"
        output.write_text("original evidence")
        completed = subprocess.run([sys.executable, "-m", "checker.nsys_projection", str(self.path),
                                    "--output", str(output)], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(output.read_text(), "original evidence")


if __name__ == "__main__":
    unittest.main()
