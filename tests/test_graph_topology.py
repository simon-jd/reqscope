"""Literal executable graph topology checks through the public CLI only."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from fixtures.fake_engine.generate import A, frame_bytes, header, member, request, step

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("REQSCOPE_BIN", ROOT / "build/reqscope"))


def resource(action, timestamp, **extra):
    return {"type": "GraphResource", "action": action, "timestamp": timestamp,
            "graph_id": 2, "graph_exec_id": 0, "node_id": 0, "dependency_node_id": 0,
            "ids_valid": True, "ctx": 0, "tid": 43, **extra}


def recording_rows():
    rows = [resource("created", 10), resource("node_created", 11, node_id=101),
            resource("dependency_created", 12, dependency_node_id=101, node_id=102),
            resource("node_created", 13, node_id=102),
            resource("exec_created", 20, graph_id=1, graph_exec_id=2, ctx=1), request("x", 0, 200)]
    for corr, start in ((1, 40), (2, 120)):
        rows += [step(corr, start-5, start+35, [member("x")]),
                 dict(type="ExtCorr", corr=corr, step_id=corr),
                 dict(type="ApiCall", corr=corr, name="cudaGraphLaunch", start=start, end=start+5, tid=43),
                 dict(type="GpuActivity", corr=corr, start=start+10, end=start+20, kind="kernel",
                      ctx=1, stream=13, graph_id=2, graph_node_id=101),
                 dict(type="GpuActivity", corr=corr, start=start+20, end=start+30, kind="kernel",
                      ctx=1, stream=13, graph_id=2, graph_node_id=102)]
    return rows


@unittest.skipUnless(BIN.is_file(), "build reqscope first")
class GraphTopologyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rqs-graph-topology-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "recording.rqs"

    def report(self, rows, capture=True):
        self.path.write_bytes(b"RQSSEG01" + frame_bytes({**header(), "graph_resource_callbacks": capture}) +
                              b"".join(frame_bytes({"producer": A, "seq": i, **row}) for i, row in enumerate(rows, 1)))
        result = subprocess.run([str(BIN), "analyze", str(self.path)], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def edges(self, report):
        return [e for e in report["edges"] if e["source"] == "cupti-resource"]

    def test_direct_executable_topology_is_per_replay_with_full_evidence(self):
        report = self.report(recording_rows())
        self.assertEqual([(e["from"], e["to"]) for e in self.edges(report)],
                         [(f"{A}:10", f"{A}:11"), (f"{A}:15", f"{A}:16")])
        for e in self.edges(report):
            self.assertEqual((e["from_graph_node_id"], e["to_graph_node_id"]), (101, 102))
            self.assertEqual(e["relation"], "graph-dependency")
            self.assertFalse(e["gpu_timestamps_used"])
            self.assertTrue({f"{A}:{s}" for s in (1, 2, 3, 4, 5)} <= set(e["evidence_record_ids"]))
        self.assertTrue(all(g["status"] == "Complete" for g in report["graph_topology"]))

    def test_parameter_update_invalidates_only_subsequent_replay_of_target_exec(self):
        rows = recording_rows() + [resource("node_set_params", 90, graph_id=0, graph_exec_id=2, node_id=101)]
        report = self.report(rows)
        self.assertEqual(len(self.edges(report)), 1)
        self.assertEqual(report["graph_topology"][1]["reason"], "unsupported_graph_update")
        rows[-1]["graph_exec_id"] = 99
        self.assertEqual(len(self.edges(self.report(rows))), 2)

    def test_graph_exec_update_without_target_identity_invalidates_conservatively(self):
        rows = recording_rows() + [dict(type="ApiCall", corr=99, name="cudaGraphExecUpdate",
                                       tid=43, start=90, end=95)]
        report = self.report(rows)
        self.assertEqual(len(self.edges(report)), 1)
        self.assertEqual(report["graph_topology"][1]["reason"], "unsupported_graph_update")

    def test_source_template_changes_never_replace_executable_edges(self):
        rows = recording_rows() + [resource("dependency_created", 90, graph_id=1, dependency_node_id=102, node_id=101)]
        self.assertEqual([(e["from_graph_node_id"], e["to_graph_node_id"]) for e in self.edges(self.report(rows))],
                         [(101, 102), (101, 102)])

    def test_missing_invalid_or_conflicting_identity_never_yields_edges(self):
        for change in ("missing_ready", "missing_node", "wrong_exec", "wrong_context", "ids_invalid",
                       "duplicate_ready", "duplicate_node", "duplicate_dependency", "no_capture_contract"):
            rows = recording_rows()
            if change == "missing_ready": rows.pop(4)
            if change == "missing_node": rows.pop(3)
            if change == "wrong_exec": rows[4]["graph_exec_id"] = 3
            if change == "wrong_context": rows[4]["ctx"] = 2
            if change == "ids_invalid": rows[2]["ids_valid"] = False
            if change == "duplicate_ready": rows.append({**rows[4], "timestamp": 21})
            if change == "duplicate_node": rows.append({**rows[3], "timestamp": 14})
            if change == "duplicate_dependency": rows.append({**rows[2], "timestamp": 14})
            with self.subTest(change=change):
                report = self.report(rows, change != "no_capture_contract")
                self.assertEqual(self.edges(report), [])
                self.assertTrue(all(g["status"] == "Partial" for g in report["graph_topology"]))

    def test_loss_during_lifecycle_prevents_topology_projection(self):
        rows = recording_rows() + [dict(type="Loss", reason="graph_metadata_error", count=1, time_range=[10, 15])]
        report = self.report(rows)
        self.assertEqual(self.edges(report), [])
        self.assertTrue(all(g["reason"] == "loss_during_graph_lifecycle" for g in report["graph_topology"]))

    def test_repeated_node_activity_without_iteration_identity_is_unknown(self):
        rows = recording_rows()
        rows.append({**rows[9], "start": 75, "end": 80})
        report = self.report(rows)
        self.assertEqual(report["graph_topology"][0]["reason"], "repeated_graph_node_without_iteration_identity")
        self.assertEqual(len(self.edges(report)), 1)
        self.assertEqual(self.edges(report)[0]["corr"], 2)

    def test_dependency_removal_and_reversal_before_instantiation_uses_direct_edges(self):
        rows = recording_rows() + [resource("dependency_destroy", 14, dependency_node_id=101, node_id=102),
                                   resource("dependency_created", 15, dependency_node_id=102, node_id=101)]
        self.assertEqual([(e["from_graph_node_id"], e["to_graph_node_id"]) for e in self.edges(self.report(rows))],
                         [(102, 101), (102, 101)])

    def test_cycles_post_instantiation_changes_and_racing_destroy_remain_unknown(self):
        for extra in (resource("dependency_created", 15, dependency_node_id=102, node_id=101),
                      resource("dependency_destroy", 30, dependency_node_id=101, node_id=102),
                      resource("exec_destroy", 42, graph_id=0, graph_exec_id=2)):
            with self.subTest(action=extra):
                report = self.report(recording_rows() + [extra])
                self.assertEqual(self.edges(report), [])


if __name__ == "__main__":
    unittest.main()
