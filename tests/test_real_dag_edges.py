"""Handwritten edge oracle for checked-in real CUDA recordings.

No analyzer implementation is imported. Raw identities come from the independent
wire reader; expected edges are literal fixture truth, never timestamp-derived.
Run directly with --output to save the report as JSON.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import unittest

from checker.raw import read_run

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("REQSCOPE_BIN", ROOT / "build" / "reqscope"))
CASES = {
    "streams": {
        "path": "fixtures/recordings/gpu-streams-system-tid",
        "producer_suffix": ":22452:61248119",
        "nodes": {134: "GpuActivity", 139: "CudaEvent", 142: "Sync", 149: "GpuActivity"},
        "expected": [(134, 139, "stream-order"), (139, 142, "sync-record"), (142, 149, "stream-order")],
        "unresolved": [],
    },
    "copies": {
        "path": "fixtures/recordings/gpu-copies-trace",
        "producer_suffix": ":4175408:61230004",
        "nodes": {127: "GpuActivity", 136: "GpuActivity", 141: "GpuActivity"},
        "expected": [(127, 136, "stream-order"), (136, 141, "stream-order")],
        "unresolved": [],
    },
    "graph_replay": {
        "path": "fixtures/recordings/gpu-graph-trace",
        "producer_suffix": ":4176844:61230438",
        "nodes": {144: "GpuActivity", 145: "GpuActivity", 151: "GpuActivity", 152: "GpuActivity",
                  158: "GpuActivity", 159: "GpuActivity"},
        "expected": [],
        "unresolved": [(144, 145), (151, 152), (158, 159)],
    },
    "graph_replay_resources": {
        "path": "fixtures/recordings/graph-resource-probe",
        "producer_suffix": ":423756:61462190",
        "nodes": {167: "GpuActivity", 168: "GpuActivity", 174: "GpuActivity", 175: "GpuActivity",
                  181: "GpuActivity", 182: "GpuActivity"},
        "expected": [(167, 168, "cupti-resource"), (174, 175, "cupti-resource"), (181, 182, "cupti-resource")],
        "unresolved": [],
    },
    "conditional_resources": {
        "path": "fixtures/recordings/graph-resource-conditional-probe",
        "producer_suffix": ":431731:61465637",
        "nodes": {202: "GpuActivity", 203: "GpuActivity", 209: "GpuActivity", 210: "GpuActivity",
                  216: "GpuActivity", 217: "GpuActivity", 246: "GpuActivity", 247: "GpuActivity", 248: "GpuActivity"},
        "expected": [(202, 203, "cupti-resource"), (209, 210, "cupti-resource"), (216, 217, "cupti-resource")],
        "unresolved": [(246, 247), (247, 248)],
    },
    "mutation_resources": {
        "path": "fixtures/recordings/graph-resource-mutation-probe",
        "producer_suffix": ":463357:61477659",
        "nodes": {186: "GpuActivity", 187: "GpuActivity", 198: "GpuActivity", 199: "GpuActivity",
                  215: "GpuActivity", 216: "GpuActivity"},
        "expected": [(186, 187, "cupti-resource"), (215, 216, "cupti-resource")],
        "unresolved": [(198, 199)],
    },
}


def inspect_case(name):
    case = CASES[name]
    path = ROOT / case["path"]
    recording = read_run(path)
    errors = list(recording.issues)
    producers = [p for p in recording.headers if p.endswith(case["producer_suffix"])]
    if len(producers) != 1:
        raise AssertionError(f"{name}: expected unique fixture producer")
    producer = producers[0]
    records = {r["seq"]: r for r in recording.records if r["producer"] == producer}
    for seq, expected_type in case["nodes"].items():
        if records.get(seq, {}).get("type") != expected_type:
            errors.append(f"raw node {seq} has unexpected identity or type")
    completed = subprocess.run([str(BIN), "analyze", str(path)], capture_output=True, text=True, timeout=20)
    if completed.returncode:
        raise AssertionError(completed.stderr)
    report = json.loads(completed.stdout)
    ids = {f"{producer}:{seq}" for seq in case["nodes"]}
    observed = sorted((e["from"], e["to"], e["source"]) for e in report["edges"]
                      if e["from"] in ids and e["to"] in ids)
    wanted = sorted((f"{producer}:{a}", f"{producer}:{b}", source) for a, b, source in case["expected"])
    if observed != wanted:
        errors.append("projected operation edge set differs from handwritten justified edge set")
    edge_evidence = [e for e in report["edges"] if e["from"] in ids and e["to"] in ids]
    for e in edge_evidence:
        if e["source"] == "stream-order":
            if e.get("ordering_basis") != "serialized-host-api" or e.get("gpu_timestamps_used") is not False:
                errors.append("stream edge lacks explicit host-submission provenance")
            seqs = {int(i.rsplit(":", 1)[1]) for i in e["evidence_record_ids"]}
            apis = [records[s] for s in seqs if records[s]["type"] == "ApiCall"]
            if len(apis) != 2 or len({a["tid"] for a in apis}) != 1:
                errors.append("stream edge does not retain both same-thread API records")
        if e["source"] == "cupti-resource":
            seqs = {int(i.rsplit(":", 1)[1]) for i in e["evidence_record_ids"]}
            dependencies = [records[s] for s in seqs if records[s]["type"] == "GraphResource"
                            and records[s]["action"] == "dependency_created"]
            source = records[int(e["from"].rsplit(":", 1)[1])]
            target = records[int(e["to"].rsplit(":", 1)[1])]
            if len(dependencies) != 1 or dependencies[0]["dependency_node_id"] != source["graph_node_id"] or \
                    dependencies[0]["node_id"] != target["graph_node_id"] or source["corr"] != target["corr"]:
                errors.append("graph edge differs from literal raw executable dependency endpoints")
    return {
        "fixture": name, "input": case["path"], "producer": producer,
        "raw_reader_issues": list(recording.issues), "expected_proven_edges": wanted,
        "observed_edges": observed, "edge_evidence": edge_evidence,
        "unresolved_source_truth_edges": [[f"{producer}:{a}", f"{producer}:{b}"] for a, b in case["unresolved"]],
        "handwritten_operation_dag_complete": not case["unresolved"] and not errors,
        "analysis_status": report["status"], "errors": errors,
        "graph_topology": report.get("graph_topology", []),
        "scope": "operation DAG projection; API/member edges checked separately; no calibrated timing claim",
    }


@unittest.skipUnless(BIN.is_file(), "build reqscope first")
class RealDagEvidenceTests(unittest.TestCase):
    def test_real_fixture_operation_dag_edges(self):
        for name, case in CASES.items():
            with self.subTest(fixture=name):
                if not (ROOT / case["path"]).is_dir():
                    self.skipTest("local real CUDA evidence absent")
                result = inspect_case(name)
                self.assertEqual(result["errors"], [], result)
                self.assertEqual(result["handwritten_operation_dag_complete"], not case["unresolved"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [inspect_case(name) for name in CASES]
    result = {"ok": all(not c["errors"] for c in cases), "cases": cases,
              "static_fixture_dags_complete": all(c["handwritten_operation_dag_complete"] for c in cases
                                                   if c["fixture"] in ("streams", "copies", "graph_replay_resources")),
              "all_dags_complete": all(c["handwritten_operation_dag_complete"] for c in cases)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ok": result["ok"], "all_dags_complete": result["all_dags_complete"]}))
    raise SystemExit(0 if result["ok"] else 1)
