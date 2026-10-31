import copy
import unittest
from checker.raw import Recording
from checker.oracle import check_graph_resource_edges


class GraphResourceOracleTests(unittest.TestCase):
    def fixture(self):
        def r(seq, kind, **values): return {"seq": seq, "producer": "b:1:2", "type": kind, **values}
        records = [r(1, "GraphResource", action="created", graph_id=7, timestamp=10, ids_valid=True),
                   r(2, "GraphResource", action="node_created", graph_id=7, node_id=24, timestamp=11, ids_valid=True),
                   r(3, "GraphResource", action="node_created", graph_id=7, node_id=25, timestamp=12, ids_valid=True),
                   r(6, "GraphResource", action="dependency_created", graph_id=7, dependency_node_id=24, node_id=25, timestamp=30, ids_valid=True),
                   r(8, "GraphResource", action="exec_created", graph_exec_id=7, ctx=1, timestamp=40, ids_valid=True),
                   r(20, "ApiCall", corr=9, start=100, end=110, return_value=0),
                   r(31, "GpuActivity", graph_id=7, graph_node_id=24, corr=9, ctx=1, start=900, end=950),
                   r(32, "GpuActivity", graph_id=7, graph_node_id=25, corr=9, ctx=1, start=800, end=850)]
        recording = Recording(headers={"b:1:2": {"graph_resource_callbacks": True}}, records=records)
        edge = {"from": "b:1:2:31", "to": "b:1:2:32", "source": "cupti-resource", "relation": "graph-dependency",
                "gpu_timestamps_used": False, "evidence_record_ids": [f"b:1:2:{n}" for n in (1, 2, 3, 6, 8, 20, 31, 32)]}
        return recording, edge

    def test_explicit_raw_dependency_does_not_depend_on_gpu_timestamp_order(self):
        raw, edge = self.fixture()
        self.assertEqual(check_graph_resource_edges(raw, [edge]), [])

    def test_missing_dependency_reverse_edge_and_foreign_record_are_rejected(self):
        for mutation in ("missing", "reverse", "foreign"):
            raw, edge = self.fixture()
            if mutation == "missing": edge["evidence_record_ids"].remove("b:1:2:6")
            elif mutation == "reverse": edge["from"], edge["to"] = edge["to"], edge["from"]
            else: raw.records[-1]["producer"] = "b:2:2"
            self.assertTrue(check_graph_resource_edges(raw, [edge]), mutation)

    def test_repeated_conditional_node_and_destroyed_exec_are_rejected(self):
        raw, edge = self.fixture()
        repeated = copy.deepcopy(raw.records[-1]); repeated["seq"] = 33
        raw.records.append(repeated)
        self.assertTrue(any("iteration" in e for e in check_graph_resource_edges(raw, [edge])))
        raw, edge = self.fixture()
        raw.records.append({"type": "GraphResource", "producer": "b:1:2", "seq": 34, "action": "exec_destroy", "graph_exec_id": 7, "timestamp": 90})
        self.assertTrue(any("destroyed" in e for e in check_graph_resource_edges(raw, [edge])))
