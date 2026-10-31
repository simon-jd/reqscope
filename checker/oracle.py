"""Membership and conservation oracle, independent of the C++ analyzer.

Run with ``python3 -m checker.oracle RUN --analysis analysis.json``. Optional
fixture truth checks prevent a consistent error in both implementations from
passing as correct. Activity share is accumulated device activity time; request
latency uses wall-clock union and is deliberately a different quantity.
"""

import argparse
import json
import math
from pathlib import Path
from .raw import read_run


def interval_union(intervals):
    spans = sorted((int(a), int(b)) for a, b in intervals if b > a)
    merged = []
    for begin, end in spans:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    return merged


def union_duration(intervals):
    return sum(b - a for a, b in interval_union(intervals))


def subtract_intervals(intervals, excluded):
    result = interval_union(intervals)
    for left, right in interval_union(excluded):
        pieces = []
        for begin, end in result:
            if end <= left or begin >= right:
                pieces.append((begin, end))
            else:
                if begin < left:
                    pieces.append((begin, left))
                if end > right:
                    pieces.append((right, end))
        result = pieces
    return result


def estimate_matches(actual, expected, uncertainty_allowed=False):
    if actual is None and uncertainty_allowed:
        return True
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6))


def expected_membership(recording):
    steps, apis, correlations = {}, {}, {}
    for record in recording.records:
        producer = record["producer"]
        if record["type"] == "Step":
            steps.setdefault((producer, record["step_id"]), []).append(record)
        elif record["type"] == "ApiCall":
            apis.setdefault((producer, record["corr"]), []).append(record)
        elif record["type"] == "ExtCorr":
            correlations.setdefault((producer, record["corr"]), set()).add(record["step_id"])
    output = []
    for record in recording.records:
        if record["type"] != "GpuActivity":
            continue
        key = record["producer"], record["corr"]
        candidates = correlations.get(key, set())
        linked = len(candidates) == 1 and len(apis.get(key, [])) == 1
        selected = next(iter(candidates)) if linked else None
        step_records = steps.get((record["producer"], selected), [])
        linked = linked and len(step_records) == 1
        step = step_records[0] if linked else None
        members = {m["req_id"]: m["tokens"] for m in step["members"] if m["tokens"] > 0} if step else {}
        output.append({"producer": record["producer"], "seq": record["seq"], "corr": record["corr"],
                       "step_ids": [selected] if linked else [], "request_ids": sorted(members),
                       "members": members, "start": record["start"], "end": record["end"]})
    return output


def check_graph_resource_edges(recording, edges):
    """Audit explicit topology provenance without using GPU timing order."""
    indexed = {f"{r['producer']}:{r['seq']}": r for r in recording.records}
    errors = []
    for edge in edges:
        if edge.get("source") != "cupti-resource":
            continue
        prefix = f"graph edge {edge.get('from')} -> {edge.get('to')}"
        def reject(reason):
            errors.append(f"{prefix}: {reason}")
        source, target = indexed.get(edge.get("from")), indexed.get(edge.get("to"))
        if not source or not target or source.get("type") != "GpuActivity" or target.get("type") != "GpuActivity":
            reject("endpoints are not raw GPU records")
            continue
        key = (source["producer"], source.get("graph_id"), source.get("corr"), source.get("ctx"))
        if key != (target["producer"], target.get("graph_id"), target.get("corr"), target.get("ctx")) or not key[1]:
            reject("endpoints do not belong to the same explicit graph execution")
            continue
        if edge.get("relation") != "graph-dependency" or edge.get("gpu_timestamps_used") is not False:
            reject("missing explicit dependency semantics")
        if recording.headers[key[0]].get("graph_resource_callbacks") is not True:
            reject("header does not promise GraphResource capture")
        evidence_ids = edge.get("evidence_record_ids", [])
        if any(identity not in indexed for identity in evidence_ids):
            reject("evidence record is missing")
            continue
        evidence = [indexed[identity] for identity in evidence_ids]
        if any(r["producer"] != key[0] for r in evidence):
            reject("cross-producer evidence")
        if edge["from"] not in evidence_ids or edge["to"] not in evidence_ids:
            reject("endpoint evidence is missing")
        resources = [r for r in evidence if r["type"] == "GraphResource"]
        if any(r.get("ids_valid") is not True for r in resources):
            reject("invalid public CUPTI resource identity")
        actions = lambda action: [r for r in resources if r.get("action") == action]
        if not any(r.get("graph_id") == key[1] for r in actions("created")):
            reject("graph creation evidence is missing")
        ready = [r for r in actions("exec_created") if r.get("graph_exec_id") == key[1] and r.get("ctx") == key[3]]
        if len(ready) != 1:
            reject("executable lifecycle evidence is not unique")
        from_node, to_node = source.get("graph_node_id"), target.get("graph_node_id")
        if not from_node or not to_node or from_node == to_node:
            reject("node identities are missing or self-referential")
        for node in (from_node, to_node):
            if not any(r.get("graph_id") == key[1] and r.get("node_id") == node for r in actions("node_created")):
                reject("node creation evidence is missing")
            occurrences = [r for r in recording.records if r["type"] == "GpuActivity" and
                           (r["producer"], r.get("graph_id"), r.get("corr"), r.get("ctx")) == key and r.get("graph_node_id") == node]
            if len(occurrences) != 1:
                reject("repeated node has no unique iteration identity")
        dependencies = [r for r in actions("dependency_created") if r.get("graph_id") == key[1] and
                        r.get("dependency_node_id") == from_node and r.get("node_id") == to_node]
        if len(dependencies) != 1:
            reject("no unique raw dependency in this direction")
        apis = [r for r in evidence if r["type"] == "ApiCall" and r.get("corr") == key[2] and r.get("return_value", 0) == 0]
        if len(apis) != 1:
            reject("launch API evidence is not unique")
        elif ready:
            launch = apis[0]
            if not ready[0]["timestamp"] < launch["start"]:
                reject("graph executable not instantiated before launch")
            for r in recording.records:
                if r["type"] != "GraphResource" or r["producer"] != key[0] or r.get("timestamp", 0) > launch["end"]:
                    continue
                if r.get("action") == "exec_destroy" and r.get("graph_exec_id") == key[1]:
                    reject("executable destroyed before launch completed")
                if r.get("graph_id") == key[1] and r.get("action") in {"dependency_destroy", "destroy", "node_destroy"}:
                    reject("dependency lifetime changed before launch")
    return errors


def check(recording, analysis, truth=None):
    errors = []
    if recording.partial and str(analysis.get("status", "")).lower() != "partial":
        errors.append("raw corruption or loss was not propagated as Partial")
    expected = expected_membership(recording)
    actual_activities = analysis.get("activities", [])
    if len(actual_activities) != len(expected):
        errors.append(f"GPU count: expected {len(expected)}, found {len(actual_activities)}")
    # Record IDs are an output convention, not a shared implementation detail.
    indexed = {(r["producer"], str(r["record_id"])): r for r in actual_activities}
    used = set()
    for wanted in expected:
        producer, seq = wanted["producer"], wanted["seq"]
        matches = [r for (p, rid), r in indexed.items() if p == producer and
                   (rid == str(seq) or rid.endswith(f":{seq}") or rid.endswith(f"/{seq}"))]
        if len(matches) != 1:
            # Fields permit matching without making record-ID syntax contractual.
            matches = [r for r in actual_activities if r.get("producer") == producer and
                       r.get("corr") == wanted["corr"] and r.get("start_ns", r.get("start")) == wanted["start"]]
        if len(matches) != 1:
            errors.append(f"{producer} seq {seq}: no unique output activity")
            continue
        actual = matches[0]
        used.add(actual["record_id"])
        if sorted(actual.get("request_ids", [])) != wanted["request_ids"]:
            errors.append(f"{producer} seq {seq}: replay members disagree")
        if sorted(actual.get("step_ids", [])) != wanted["step_ids"]:
            errors.append(f"{producer} seq {seq}: step identity disagrees")
    request_records = {(r["producer"], r["req_id"]): r for r in recording.records if r["type"] == "Request"}
    actual_requests = {(r["producer"], r["request_id"]): r for r in analysis.get("requests", [])}
    for key, request in request_records.items():
        actual = actual_requests.get(key)
        if actual is None:
            errors.append(f"{key}: missing request")
            continue
        e2e = request["end"] - request["start"]
        components = actual.get("components_ns", {})
        if actual.get("e2e_ns") != e2e or sum(components.values()) != e2e:
            errors.append(f"{key}: E2E conservation failed")
        if any(not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0 for n in components.values()):
            errors.append(f"{key}: invalid component duration")
        spans, share = [], 0.0
        for activity in expected:
            if activity["producer"] == key[0] and key[1] in activity["members"]:
                spans.append((max(request["start"], activity["start"]), min(request["end"], activity["end"])))
                share += (activity["end"] - activity["start"]) * activity["members"][key[1]] / sum(activity["members"].values())
        lost = []
        for row in recording.records:
            if row["type"] == "Loss" and row.get("target_producer", row["producer"]) == key[0]:
                lost.append(tuple(row.get("time_range", [request["start"], request["end"]])))
        reliable_gpu = subtract_intervals(spans, lost)
        uncertain = recording.partial or recording.headers[key[0]].get("clock_error_ns") != 0
        actual_gpu = components.get("gpu_exec", -1)
        if (not uncertain and actual_gpu != union_duration(spans)) or actual_gpu > union_duration(reliable_gpu):
            errors.append(f"{key}: GPU wall-clock union disagrees")
        if any(left < request["end"] and right > request["start"] for left, right in lost):
            if str(actual.get("status", "")).lower() != "partial":
                errors.append(f"{key}: overlapping loss was not propagated")
        if not estimate_matches(actual.get("gpu_share_estimate_ns", -1), share, uncertain):
            errors.append(f"{key}: token-weighted activity estimate disagrees")
        intervals = actual.get("intervals", [])
        if intervals:
            spans = [(r["start_ns"], r["end_ns"]) for r in intervals]
            if (union_duration(spans) != e2e or sum(b - a for a, b in spans) != e2e
                    or any(a < request["start"] or b > request["end"] for a, b in spans)):
                errors.append(f"{key}: request intervals overlap or leave holes")
            by_category = {}
            for row in intervals:
                by_category[row["category"]] = by_category.get(row["category"], 0) + row["end_ns"] - row["start_ns"]
                if row["category"] == "gpu_exec" and any(left < row["end_ns"] and right > row["start_ns"] for left, right in lost):
                    errors.append(f"{key}: a lost interval was reported as known GPU execution")
            if any(by_category.get(category, 0) != value for category, value in components.items()):
                errors.append(f"{key}: request intervals disagree with component totals")
    for edge in analysis.get("edges", []):
        if edge.get("source") not in {"cupti-correlation", "cupti-resource", "external-correlation", "stream-order", "sync-record", "engine-membership", "ordered-inference", "unknown"}:
            errors.append("edge lacks a recognized provenance")
    errors.extend(check_graph_resource_edges(recording, analysis.get("edges", [])))
    summary = analysis.get("summary", {})
    linked = sum(bool(a["step_ids"]) for a in expected)
    if summary.get("gpu_activities") != len(expected) or summary.get("linked_gpu_activities") != linked:
        errors.append("completeness numerator or denominator disagrees")
    if truth:
        for item in truth.get("activities", []):
            candidates = [a for a in expected if a["producer"] == item["producer"] and a["seq"] == item["seq"]]
            if len(candidates) != 1 or candidates[0]["request_ids"] != item["request_ids"]:
                errors.append(f"handwritten truth disagrees for activity {item}")
        for item in truth.get("requests", []):
            actual = actual_requests.get((item["producer"], item["request_id"]), {})
            exact = not recording.partial and recording.headers[item["producer"]].get("clock_error_ns") == 0
            if exact and actual.get("components_ns", {}).get("gpu_exec") != item["gpu_exec_ns"]:
                errors.append(f"handwritten GPU union disagrees for {item['request_id']}")
            if not estimate_matches(actual.get("gpu_share_estimate_ns", -1), item["gpu_share_estimate_ns"], not exact):
                errors.append(f"handwritten share disagrees for {item['request_id']}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--truth", type=Path)
    args = parser.parse_args()
    recording = read_run(args.run)
    truth = json.loads(args.truth.read_text()) if args.truth else None
    errors = check(recording, json.loads(args.analysis.read_text()), truth)
    print(json.dumps({"ok": not errors, "reader_issues": recording.issues, "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
