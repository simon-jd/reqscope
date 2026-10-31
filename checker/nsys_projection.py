"""Independent, evidence-only reqscope NVTX -> CUDA runtime -> GPU projection.

This implementation uses Python's standard library and the exported SQLite
schema. It does not import NVIDIA recipes or reqscope's analyzer. GPU times
are measurements only: they never select a runtime API or a request identity.
"""

import argparse
import bisect
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3

GLOBAL_PID_MASK = 0xFFFFFFFFFF000000
GPU_TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL": "kernel",
    "CUPTI_ACTIVITY_KIND_MEMCPY": "memcpy",
    "CUPTI_ACTIVITY_KIND_MEMSET": "memset",
}


class ProjectionError(ValueError):
    pass


def _integer(value):
    return type(value) is int


def _interval(start, end):
    return _integer(start) and _integer(end) and end >= start


def _pid(tid):
    return tid & GLOBAL_PID_MASK if _integer(tid) and 0 <= tid < 2**63 else None


def _correlation(value):
    return _integer(value) and value > 0


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value
    return json.loads(text, object_pairs_hook=pairs, parse_float=finite_float,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _membership(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > 1024 * 1024:
        raise ValueError("missing or oversized JSON payload")
    payload = _strict_json(text)
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    step = payload.get("step_id")
    if not _integer(step) or not 0 < step < 2**64:
        raise ValueError("invalid step_id")
    if payload.get("phase") not in ("execute_model", "sample_tokens"):
        raise ValueError("unsupported phase")
    members = payload.get("members")
    if not isinstance(members, list):
        raise ValueError("members must be an array")
    ids = []
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get("req_id"), str) or not member["req_id"]:
            raise ValueError("invalid request member")
        if "tokens" in member and (not _integer(member["tokens"]) or member["tokens"] < 0):
            raise ValueError("invalid member token count")
        if "phase" in member and member["phase"] not in ("prefill", "decode", "unknown"):
            raise ValueError("invalid member phase")
        ids.append(member["req_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate request member")
    return payload


def _distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "sum_ns": 0, "min_ns": None, "max_ns": None,
                "mean_ns": None, "p50_ns": None, "p95_ns": None, "p99_ns": None}
    def percentile(q):
        position = (len(values) - 1) * q
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"count": len(values), "sum_ns": sum(values), "min_ns": values[0],
            "max_ns": values[-1], "mean_ns": sum(values) / len(values),
            "p50_ns": percentile(.5), "p95_ns": percentile(.95), "p99_ns": percentile(.99)}


class _Budget:
    def __init__(self, max_bytes, max_work):
        self.max_bytes, self.max_work = max_bytes, max_work
        self.input_bytes = self.association_bytes = self.work = 0

    def read(self, row):
        # SQLite's separate length limit bounds each value/row before Python
        # materializes it; this counter bounds the cumulative retained input.
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ProjectionError("non-finite SQLite number")
            self.input_bytes += len(value.encode("utf-8")) if isinstance(value, str) else len(value) if isinstance(value, bytes) else 8
        if self.input_bytes > self.max_bytes:
            raise ProjectionError("input byte budget exceeded; no completeness result was produced")

    def association(self, byte_count):
        self.association_bytes += byte_count
        if self.association_bytes > self.max_bytes:
            raise ProjectionError("association byte budget exceeded; no completeness result was produced")

    def use_work(self, count=1):
        self.work += count
        if self.work > self.max_work:
            raise ProjectionError("join work budget exceeded; no completeness result was produced")


class _IntervalIndex:
    def __init__(self, ranges, budget):
        self.budget = budget
        self.rows = sorted(ranges, key=lambda r: (r["start_ns"], r["rowid"]))
        self.starts = [r["start_ns"] for r in self.rows]
        self.max_ends = []
        maximum = -(2**63)
        for row in self.rows:
            # An unclosed range cannot prove membership, but must not disappear
            # when testing ambiguity with a complete range on the same thread.
            end = row["end_ns"] if _interval(row["start_ns"], row["end_ns"]) else 2**63 - 1
            maximum = max(maximum, end)
            self.max_ends.append(maximum)

    def contains(self, start, end):
        result = []
        i = bisect.bisect_right(self.starts, start) - 1
        while i >= 0 and self.max_ends[i] >= end:
            self.budget.use_work()
            row = self.rows[i]
            if not _interval(row["start_ns"], row["end_ns"]) or row["end_ns"] >= end:
                result.append(row)
            i -= 1
        return sorted(result, key=lambda r: r["rowid"])


def project_sqlite(path, *, max_rows=2_000_000, max_bytes=256 * 1024**2,
                   max_value_bytes=2 * 1024**2, max_work=10_000_000):
    """Project a completed Nsight export read-only; fail explicitly on budget/schema errors."""
    path = Path(path).resolve()
    for name, value in (("max_rows", max_rows), ("max_bytes", max_bytes),
                        ("max_value_bytes", max_value_bytes), ("max_work", max_work)):
        if not _integer(value) or value <= 0:
            raise ProjectionError(f"{name} must be positive")
    if max_value_bytes > 2**31 - 1:
        raise ProjectionError("max_value_bytes exceeds SQLite's length-limit range")
    budget = _Budget(max_bytes, max_work)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if not hasattr(connection, "setlimit"):
            raise ProjectionError("Python SQLite setlimit support is required to bound values before allocation")
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max_value_bytes)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "ENUM_NSYS_EVENT_TYPE"}
        if not required <= tables:
            raise ProjectionError("missing required tables: " + ", ".join(sorted(required - tables)))
        count = 0
        def rows(table, required_columns=()):
            nonlocal count
            columns = {r[1] for r in connection.execute(f'PRAGMA table_info("{table}")')}
            if not set(required_columns) <= columns:
                raise ProjectionError(f"{table}: missing columns {sorted(set(required_columns) - columns)}")
            for row in connection.execute(f'SELECT rowid AS _source_rowid, * FROM "{table}"'):
                count += 1
                if count > max_rows:
                    raise ProjectionError("input row budget exceeded; no completeness result was produced")
                result = dict(row)
                if any(isinstance(value, bytes) for value in result.values()):
                    raise ProjectionError(f"{table}: binary SQLite values are unsupported; expected text/numeric columns")
                budget.read(result)
                yield result
        export_metadata = {}
        for table in ("META_DATA_EXPORT", "EXPORT_META_DATA"):
            if table in tables:
                export_metadata[table] = list(rows(table))
        strings = {}
        if "StringIds" in tables:
            for row in rows("StringIds", ("id", "value")):
                if not _integer(row["id"]) or not isinstance(row["value"], str):
                    raise ProjectionError("StringIds requires integer ids and text values")
                if row["id"] in strings:
                    raise ProjectionError("duplicate StringIds id")
                strings[row["id"]] = row["value"]
        event_types = {}
        for row in rows("ENUM_NSYS_EVENT_TYPE", ("id", "name")):
            if not isinstance(row["name"], str) or not _integer(row["id"]) or row["name"] in event_types:
                raise ProjectionError("invalid or duplicate NVTX event type")
            event_types[row["name"]] = row["id"]
        domain_create = event_types.get("NvtxDomainCreate")
        domain_destroy = event_types.get("NvtxDomainDestroy")
        range_types = {event_types[name] for name in ("NvtxPushPopRange", "NvtxStartEndRange",
                      "NvtxtPushPopRange", "NvtxtStartEndRange") if name in event_types}
        if domain_create is None or not range_types:
            raise ProjectionError("unsupported NVTX event type catalog")
        nvtx = list(rows("NVTX_EVENTS", ("start", "end", "eventType", "globalTid", "domainId")))
        def text_field(row, payload=False):
            fields = ("jsonText", "jsonTextId", "text", "textId") if payload else ("text", "textId")
            for field in fields:
                value = row.get(field)
                if field.endswith("Id"):
                    value = strings.get(value)
                if isinstance(value, str) and value:
                    return value, field
            return None, None
        domains = defaultdict(list)
        destroys = defaultdict(list)
        for row in nvtx:
            key = (_pid(row.get("globalTid")), row.get("domainId"))
            if key[0] is None or not _integer(key[1]) or key[1] < 0:
                continue
            if row["eventType"] == domain_create:
                text, field = text_field(row)
                domains[key].append({"name": text, "rowid": row["_source_rowid"],
                                     "start": row["start"], "field": field})
            elif row["eventType"] == domain_destroy:
                destroys[key].append(row["start"])
        ranges, ignored_ranges, membership_sizes = [], 0, {}
        for row in nvtx:
            if row["eventType"] not in range_types:
                continue
            domain_rows = domains.get((_pid(row.get("globalTid")), row.get("domainId")), [])
            matching = [r for r in domain_rows if r["name"] == "reqscope"]
            if not matching:
                ignored_ranges += 1
                continue
            if not _integer(row["start"]):
                raise ProjectionError("reqscope NVTX range has no integer start; its overlap cannot be bounded")
            item = {"table": "NVTX_EVENTS", "rowid": row["_source_rowid"],
                    "start_ns": row["start"], "end_ns": row["end"],
                    "global_tid": row["globalTid"], "domain_id": row["domainId"],
                    "domain_source_rowids": [r["rowid"] for r in domain_rows],
                    "status": "valid", "reasons": []}
            text, field = text_field(row, payload=True)
            item.update(payload_text=text, payload_field=field,
                        payload_string_id=row.get(field) if field and field.endswith("Id") else None)
            if len(domain_rows) != 1:
                item["reasons"].append("ambiguous_domain_registration")
            elif not _integer(matching[0]["start"]) or not _integer(row["start"]) or matching[0]["start"] > row["start"]:
                item["reasons"].append("domain_registration_not_before_range")
            if any(not _integer(t) or not _integer(row["end"]) or t < row["end"]
                   for t in destroys.get((_pid(row.get("globalTid")), row.get("domainId")), [])):
                item["reasons"].append("domain_lifetime_unproven")
            if not _interval(row["start"], row["end"]):
                item["reasons"].append("unclosed_or_invalid_range")
            if row.get("endGlobalTid") not in (None, row["globalTid"]):
                item["reasons"].append("cross_thread_range")
            if isinstance(text, str):
                budget.association(len(text.encode("utf-8")))
            try:
                payload = _membership(text)
                item.update(step_id=payload["step_id"], phase=payload["phase"], members=payload["members"])
                # Includes repeated members, IDs, source paths and per-member
                # references, not just input text. This is an explicit output
                # expansion budget, not a claim about exact Python RSS.
                membership_sizes[item["rowid"]] = 2 * len(json.dumps(payload["members"]).encode("utf-8")) + 256 * len(payload["members"])
            except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
                item["reasons"].append("invalid_membership_payload: " + str(exc))
            if item["reasons"]:
                item["status"] = "invalid"
            ranges.append(item)
        # execute_model and sample_tokens describe one process-scoped Step.
        # Disjoint ranges are not permission to reuse a step identity or to
        # silently change its members between phases.
        step_ranges = defaultdict(list)
        for row in ranges:
            if "step_id" in row:
                step_ranges[(_pid(row["global_tid"]), row["step_id"])].append(row)
        for group in step_ranges.values():
            phases = Counter(row["phase"] for row in group)
            memberships = {json.dumps(sorted(row["members"], key=lambda m: m["req_id"]), sort_keys=True)
                           for row in group}
            reasons = []
            if any(number > 1 for number in phases.values()):
                reasons.append("duplicate_step_phase")
            if len(memberships) > 1:
                reasons.append("conflicting_step_membership")
            for row in group:
                if reasons:
                    row["status"] = "invalid"
                    row["reasons"].extend(reasons)
        by_thread = defaultdict(list)
        for row in ranges:
            if _integer(row["start_ns"]) and _pid(row["global_tid"]) is not None:
                by_thread[row["global_tid"]].append(row)
        indices = {tid: _IntervalIndex(group, budget) for tid, group in by_thread.items()}
        runtime, by_correlation = [], defaultdict(list)
        for row in rows("CUPTI_ACTIVITY_KIND_RUNTIME", ("start", "end", "globalTid", "correlationId")):
            pid = _pid(row["globalTid"])
            api = {"table": "CUPTI_ACTIVITY_KIND_RUNTIME", "rowid": row["_source_rowid"],
                   "start_ns": row["start"], "end_ns": row["end"], "global_tid": row["globalTid"],
                   "global_pid_from_tid": pid, "correlation_id": row["correlationId"],
                   "name": strings.get(row.get("nameId")), "name_id": row.get("nameId"),
                   "status": "unknown", "reason": "no_containing_reqscope_range", "nvtx_rowids": []}
            if not _interval(row["start"], row["end"]) or pid is None:
                api["reason"] = "invalid_runtime_interval_or_thread"
            else:
                index = indices.get(row["globalTid"])
                candidates = index.contains(row["start"], row["end"]) if index else []
                budget.association(16 * len(candidates))
                api["nvtx_rowids"] = [r["rowid"] for r in candidates]
                if len(candidates) > 1:
                    api.update(status="ambiguous", reason="multiple_containing_reqscope_ranges")
                elif candidates:
                    nv = candidates[0]
                    if nv["status"] != "valid":
                        api["reason"] = "invalid_containing_reqscope_range"
                    else:
                        budget.use_work(len(nv["members"]))
                        budget.association(membership_sizes[nv["rowid"]])
                        api.update(step_id=nv["step_id"], phase=nv["phase"], members=nv["members"])
                        api.update(status="linked" if nv["members"] else "unknown",
                                   reason="exact_thread_and_full_api_containment" if nv["members"] else "empty_request_members")
            runtime.append(api)
            if pid is not None and _correlation(row["correlationId"]):
                by_correlation[(pid, row["correlationId"])].append(api)
        range_lookup = {r["rowid"]: r for r in ranges}
        activities = []
        for table, kind in GPU_TABLES.items():
            if table not in tables:
                continue
            for row in rows(table, ("start", "end", "correlationId", "globalPid")):
                name_id = row.get("demangledName", row.get("shortName"))
                activity = {"table": table, "rowid": row["_source_rowid"], "kind": kind,
                            "start_ns": row["start"], "end_ns": row["end"],
                            "duration_ns": row["end"] - row["start"] if _interval(row["start"], row["end"]) else None,
                            "timing_status": "valid" if _interval(row["start"], row["end"]) else "invalid",
                            "global_pid": row["globalPid"], "correlation_id": row["correlationId"],
                            "name": strings.get(name_id), "name_id": name_id,
                            "device_id": row.get("deviceId"), "context_id": row.get("contextId"),
                            "stream_id": row.get("streamId"), "graph_id": row.get("graphId"),
                            "graph_node_id": row.get("graphNodeId"), "bytes": row.get("bytes"),
                            "status": "unknown", "reason": "missing_runtime_same_process_correlation",
                            "runtime_rowids": [], "nvtx_rowids": [], "request_ids": [], "membership_sources": []}
                candidates = by_correlation.get((row["globalPid"], row["correlationId"]), []) if (
                    _integer(row["globalPid"]) and _correlation(row["correlationId"])) else []
                budget.use_work(len(candidates))
                nvtx_reference_count = sum(len(api["nvtx_rowids"]) for api in candidates)
                budget.use_work(nvtx_reference_count)
                budget.association(16 * (len(candidates) + nvtx_reference_count))
                activity["runtime_rowids"] = [a["rowid"] for a in candidates]
                activity["nvtx_rowids"] = sorted({r for a in candidates for r in a["nvtx_rowids"]})
                if len(candidates) > 1:
                    activity.update(status="ambiguous", reason="multiple_runtime_same_process_correlation")
                elif candidates:
                    api = candidates[0]
                    activity.update(status=api["status"], reason=api["reason"])
                    if api["status"] == "linked":
                        nv = range_lookup[api["nvtx_rowids"][0]]
                        budget.use_work(len(api["members"]))
                        budget.association(membership_sizes[nv["rowid"]])
                        activity.update(step_id=api["step_id"], phase=api["phase"], members=api["members"])
                        activity["request_ids"] = [m["req_id"] for m in api["members"]]
                        activity["membership_sources"] = [{"request_id": m["req_id"], "nvtx_table": "NVTX_EVENTS",
                            "nvtx_rowid": nv["rowid"], "payload_field": nv["payload_field"],
                            "json_path": f"$.members[{i}].req_id"} for i, m in enumerate(api["members"])]
                # Invalid GPU timing does not invent/destroy identity evidence;
                # it is separately excluded from duration distributions.
                activities.append(activity)
        def coverage(group):
            statuses = Counter(r["status"] for r in group)
            valid = [r["duration_ns"] for r in group if r.get("duration_ns") is not None]
            return {"denominator": len(group), "linked": statuses["linked"], "ambiguous": statuses["ambiguous"],
                    "unknown": statuses["unknown"], "linked_fraction": statuses["linked"] / len(group) if group else None,
                    "duration_valid_denominator": len(valid), "invalid_duration_rows": len(group) - len(valid),
                    "duration_ns": _distribution(valid),
                    "by_status_duration_ns": {status: _distribution([r["duration_ns"] for r in group
                        if r["status"] == status and r.get("duration_ns") is not None])
                        for status in ("linked", "ambiguous", "unknown")}}
        by_name, requests = defaultdict(list), defaultdict(list)
        for row in activities:
            if row["kind"] == "kernel":
                by_name[row["name"] or "<unresolved kernel name>"].append(row)
            for req_id in row["request_ids"]:
                requests[(row["global_pid"], req_id)].append(row)
        report = {"schema_version": 1, "source": {"sqlite": str(path), "size_bytes": path.stat().st_size,
                    "rows_read": count, "row_budget": max_rows, "opened_read_only": True,
                    "input_bytes": budget.input_bytes, "association_expansion_bytes": budget.association_bytes,
                    "byte_budget_each": max_bytes, "sqlite_value_or_row_byte_limit": max_value_bytes,
                    "join_work": budget.work, "join_work_budget": max_work, "export_metadata": export_metadata,
                    "schema_contract_status": "assumed fixed Nsight Systems runtime association contract; metadata recorded without claiming forward schema compatibility"},
                  "method": {"domain": "exact reqscope domain registration scoped by globalPid",
                    "api_nvtx": "same globalTid and nvtx.start <= runtime.start <= runtime.end <= nvtx.end; unique range",
                    "gpu_api": "equal correlationId and kernel.globalPid == (runtime.globalTid & 0xFFFFFFFFFF000000); unique runtime row",
                    "correlation_namespace": "Nsight export schema association contract/comment, not an enforced SQLite FK: GPU correlationId refers to CUPTI_ACTIVITY_KIND_RUNTIME; other API tables are not identity candidates",
                    "gpu_time_used_for_identity": False, "request_id_rewriting": False,
                    "duration_semantics": "sum of activity durations, including overlapping/shared GPU work; not exclusive request time",
                    "percentile_method": "linear interpolation at (n-1)*q"},
                  "nvtx_summary": {"reqscope_ranges": len(ranges), "invalid_reqscope_ranges": sum(r["status"] != "valid" for r in ranges),
                                   "other_or_unregistered_domain_ranges": ignored_ranges},
                  "runtime_summary": {"denominator": len(runtime), **{status: sum(r["status"] == status for r in runtime)
                                       for status in ("linked", "ambiguous", "unknown")}},
                  "gpu_summary": coverage(activities),
                  "gpu_by_kind": {kind: coverage([r for r in activities if r["kind"] == kind]) for kind in GPU_TABLES.values()},
                  "kernel_statistics": [{"name": name, **coverage(group)} for name, group in sorted(by_name.items())],
                  "requests": [{"global_pid": pid, "request_id": req_id, "gpu_activity_refs": [{"table": r["table"], "rowid": r["rowid"]} for r in group],
                      "shared_full_activity_duration_ns": _distribution([r["duration_ns"] for r in group if r["duration_ns"] is not None])}
                      for (pid, req_id), group in sorted(requests.items())],
                  "nvtx_ranges": ranges, "runtime_apis": runtime, "gpu_activities": activities}
        return report
    except sqlite3.DataError as exc:
        raise ProjectionError(f"SQLite value/row length limit exceeded: {exc}") from exc
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-value-bytes", type=int, default=2 * 1024**2)
    parser.add_argument("--max-work", type=int, default=10_000_000)
    args = parser.parse_args()
    try:
        report = project_sqlite(args.sqlite, max_rows=args.max_rows, max_bytes=args.max_bytes,
                                max_value_bytes=args.max_value_bytes, max_work=args.max_work)
    except (ProjectionError, sqlite3.Error, OSError) as exc:
        parser.exit(2, f"projection failed: {exc}\n")
    document = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x") as output:
                output.write(document)
        except OSError as exc:
            parser.exit(2, f"cannot create new projection output: {exc}\n")
    else:
        print(document, end="")


if __name__ == "__main__":
    main()
