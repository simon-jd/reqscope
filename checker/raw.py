"""Independent reference reader for RQSSEG01.

This module deliberately imports no collector, adapter, or analyzer code. A bad
frame makes a recording partial; it is never treated as valid evidence.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import struct
import zlib

MAGIC = b"RQSSEG01"
MAX_FRAME_BYTES = 1 << 20
MAX_RUN_BYTES = 256 << 20
MAX_RECORDS = 1_000_000


@dataclass
class Recording:
    headers: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    issues: list = field(default_factory=list)

    @property
    def partial(self):
        return bool(self.issues)


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _validate_json_tree(value, depth=0):
    if depth > 64:
        raise ValueError("JSON nesting limit")
    if isinstance(value, str):
        value.encode("utf-8")  # Reject lone escaped surrogate code points.
    elif isinstance(value, dict):
        for key, item in value.items():
            key.encode("utf-8")
            _validate_json_tree(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json_tree(item, depth + 1)


def _uint(value):
    return not isinstance(value, bool) and isinstance(value, int) and 0 <= value < 2 ** 64


def _valid_record_shape(row):
    kind = row.get("type")
    if not isinstance(kind, str) or not kind or kind == "Header":
        return False
    if kind in {"ApiCall", "ExtCorr", "GpuActivity", "GraphLaunch"} and not _uint(row.get("corr")):
        return False
    if kind in {"Step", "ExtCorr"} and not _uint(row.get("step_id")):
        return False
    if kind == "Request" and not isinstance(row.get("req_id"), str):
        return False
    if kind == "Step":
        members = row.get("members")
        if not isinstance(members, list):
            return False
        if any(not isinstance(m, dict) or not isinstance(m.get("req_id"), str) or not _uint(m.get("tokens")) for m in members):
            return False
        if len({m["req_id"] for m in members}) != len(members):
            return False
    return True


def read_run(path, max_frame_bytes=MAX_FRAME_BYTES, max_run_bytes=MAX_RUN_BYTES, max_records=MAX_RECORDS):
    path = Path(path)
    files = sorted(path.glob("*.rqs")) if path.is_dir() else [path]
    result = Recording()
    total = 0
    if not files:
        result.issues.append("no segment files")
    for filename in files:
        try:
            total += filename.stat().st_size
            if total > max_run_bytes:
                result.issues.append("run byte limit exceeded")
                break
            with filename.open("rb") as stream:
                if stream.read(8) != MAGIC:
                    result.issues.append(f"{filename.name}: invalid magic")
                    continue
                producer = None
                frame_number = 0
                while True:
                    frame_header = stream.read(8)
                    if not frame_header:
                        break
                    frame_number += 1
                    where = f"{filename.name}:frame {frame_number}"
                    if len(frame_header) != 8:
                        result.issues.append(f"{where}: truncated frame header")
                        break
                    length, checksum = struct.unpack("<II", frame_header)
                    if length == 0 or length > max_frame_bytes:
                        result.issues.append(f"{where}: invalid frame length")
                        break
                    raw = stream.read(length)
                    if len(raw) != length:
                        result.issues.append(f"{where}: truncated payload")
                        break
                    if zlib.crc32(raw) & 0xFFFFFFFF != checksum:
                        result.issues.append(f"{where}: CRC mismatch")
                        continue
                    try:
                        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object,
                                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                        _validate_json_tree(obj)
                    except (UnicodeError, ValueError, RecursionError) as exc:
                        result.issues.append(f"{where}: invalid JSON ({type(exc).__name__})")
                        continue
                    if not isinstance(obj, dict):
                        result.issues.append(f"{where}: record must be an object")
                        continue
                    if frame_number == 1:
                        if obj.get("type") != "Header" or not isinstance(obj.get("producer"), str):
                            result.issues.append(f"{where}: missing Header")
                            break
                        producer = obj["producer"]
                        canonical = f"{obj.get('boot_id')}:{obj.get('pid')}:{obj.get('start_time_ticks')}"
                        if producer != canonical:
                            result.issues.append(f"{where}: producer/header identity mismatch")
                        if obj.get("clock") != "CLOCK_MONOTONIC":
                            result.issues.append(f"{where}: unsupported clock")
                        clock_error = obj.get("clock_error_ns")
                        if clock_error is not None and not _uint(clock_error):
                            result.issues.append(f"{where}: invalid clock error")
                        old = result.headers.get(producer)
                        if old and any(old.get(k) != obj.get(k) for k in ("clock", "clock_error_ns")):
                            result.issues.append(f"{where}: inconsistent producer header")
                        result.headers[producer] = obj
                        if obj.get("retention_loss"):
                            result.issues.append(f"{where}: retained header reports ring overwrite")
                        continue
                    if producer is None:
                        result.issues.append(f"{where}: record without valid Header")
                        continue
                    if obj.get("producer") != producer:
                        result.issues.append(f"{where}: foreign producer")
                        continue
                    sequence = obj.get("seq")
                    if not _uint(sequence) or sequence < 1:
                        result.issues.append(f"{where}: invalid sequence")
                        continue
                    if not _valid_record_shape(obj):
                        result.issues.append(f"{where}: invalid record shape")
                        continue
                    time_fields = ("t_begin", "t_end") if obj.get("type") == "Step" else ("start", "end")
                    if any(key in obj for key in time_fields):
                        begin, end = (obj.get(key) for key in time_fields)
                        if (any(not _uint(n) for n in (begin, end))
                                or end < begin):
                            result.issues.append(f"{where}: invalid time interval")
                            continue
                    if obj.get("type") == "Loss":
                        result.issues.append(f"{where}: explicit loss ({obj.get('reason', 'unknown')})")
                    if len(result.records) >= max_records:
                        result.issues.append(f"{where}: record count limit")
                        return result
                    result.records.append(obj)
                if not frame_number:
                    result.issues.append(f"{filename.name}: missing Header")
        except OSError as exc:
            result.issues.append(f"{filename.name}: {exc}")
    sequences = {}
    for record in result.records:
        sequences.setdefault(record["producer"], []).append(record["seq"])
    for producer, seqs in sequences.items():
        ordered = sorted(seqs)
        if len(ordered) != len(set(ordered)):
            result.issues.append(f"{producer}: duplicate sequence")
        if ordered and (ordered[0] != 1 or any(b != a + 1 for a, b in zip(ordered, ordered[1:]))):
            result.issues.append(f"{producer}: sequence gap")
    for producer, metadata in result.headers.items():
        rows = [r for r in result.records if r["producer"] == producer]
        sessions = [r for r in rows if r.get("type") == "Session"]
        states = [r.get("state") for r in sessions] + [metadata.get("session_state")]
        if "Partial" in states or metadata.get("backend_status") in {"unavailable", "failed"}:
            result.issues.append(f"{producer}: partial capture session")
        if any(state in {"Configured", "Attached", "Capturing", "Draining"} for state in states):
            last_work = max((r["seq"] for r in rows if r.get("type") not in {"Session", "Loss", "ClockCalibration"}), default=0)
            last_work = max(last_work, max((r["seq"] for r in sessions if r.get("state") in
                                           {"Configured", "Attached", "Capturing", "Draining"}), default=0))
            last_boundary = max((r["seq"] for r in sessions if r.get("state") in {"Finalized", "Snapshot"}), default=-1)
            if last_boundary < last_work:
                result.issues.append(f"{producer}: capture lacks a final or snapshot boundary")
    return result
