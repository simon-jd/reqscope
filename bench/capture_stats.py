"""Summarize the raw captures of bench/interference.py runs.

    python bench/capture_stats.py results/raw/interference results/raw/overhead-{light,trace}

Reads each run's capture/segment-*.rqs frame by frame (CRC checked, one
segment in memory at a time) and its client.json, and reports per run:

- capture bytes, segments, records by type, and the most frequent API
  call names;
- the capture header's API policy (all driver/runtime APIs, or the
  submission/copy/sync whitelist) and segment rotation size;
- GPU activities by kind, and how many have no GPU timestamps (CUPTI
  reports 0 when it could not collect one);
- the sum of Loss record counts by reason and unit;
- Session states seen: a capture flushed before vLLM SIGKILLs EngineCore
  has no Finalized record;
- write rate: bytes of the frames the collector wrote during the client's
  load window (by each record's `timestamp`), divided by that window.

Uses only the standard library and none of the analyzer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import struct
import sys
import zlib

MAGIC = b"RQSSEG01"


def frames(path: Path):
    """Yield (record or None, frame bytes); None marks a bad or truncated frame."""
    data = path.read_bytes()
    if data[:8] != MAGIC:
        yield None, len(data)
        return
    offset = 8
    while offset + 8 <= len(data):
        length, checksum = struct.unpack_from("<II", data, offset)
        end = offset + 8 + length
        if end > len(data):
            yield None, len(data) - offset
            return
        payload = data[offset + 8:end]
        if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
            yield None, end - offset
        else:
            yield json.loads(payload), end - offset
        offset = end


def summarize(run: Path) -> dict:
    client = json.loads((run / "client.json").read_text())
    begin, end = client["started_ns"], client["ended_ns"]
    seconds = (end - begin) / 1e9
    segments = sorted((run / "capture").glob("segment-*.rqs"))
    types, sessions, losses, apis = Counter(), Counter(), Counter(), Counter()
    kinds: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    last_seq: dict[str, int] = {}
    header: dict = {}
    capture_bytes = in_window = bad = 0
    segment_sizes = []
    for segment in segments:
        size = segment.stat().st_size
        capture_bytes += size
        segment_sizes.append(size)
        for record, frame_bytes in frames(segment):
            if record is None:
                bad += 1
                continue
            kind = record.get("type")
            types[kind] += 1
            if kind == "Header":
                header = header or record
                continue
            if begin <= record.get("timestamp", -1) <= end:
                in_window += frame_bytes
            producer = record.get("producer", "")
            last_seq[producer] = max(last_seq.get(producer, 0), record.get("seq", 0))
            if kind == "GpuActivity":
                entry = kinds[record.get("kind", "?")]
                entry[0] += 1
                entry[1] += not record.get("start")
            elif kind == "ApiCall":
                apis[record.get("name", "?")] += 1
            elif kind == "Loss":
                losses[f"{record.get('reason')} ({record.get('count_unit', 'unspecified')})"] += record.get("count", 0)
            elif kind == "Session":
                sessions[record.get("state")] += 1
    segment_sizes.sort()
    records = sum(n for t, n in types.items() if t != "Header")
    return {
        "run": run.name,
        "mode": client["mode"],
        "api_capture_policy": header.get("api_capture_policy"),
        "load_seconds": round(seconds, 3),
        "capture_bytes": capture_bytes,
        "segments": len(segments),
        "segment_bytes_median": segment_sizes[len(segment_sizes) // 2] if segment_sizes else None,
        "records": records,
        "records_by_type": dict(sorted(types.items(), key=lambda item: str(item[0]))),
        "api_calls_top": dict(apis.most_common(8)),
        "max_seq_by_producer": last_seq,
        "bad_frames": bad,
        "gpu_activities_by_kind": {k: {"total": v[0], "untimed": v[1]} for k, v in sorted(kinds.items())},
        "loss_counts": dict(sorted(losses.items())),
        "session_states": dict(sorted(sessions.items(), key=lambda item: str(item[0]))),
        "load_window_bytes": in_window,
        "load_window_mb_per_s": round(in_window / seconds / 1e6, 1),
    }


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(json.dumps([summarize(Path(p)) for p in sys.argv[1:]], indent=2))


if __name__ == "__main__":
    main()
