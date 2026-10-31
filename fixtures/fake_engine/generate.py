"""Generate deterministic wire recordings and separately handwritten truth.

``python3 -m fixtures.fake_engine.generate /tmp/reqscope-fixture`` creates two
producers that intentionally reuse PID, correlation, step, and graph IDs.
No CUDA/GPU claims can be inferred from this fixture.
"""

import argparse
import copy
import json
from pathlib import Path
import struct
import zlib

A = "fixture-boot:42:100"
B = "fixture-boot:42:200"


def frame_bytes(obj):
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return struct.pack("<II", len(payload), zlib.crc32(payload) & 0xFFFFFFFF) + payload


def header(producer=A, clock_error_ns=0):
    boot, pid, start = producer.rsplit(":", 2)
    return {"type": "Header", "format_version": 1, "producer": producer,
            "boot_id": boot, "pid": int(pid), "start_time_ticks": int(start),
            "clock": "CLOCK_MONOTONIC", "clock_error_ns": clock_error_ns,
            "mode": "trace", "source": "synthetic", "cupti_version": "not-loaded",
            "subscription": "none"}


def write_segment(path, records, producer=A, clock_error_ns=0, *, omit_clock_error=False, header_extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preamble = header(producer, clock_error_ns)
    if omit_clock_error:
        preamble.pop("clock_error_ns", None)
    if header_extra:
        preamble.update(header_extra)
    with path.open("wb") as stream:
        stream.write(b"RQSSEG01")
        stream.write(frame_bytes(preamble))
        for i, record in enumerate(records, 1):
            row = {"producer": producer, "seq": i, **record}
            stream.write(frame_bytes(row))
    return path


def request(req_id, start, end, prompt_tokens=16):
    return dict(type="Request", req_id=req_id, start=start, end=end, prompt_tokens=prompt_tokens)


def member(req_id, tokens=1, phase="decode"):
    return dict(req_id=req_id, tokens=tokens, phase=phase)


def step(step_id, begin, end, members, **extra):
    return dict(type="Step", step_id=step_id, t_begin=begin, t_end=end, tid=43,
                members=members, preempted=[], finished=[], **extra)


def launch(corr, step_id, begin):
    return [dict(type="ExtCorr", corr=corr, step_id=step_id),
            dict(type="ApiCall", corr=corr, cbid="cudaGraphLaunch", tid=43, start=begin, end=begin + 5)]


def gpu(corr, begin, end, node=1, **extra):
    return dict(type="GpuActivity", corr=corr, kind="kernel", ctx=1, stream=node,
                start=begin, end=end, graph_id=7, graph_node_id=node, bytes=0, **extra)


def replay_records():
    # Seq numbering is intentional and referenced in handwritten TRUTH below.
    first = [request("alpha", 0, 600), request("beta", 0, 500), request("gamma", 280, 600),
             step(1, 100, 250, [member("alpha", 3, "prefill"), member("beta")], padding_tokens=4)]
    first += launch(1, 1, 110)                         # 5, 6
    first += [gpu(1, 130, 210), gpu(1, 150, 230, 2)]   # 7, 8: overlap 60 ns
    first += [step(2, 300, 450, [member("beta"), member("gamma")], padding_tokens=6)]
    first[-1]["preempted"] = ["alpha"]
    first[-1]["finished"] = ["beta"]
    first += launch(2, 2, 310) + [gpu(2, 330, 380)]    # 10, 11, 12: same graph, new members
    first += [step(3, 500, 550, [member("alpha")])]
    first += launch(3, 3, 501) + [gpu(3, 510, 530)]    # 14, 15, 16: resume alpha
    first += [gpu(404, 50, 70)]                       # 17: deliberately unlinked
    first += [step(4, 560, 580, [member("gamma")])]    # 18: empty GPU step
    first += [dict(type="Sched", tid=43, start=100, end=130, state="runnable"),
              dict(type="Sched", tid=43, start=230, end=250, state="sleeping")]
    second = [request("delta", 0, 300), step(1, 100, 250, [member("delta")])]
    second += launch(1, 1, 110) + [gpu(1, 140, 180)]   # 5: reused PID, step, corr, graph
    return {A: first, B: second}


# Literal answers, never computed from the reader or analyzer under test.
TRUTH = {
    "description": "graph replay, padding, overlapping GPU, preemption, empty step, PID reuse, orphan activity",
    "activities": [
        {"producer": A, "seq": 7, "request_ids": ["alpha", "beta"]},
        {"producer": A, "seq": 8, "request_ids": ["alpha", "beta"]},
        {"producer": A, "seq": 12, "request_ids": ["beta", "gamma"]},
        {"producer": A, "seq": 16, "request_ids": ["alpha"]},
        {"producer": A, "seq": 17, "request_ids": []},
        {"producer": B, "seq": 5, "request_ids": ["delta"]},
    ],
    "requests": [
        {"producer": A, "request_id": "alpha", "gpu_exec_ns": 120, "gpu_share_estimate_ns": 140},
        {"producer": A, "request_id": "beta", "gpu_exec_ns": 150, "gpu_share_estimate_ns": 65},
        {"producer": A, "request_id": "gamma", "gpu_exec_ns": 50, "gpu_share_estimate_ns": 25},
        {"producer": B, "request_id": "delta", "gpu_exec_ns": 40, "gpu_share_estimate_ns": 40},
    ],
    "total_linked_activity_ns": 270,
    "gpu_activity_count": 6,
    "linked_gpu_activity_count": 5,
}


def generate(destination, scenario="replay"):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = replay_records()
    error_ns = 0
    if scenario == "loss":
        records[A].append(dict(type="Loss", reason="buffer_exhausted", count=3,
                               time_range=[100, 200], seq_range=[21, 23]))
    elif scenario == "clock":
        error_ns = 50
    elif scenario == "ambiguous":
        records[A].append(dict(type="ExtCorr", corr=1, step_id=2))
    elif scenario != "replay":
        raise ValueError(f"unknown scenario {scenario}")
    for i, (producer, rows) in enumerate(records.items()):
        write_segment(destination / f"producer-{i}.rqs", rows, producer, error_ns)
    truth = copy.deepcopy(TRUTH)
    truth["scenario"] = scenario
    if scenario == "ambiguous":
        # The injected contradictory evidence cannot identify replay 1, even
        # though the fixture author knows the original physical membership.
        for row in truth["activities"]:
            if row["producer"] == A and row["seq"] in (7, 8):
                row["request_ids"] = []
        truth["requests"] = [
            {"producer": A, "request_id": "alpha", "gpu_exec_ns": 20, "gpu_share_estimate_ns": 20},
            {"producer": A, "request_id": "beta", "gpu_exec_ns": 50, "gpu_share_estimate_ns": 25},
            {"producer": A, "request_id": "gamma", "gpu_exec_ns": 50, "gpu_share_estimate_ns": 25},
            {"producer": B, "request_id": "delta", "gpu_exec_ns": 40, "gpu_share_estimate_ns": 40},
        ]
        truth["total_linked_activity_ns"] = 110
        truth["linked_gpu_activity_count"] = 3
    (destination / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--scenario", choices=["replay", "loss", "clock", "ambiguous"], default="replay")
    args = parser.parse_args()
    print(generate(args.destination, args.scenario))


if __name__ == "__main__":
    main()
