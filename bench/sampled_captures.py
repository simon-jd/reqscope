"""What a sampled capture holds, and why the trigger skips steps near a prefill.

    python bench/sampled_captures.py results/raw/multicause-sampled-s4-s1 results/raw/openloop-sampled-light-r1 ...

Per run directory (capture/ and client.json from bench/interference.py):
- the collector mode and, for sampled runs, the header's sampling rules;
- engine steps and TracedStep records by reason;
- GPU records by kind, how many have no timestamps, and per decode-only
  step (no prefill member) the median kernel and graph records linked to
  traced steps, to guard steps (an untraced step right after a traced one,
  which keeps kernel tracing on since commit 6fd8ae3) and to the other
  untraced steps;
- per traced decode-only step, the time from its first kernel's start to
  its last kernel's end, split by whether the next step was traced too
  (kernel_span_ms): median, 90th percentile and longest; the same for
  guard steps, whose end times are not their own. For traced steps before
  an untraced one, also how many spanned over 10 ms, and how many of those
  were followed by a step that launched none of their graphs or had
  prefill;
- the median host time of graph launch calls (cuGraphLaunch and
  cudaGraphLaunch API records) on traced, guard and other untraced steps;
- Loss counts by reason;
- from the last CollectorStats record: the sampling counts, and calls,
  mean and longest time of the switch between kernel tracing on and off
  (switch_granularity) and of the writer's forced CUPTI flush
  (sampled_flush);
- intervals: each step's start-to-start interval against the median of
  the previous 64 intervals that counted (the upper median, as the
  collector's trigger takes it; only after 16), in four groups: the step and the
  one before it had no prefill (the intervals the trigger uses), the step
  had prefill, only the step before it had prefill, both had. For each
  group, how many intervals reached 1.3, 1.5, 2 and 3 times that median.

Uses only the standard library and none of the analyzer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_stats import frames  # noqa: E402

THRESHOLDS = (1.3, 1.5, 2.0, 3.0)
TYPES = {"Header", "Step", "TracedStep", "GpuActivity", "ExtCorr", "Loss", "CollectorStats", "ApiCall"}


def intervals(steps: dict[int, tuple[int, bool]]) -> dict:
    groups = {"decode_after_decode": [], "prefill": [], "after_prefill": [], "prefill_after_prefill": []}
    counted: list[int] = []
    for a in sorted(steps):
        if a + 1 not in steps or a - 1 not in steps:
            continue
        interval = steps[a + 1][0] - steps[a][0]
        if interval <= 0:
            continue
        group = {(False, False): "decode_after_decode", (False, True): "prefill",
                 (True, False): "after_prefill", (True, True): "prefill_after_prefill"}[(steps[a - 1][1], steps[a][1])]
        if len(counted) >= 16:
            groups[group].append(interval / statistics.median_high(counted[-64:]))
        if group == "decode_after_decode":
            counted.append(interval)
    return {name: {"intervals": len(ratios), **{f"at_least_{t}x": sum(r >= t for r in ratios) for t in THRESHOLDS}}
            for name, ratios in groups.items()}


def site(stats: dict, name: str) -> dict | None:
    entry = stats.get("sites", {}).get(name)
    if not entry or not entry["calls"]:
        return None
    return {"calls": entry["calls"], "mean_us": round(entry["ns"] / entry["calls"] / 1e3, 1),
            "max_us": round(entry["max_ns"] / 1e3, 1)}


def summarize(run: Path) -> dict:
    header, stats = None, None
    steps: dict[int, tuple[int, bool]] = {}
    traced: dict[int, str] = {}
    corr_step: dict[int, int] = {}
    activities: list[tuple[int, str, int, int, int]] = []
    launches: list[tuple[int, int]] = []
    kinds, untimed, losses = Counter(), Counter(), Counter()
    for segment in sorted((run / "capture").glob("segment-*.rqs")):
        for record, _ in frames(segment):
            if not record or record.get("type") not in TYPES:
                continue
            kind = record["type"]
            if kind == "Header" and header is None:
                header = record
            elif kind == "Step":
                steps[record["step_id"]] = (record["t_begin"],
                                            any(m.get("phase") == "prefill" for m in record.get("members", [])))
            elif kind == "TracedStep":
                traced[record["step_id"]] = record["reason"]
            elif kind == "ExtCorr":
                corr_step[record["corr"]] = record["step_id"]
            elif kind == "GpuActivity":
                kinds[record["kind"]] += 1
                if not record.get("start") or not record.get("end"):
                    untimed[record["kind"]] += 1
                activities.append((record["corr"], record["kind"], record.get("start") or 0, record.get("end") or 0,
                                   record.get("graph_id") or 0))
            elif kind == "ApiCall":
                name = str(record.get("name") or record.get("cbid", ""))
                if "GraphLaunch" in name and record.get("end", 0) >= record.get("start", 0):
                    launches.append((record["corr"], record["end"] - record["start"]))
            elif kind == "Loss":
                losses[record["reason"]] += record.get("count", 0)
            elif kind == "CollectorStats":
                stats = record
    per_step: dict[int, Counter] = defaultdict(Counter)
    spans: dict[int, tuple[int, int]] = {}
    graphs: dict[int, set[int]] = defaultdict(set)
    for corr, kind, start, end, graph in activities:
        if corr in corr_step:
            step = corr_step[corr]
            per_step[step][kind] += 1
            if graph:
                graphs[step].add(graph)
            if kind == "kernel" and start and end:
                low, high = spans.get(step, (start, end))
                spans[step] = (min(low, start), max(high, end))
    sampled = header.get("mode") == "sampled"
    def group(s: int) -> str:
        return "traced" if s in traced else "guard" if s - 1 in traced else "untraced"
    linked = {}
    for name in ("traced", "guard", "untraced"):
        ids = [s for s, (_, prefill) in steps.items() if not prefill and group(s) == name]
        if ids:
            linked[name] = {"decode_steps": len(ids),
                            "kernels_median": statistics.median(per_step[s]["kernel"] for s in ids),
                            "graphs_median": statistics.median(per_step[s]["graph"] for s in ids),
                            "steps_with_kernels": sum(per_step[s]["kernel"] > 0 for s in ids)}
    kernel_span = {}
    for name, chosen in (("next_traced", True), ("next_untraced", False)):
        values = sorted((spans[s][1] - spans[s][0]) / 1e6 for s in traced
                        if s in spans and s in steps and not steps[s][1] and (s + 1 in traced) == chosen)
        if values:
            kernel_span[name] = {"decode_steps": len(values), "median": round(statistics.median(values), 2),
                                 "p90": round(values[int(0.9 * (len(values) - 1))], 2), "max": round(values[-1], 1)}
    if "next_untraced" in kernel_span:
        long = [s for s in traced if s in spans and s in steps and not steps[s][1] and s + 1 not in traced
                and spans[s][1] - spans[s][0] > 10_000_000]
        kernel_span["next_untraced"].update(
            over_10ms=len(long), over_10ms_next_shares_no_graph=sum(not graphs[s] & graphs[s + 1] for s in long),
            over_10ms_next_has_prefill=sum(s + 1 in steps and steps[s + 1][1] for s in long))
    values = sorted((spans[s][1] - spans[s][0]) / 1e6 for s in spans
                    if s in steps and not steps[s][1] and group(s) == "guard")
    if values:
        kernel_span["guard"] = {"decode_steps": len(values), "median": round(statistics.median(values), 2),
                                "p90": round(values[int(0.9 * (len(values) - 1))], 2), "max": round(values[-1], 1)}
    launch_us = defaultdict(list)
    for corr, duration in launches:
        if corr in corr_step:
            launch_us[group(corr_step[corr])].append(duration / 1e3)
    out = {"run": run.name, "mode": header.get("mode"), "steps": len(steps),
           "graph_launch_us_median": {k: round(statistics.median(v), 1) for k, v in sorted(launch_us.items())},
           "gpu_records_by_kind": dict(kinds), "untimed_gpu_records_by_kind": dict(untimed),
           "losses": dict(losses), "intervals": intervals(steps)}
    if sampled:
        out.update(trace_sampling=header.get("trace_sampling"), traced_steps=len(traced),
                   traced_by_reason=dict(Counter(traced.values())), linked_per_decode_step=linked,
                   kernel_span_ms=kernel_span)
        if stats:
            out.update(sampling=stats.get("sampling"), switch_granularity=site(stats, "switch_granularity"),
                       sampled_flush=site(stats, "sampled_flush"))
    return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(json.dumps({"runs": [summarize(Path(p)) for p in sys.argv[1:]]}, indent=2))


if __name__ == "__main__":
    main()
