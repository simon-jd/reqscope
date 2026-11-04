"""Score blame's slow-step causes against the injected causes of multicause runs.

    python bench/score_causes.py [--rule RULE] results/raw/multicause-trace-s1 results/raw/multicause-plugin-s1 ...

For each run directory: runs `reqscope blame` on the capture unless
blame.json exists, then reads the injected events from client.json, which
blame never sees. The 16 decodes run in lockstep, so slow gaps are grouped
by the engine step that produced them; a step's predicted cause is the most
common cause blame gave its gaps. Its true cause is the injected event whose
window [start, end + 20 ms] overlaps the step's gap the most, or "none" when
no event overlaps it (the run's own slow steps).

--rule picks how a slow step's cause is predicted; the slow steps and the
scoring are the same for every rule:
- blame (default): the most common cause blame gave the step's gaps;
- trivial: interference if the step had any co-scheduled prefill, else host
  delay;
- prefill_only: interference if the step had any co-scheduled prefill, else
  undetermined: what per-step token counts alone can say;
- vllm: vLLM's own per-step statistics (--vllm-iteration-log runs;
  vllm-iterations.json from bench/vllm_iterations.py): interference if the
  step had context (prefill) tokens; otherwise GPU slowdown if the step's
  "iteration elapsed time", the engine's wait for the step's output,
  exceeds the median of steps without context tokens by at least half the
  step's excess, else host delay;
- nsys: the same test on the step's GPU busy time measured by Nsight
  Systems (--nsys runs; nsys-steps.json from checker.nsys_projection
  --steps).

Per run and summed per collector mode:
- the confusion matrix, true cause by predicted cause;
- per cause: recall over the steps with that true cause; precision over
  all steps given that cause, and over the steps inside some event window
  (excluding the run's own slow steps);
- per cause: the share of injected events with at least one step given the
  right cause;
- per run, the host's mean 1-minute load average during the run;
- for steps blame gave their true cause, GPU slowdown or host delay, the
  medians of the evidence behind the verdict (the first gap's
  cause_evidence): GPU busy time and its baseline, launch wait and its
  baseline, and host extra;
- host-delay verdicts that blame flagged "collector_active" (the
  collector was busy for at least half of the host extra), by true cause,
  and host-delay precision over the verdicts that were not flagged.

Sampled runs (--mode sampled) are grouped by their sampling rules, as in
bench/score.py. For them the output also has the share of all engine steps
traced (blame's sampled_steps), how many slow steps were traced, and the
same metrics over the traced slow steps alone (traced_only): what blame
gets right on the steps it can see.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import arm, arm_order, mean_load, reqscope  # noqa: E402

CAUSES = ["interference", "gpu_slowdown", "host_delay"]
PREDICTED = CAUSES + ["unexplained", "undetermined"]
SPILL_NS = 20_000_000
EVIDENCE = ["gpu_busy_ns", "gpu_busy_baseline_ns", "launch_wait_ns", "launch_wait_baseline_ns", "host_extra_ns"]
RULES = ["blame", "trivial", "prefill_only", "vllm", "nsys"]


def steps_of(blame: dict) -> dict[int, dict]:
    steps: dict[int, dict] = {}
    for row in blame["requests"]:
        if "decode-" not in row["req_id"]:
            continue
        for gap in row["slow_gaps"]:
            if not gap["steps"]:
                continue
            step = gap["steps"][0]
            entry = steps.setdefault(step["step_id"],
                                     {"begin": gap["begin_ns"], "end": gap["end_ns"], "causes": Counter(),
                                      "flagged": False, "evidence": gap.get("cause_evidence", {}), "excess": 0,
                                      "prefill": any(m["phase"] == "prefill" for m in step["co_scheduled"]),
                                      "traced": step.get("traced", True)})
            entry["causes"][gap["cause"]] += 1
            entry["excess"] = max(entry["excess"], gap["excess_ns"])
            entry["flagged"] |= "collector_active" in gap.get("flags", [])
    for step_id, entry in steps.items():
        entry["id"] = step_id
    return steps


def truth(step: dict, events: list[dict]) -> tuple[str, int | None]:
    best, index, most = "none", None, 0
    for i, event in enumerate(events):
        overlap = min(step["end"], event["end_ns"] + SPILL_NS) - max(step["begin"], event["start_ns"])
        if overlap > most:
            best, index, most = event["cause"], i, overlap
    return best, index


def step_times(run: Path, rule: str) -> tuple[dict[int, int], int]:
    """Per step id, the time the rule tests (ns), and its median over steps without prefill."""
    if rule == "vllm":
        rows = json.loads((run / "vllm-iterations.json").read_text())["steps"]
        times = {r["step_id"]: r["elapsed_ns"] for r in rows}
        plain = [r["elapsed_ns"] for r in rows if r["context_tokens"] == 0]
    else:
        rows = json.loads((run / "nsys-steps.json").read_text())["steps"]
        times = {r["step_id"]: r["gpu_busy_ns"] for r in rows}
        plain = [r["gpu_busy_ns"] for r in rows if r["prefill_tokens"] == 0]
    return times, int(statistics.median(plain))


def predict(rule: str, step: dict, times: dict[int, int], median: int) -> str:
    if rule == "blame":
        return step["causes"].most_common(1)[0][0]
    if step["prefill"]:
        return "interference"
    if rule == "trivial":
        return "host_delay"
    if rule == "prefill_only" or step["id"] not in times:
        return "undetermined"
    return "gpu_slowdown" if times[step["id"]] - median >= step["excess"] / 2 else "host_delay"


def ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def medians(evidence: dict) -> dict:
    return {cause: {"steps": len(values[EVIDENCE[0]]),
                    **{f: statistics.median(values[f]) if values[f] else None for f in EVIDENCE}}
            for cause, values in sorted(evidence.items())}


def metrics(confusion: Counter, detected: Counter, injected: Counter, flagged: Counter) -> dict:
    per_cause = {}
    for cause in CAUSES:
        true = sum(n for (t, _), n in confusion.items() if t == cause)
        given = sum(n for (_, p), n in confusion.items() if p == cause)
        given_inside = sum(n for (t, p), n in confusion.items() if p == cause and t != "none")
        right = confusion[(cause, cause)]
        per_cause[cause] = {"true_steps": true, "recall": ratio(right, true),
                            "precision": ratio(right, given), "precision_inside_windows": ratio(right, given_inside),
                            "events": injected[cause], "events_detected": detected[cause]}
    matrix = {t: {p: confusion[(t, p)] for p in PREDICTED} for t in CAUSES + ["none"]}
    given = sum(n for (_, p), n in confusion.items() if p == "host_delay")
    unflagged_right = confusion[("host_delay", "host_delay")] - flagged[("host_delay", "host_delay")]
    unflagged = given - sum(n for (_, p), n in flagged.items() if p == "host_delay")
    flags = {"host_delay_flagged_by_true_cause": {t: flagged[(t, "host_delay")] for t in CAUSES + ["none"]},
             "host_delay_precision_unflagged": ratio(unflagged_right, unflagged),
             "host_delay_recall_unflagged": ratio(unflagged_right, sum(n for (t, _), n in confusion.items()
                                                                        if t == "host_delay"))}
    return {"per_cause": per_cause, "confusion": matrix, "collector_active_flags": flags}


def score(run: Path, rule: str = "blame") -> tuple[dict, Counter, Counter, Counter, Counter, dict,
                                                   tuple[Counter, Counter]]:
    client = json.loads((run / "client.json").read_text())
    blame_path = run / "blame.json"
    if not blame_path.exists():
        blame_path.write_text(json.dumps(reqscope("blame", str(run / "capture"))))
    blame = json.loads(blame_path.read_text())
    events = client["results"]["events"]
    confusion, detected, injected = Counter(), Counter(), Counter(e["cause"] for e in events)
    flagged, traced, traced_hits = Counter(), Counter(), defaultdict(set)
    hits = defaultdict(set)
    evidence = defaultdict(lambda: defaultdict(list))
    times, median = step_times(run, rule) if rule in ("vllm", "nsys") else ({}, 0)
    for step in steps_of(blame).values():
        cause, index = truth(step, events)
        predicted = predict(rule, step, times, median)
        confusion[(cause, predicted)] += 1
        flagged[(cause, predicted)] += step["flagged"]
        if index is not None and predicted == cause:
            hits[cause].add(index)
        if step["traced"]:
            traced[(cause, predicted)] += 1
            if index is not None and predicted == cause:
                traced_hits[cause].add(index)
        if rule == "blame" and predicted == cause and cause in ("gpu_slowdown", "host_delay"):
            for field in EVIDENCE:
                if step["evidence"].get(field) is not None:
                    evidence[cause][field].append(step["evidence"][field])
    for cause, found in hits.items():
        detected[cause] = len(found)
    sampled = client["mode"] == "sampled"
    traced_detected = Counter({cause: len(found) for cause, found in traced_hits.items()})
    result = {"run": run.name, "rule": rule, "mode": arm(client), "seed": client["results"]["seed"],
              "load1_mean": mean_load(client.get("loadavg", []), client["started_ns"], client["ended_ns"]),
              "slow_steps": sum(confusion.values()), "trace_status": blame.get("trace", {}).get("status"),
              **({"rule_median_ns": median, "rule_steps_timed": len(times)} if times else {}),
              **metrics(confusion, detected, injected, flagged), "verdict_evidence_medians": medians(evidence)}
    if sampled:
        result["sampled_steps"] = blame["summary"].get("sampled_steps")
        result["traced_slow_steps"] = sum(traced.values())
        result["traced_only"] = metrics(traced, traced_detected, injected, Counter())["per_cause"]
    return result, confusion, detected, injected, flagged, evidence, (traced, traced_detected)


def main() -> None:
    args = sys.argv[1:]
    rule = "blame"
    if args[:1] == ["--rule"] and len(args) > 1:
        rule, args = args[1], args[2:]
    if not args or rule not in RULES:
        sys.exit(__doc__)
    runs, totals = [], defaultdict(lambda: [Counter(), Counter(), Counter(), Counter(), [],
                                            defaultdict(lambda: defaultdict(list)), Counter(), Counter(), [0, 0]])
    for path in args:
        result, confusion, detected, injected, flagged, evidence, (traced, traced_detected) = score(Path(path), rule)
        runs.append(result)
        total = totals[result["mode"]]
        total[0].update(confusion); total[1].update(detected); total[2].update(injected); total[3].update(flagged)
        total[4].append(result["seed"])
        for cause, fields in evidence.items():
            for field, values in fields.items():
                total[5][cause][field] += values
        total[6].update(traced); total[7].update(traced_detected)
        if result.get("sampled_steps"):
            total[8][0] += result["sampled_steps"]["steps"]; total[8][1] += result["sampled_steps"]["traced"]
    by_mode = {}
    for mode, (c, d, i, f, seeds, e, t, td, steps) in sorted(totals.items(), key=lambda item: arm_order(item[0])):
        by_mode[mode] = {"seeds": seeds, "slow_steps": sum(c.values()), **metrics(c, d, i, f),
                         "verdict_evidence_medians": medians(e)}
        if mode.startswith("sampled"):
            by_mode[mode].update(traced_share=ratio(steps[1], steps[0]), traced_slow_steps=sum(t.values()),
                                 traced_only=metrics(t, td, i, Counter())["per_cause"])
    print(json.dumps({"rule": rule, "by_mode": by_mode, "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
