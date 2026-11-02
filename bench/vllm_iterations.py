"""Match vLLM's own per-step log lines to the plugin's step ids.

    python bench/vllm_iterations.py results/raw/multicause-nsys-s1

A run made with --vllm-iteration-log has one "Iteration(N): ..." line per
engine step in server.log: context (prefill) requests and tokens, generation
requests and tokens, and the iteration's elapsed time. vLLM numbers the
lines itself, so this finds the constant offset between its index and the
plugin's step ids at which the two agree on context requests, context tokens
and generation requests, and writes vllm-iterations.json: the offset, how
many steps agree, and each agreeing step's counts and elapsed time.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checker.raw import read_run  # noqa: E402

LINE = re.compile(r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests, "
                  r"(\d+) generation tokens, iteration elapsed time: ([\d.]+) ms")


def plugin_steps(capture: Path) -> dict[int, tuple[int, int, int]]:
    steps = {}
    for record in read_run(capture).records:
        if record.get("type") == "Step":
            members = record.get("members", [])
            prefill = [m for m in members if m.get("phase") == "prefill"]
            steps[record["step_id"]] = (len(prefill), sum(m.get("tokens", 0) for m in prefill),
                                        len(members) - len(prefill))
    return steps


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    lines = {}
    for line in (run / "server.log").read_text(errors="replace").splitlines():
        if match := LINE.search(line):
            index, ctx_req, ctx_tok, gen_req, _, elapsed = match.groups()
            lines[int(index)] = (int(ctx_req), int(ctx_tok), int(gen_req), float(elapsed))
    steps = plugin_steps(run / "capture")
    # Candidate offsets from a few context steps, which are rare; then count agreement over all lines.
    candidates = Counter(step - index for index, line in lines.items() if line[1]
                         for step, counts in steps.items() if counts[:2] == line[:2])
    best, agree = None, []
    for offset, _ in candidates.most_common(20):
        matched = [i for i, line in lines.items() if steps.get(i + offset) == line[:3]]
        if len(matched) > len(agree):
            best, agree = offset, matched
    rows = [{"step_id": i + best, "context_requests": lines[i][0], "context_tokens": lines[i][1],
             "generation_requests": lines[i][2], "elapsed_ns": round(lines[i][3] * 1e6)} for i in sorted(agree)]
    out = {"offset": best, "log_lines": len(lines), "plugin_steps": len(steps), "agreeing": len(rows), "steps": rows}
    (run / "vllm-iterations.json").write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "steps"}))


if __name__ == "__main__":
    main()
