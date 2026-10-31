"""Draw the slow engine steps of the interference run as an SVG timeline.

    python bench/plot_steps.py results/interference-blame.json results/interference-steps.json > docs/slow-steps.svg

One mark per slow step: when its first slow gap began, and the gap's length.
Marks are colored by what blame said: blamed on a long prompt, on other
requests, or unexplained. Shaded bands are the long prompts' prefill windows
(sent until first token), which blame never saw. Standard library only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

W, H = 820, 320
LEFT, RIGHT, TOP, BOTTOM = 56, 16, 56, 44
COLORS = {"long": "#2f6fd6", "other": "#8a63c9", "unexplained": "#e07b28"}
LABELS = {"long": "blamed on a long prompt", "other": "blamed on other requests",
          "unexplained": "unexplained (no co-scheduled prefill)"}


def main() -> None:
    blame = json.loads(Path(sys.argv[1]).read_text())
    steps = json.loads(Path(sys.argv[2]).read_text())
    begin = {}
    for row in blame["requests"]:
        if "decode-" not in row["req_id"]:
            continue
        for gap in row["slow_gaps"]:
            for step in gap["steps"]:
                begin[step["step_id"]] = min(begin.get(step["step_id"], gap["begin_ns"]), gap["begin_ns"])
    windows = steps["long_prompt_windows_ns"]
    origin = min(min(begin.values()), min(a for a, _ in windows.values()))
    end = max(max(begin.values()), max(b for _, b in windows.values()))
    span_s = (end - origin) / 1e9 * 1.04
    top_ms = 60.0

    def x(ns: float) -> float:
        return LEFT + (ns - origin) / 1e9 / span_s * (W - LEFT - RIGHT)

    def y(ms: float) -> float:
        return H - BOTTOM - min(ms, top_ms) / top_ms * (H - TOP - BOTTOM)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="12">',
           '<style>text{fill:#24292f}.axis{stroke:#57606a}.grid{stroke:#d0d7de}.band{fill:#2f6fd6;opacity:.10}'
           '@media (prefers-color-scheme: dark){text{fill:#e6edf3}.axis{stroke:#8b949e}.grid{stroke:#30363d}'
           '.band{opacity:.22}}</style>']
    for name, (a, b) in sorted(windows.items()):
        out.append(f'<rect class="band" x="{x(a):.1f}" y="{TOP}" width="{max(x(b) - x(a), 1.5):.1f}" '
                   f'height="{H - TOP - BOTTOM}"/>')
        out.append(f'<text x="{x(a):.1f}" y="{TOP - 6}" font-size="10">{name}</text>')
    for ms in (0, 10, 20, 30, 40, 50, 60):
        out.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y(ms):.1f}" y2="{y(ms):.1f}"/>')
        out.append(f'<text x="{LEFT - 8}" y="{y(ms) + 4:.1f}" text-anchor="end">{ms}</text>')
    typical = 5.43
    out.append(f'<line class="axis" stroke-dasharray="4 3" x1="{LEFT}" x2="{W - RIGHT}" '
               f'y1="{y(typical):.1f}" y2="{y(typical):.1f}"/>')
    out.append(f'<text x="{LEFT + 6}" y="{y(typical) + 13:.1f}" font-size="10">'
               f'typical decode gap {typical} ms</text>')
    for second in range(0, int(span_s) + 1, 2):
        px = LEFT + second / span_s * (W - LEFT - RIGHT)
        out.append(f'<line class="axis" x1="{px:.1f}" x2="{px:.1f}" y1="{H - BOTTOM}" y2="{H - BOTTOM + 4}"/>')
        out.append(f'<text x="{px:.1f}" y="{H - BOTTOM + 17}" text-anchor="middle">{second}</text>')
    out.append(f'<line class="axis" x1="{LEFT}" x2="{W - RIGHT}" y1="{H - BOTTOM}" y2="{H - BOTTOM}"/>')
    out.append(f'<text x="{(LEFT + W - RIGHT) / 2:.0f}" y="{H - 8}" text-anchor="middle">'
               'seconds into the run (slow steps of the 16 lockstep decodes)</text>')
    out.append(f'<text transform="translate(14 {(TOP + H - BOTTOM) / 2:.0f}) rotate(-90)" '
               'text-anchor="middle">inter-token gap, ms</text>')
    for row in steps["steps"]:
        kind = "long" if row["blamed_long_prompts"] else "other" if row["blamed_other_requests"] else "unexplained"
        out.append(f'<circle cx="{x(begin[row["step_id"]]):.1f}" cy="{y(row["gap_ms_median"]):.1f}" r="4.5" '
                   f'fill="{COLORS[kind]}" fill-opacity=".85"><title>step {row["step_id"]}: '
                   f'{row["gap_ms_median"]} ms gap, GPU busy {row["gpu_busy_ms"]} ms, {LABELS[kind]}</title></circle>')
    lx = LEFT + 10
    for kind in ("long", "unexplained", "other"):
        out.append(f'<circle cx="{lx}" cy="14" r="4.5" fill="{COLORS[kind]}"/>')
        out.append(f'<text x="{lx + 9}" y="18">{LABELS[kind]}</text>')
        lx += 9 + 7 * len(LABELS[kind]) + 24
    out.append("</svg>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
