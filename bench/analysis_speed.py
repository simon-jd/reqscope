"""Time `reqscope blame` on a capture: wall time and peak memory per run.

    python bench/analysis_speed.py results/raw/interference-s5/capture --repeat 3 > results/analysis-speed.json

Each run is a separate `reqscope blame` process (REQSCOPE_BIN, default
build/reqscope); its peak resident memory comes from wait4. The output's
SHA-256 shows every run produced the same report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def once(binary: Path, capture: Path) -> dict:
    with tempfile.TemporaryFile() as output:
        started = time.monotonic()
        process = subprocess.Popen([str(binary), "blame", str(capture)], stdout=output)
        _, status, usage = os.wait4(process.pid, 0)
        wall = time.monotonic() - started
        if os.waitstatus_to_exitcode(status) != 0:
            raise SystemExit(f"reqscope blame exited with {os.waitstatus_to_exitcode(status)}")
        output.seek(0)
        digest = hashlib.file_digest(output, "sha256").hexdigest()
    return {"wall_s": round(wall, 2), "max_rss_mib": round(usage.ru_maxrss / 1024), "output_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    binary = Path(os.environ.get("REQSCOPE_BIN", ROOT / "build" / "reqscope"))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    runs = [once(binary, args.capture) for _ in range(args.repeat)]
    size = sum(f.stat().st_size for f in args.capture.rglob("*") if f.is_file())
    print(json.dumps({"command": "reqscope blame", "capture": str(args.capture), "capture_bytes": size,
                      "commit": commit, "host_load1": os.getloadavg()[0],
                      "median_wall_s": statistics.median(r["wall_s"] for r in runs),
                      "median_max_rss_mib": statistics.median(r["max_rss_mib"] for r in runs),
                      "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
