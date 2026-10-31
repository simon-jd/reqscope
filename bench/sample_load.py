"""Sample the host's load average every 5 s into a CSV, for runs on a shared host.

    python bench/sample_load.py load.csv &

Columns: CLOCK_MONOTONIC ns (the clock of client.json), Unix time, and the
1, 5 and 15 minute load averages. Stops after 12 hours or when the file
load.csv.stop exists. bench/interference.py records the same samples in
client.json itself; this is for runs scored together across a session.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out, stop = Path(sys.argv[1]), Path(sys.argv[1] + ".stop")
    end = time.time() + 12 * 3600
    with out.open("a", buffering=1) as csv:
        while time.time() < end and not stop.exists():
            load = Path("/proc/loadavg").read_text().split()[:3]
            csv.write(f"{time.monotonic_ns()},{int(time.time())},{','.join(load)}\n")
            time.sleep(5)


if __name__ == "__main__":
    main()
