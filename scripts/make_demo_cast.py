#!/usr/bin/env python3
"""Generate the README demo cast from a live refinery pipeline run.

Reproducible: `python3 scripts/make_demo_cast.py [output.cast]`
Uses the installed `foundry` console script (override with FOUNDRY_BIN=...).
Runs in a throwaway working directory; no network, no model calls, synthetic
only. Long JSON output is elided on-screen with an explicit marker.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WIDTH, HEIGHT, TS_SCALE = 100, 28, 1.0
FOUNDRY = os.environ.get("FOUNDRY_BIN", "foundry")
CAP_HEAD, CAP_TAIL = 3, 7


def run(argv: list[str], cwd: str) -> tuple[str, int]:
    proc = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=120, check=False
    )
    out = proc.stdout + proc.stderr
    return out.rstrip("\n"), proc.returncode


def elide(lines: list[str]) -> list[str]:
    if len(lines) <= CAP_HEAD + CAP_TAIL + 1:
        return lines
    hidden = len(lines) - CAP_HEAD - CAP_TAIL
    marker = f"... ({hidden} lines elided) ..."
    return lines[:CAP_HEAD] + [marker] + lines[-CAP_TAIL:]


def main() -> None:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/refinery-demo.cast")
    steps: list[tuple[str, list[str], int]] = []
    with tempfile.TemporaryDirectory(prefix="foundry-demo-") as cwd:
        for label in ("doctor", "refinery validate-dataset", "refinery preflight"):
            out, rc = run([FOUNDRY, *label.split()], cwd)
            steps.append((f"{FOUNDRY} {label}", elide(out.splitlines()), rc))
        run_argv = [FOUNDRY, "refinery", "run", "--output-root", ".foundry/runs"]
        out, rc = run(run_argv, cwd)
        m = re.search(r'"run_id":\s*"([^"]+)"', out)
        run_id = m.group(1) if m else "unknown"
        steps.append((" ".join(run_argv), elide(out.splitlines()), rc))
        for verb in ("inspect", "promote"):
            argv = [FOUNDRY, "refinery", verb, run_id, "--output-root", ".foundry/runs"]
            out, rc = run(argv, cwd)
            steps.append((" ".join(argv), elide(out.splitlines()), rc))

    events: list[list] = []
    t = 0.5
    for i, (cmd, lines, rc) in enumerate(steps):
        events.append([t, "o", "\x1b[32m$ \x1b[0m" + cmd + "\r\n"]); t += 0.8
        for line in lines:
            events.append([t, "o", line + "\r\n"]); t += 0.045
        if rc != 0:
            tag = "promotion blocked — by design" if i == len(steps) - 1 else f"exit {rc}"
            events.append(
                [t, "o", f"\x1b[31m[exit {rc}] {tag}\x1b[0m\r\n"]
            ); t += 0.2
        events.append([t, "o", "\r\n"]); t += (2.2 if i == len(steps) - 1 else 1.1)

    header = {
        "version": 2,
        "width": WIDTH,
        "height": HEIGHT,
        "timestamp": int(time.time()),
        "idle_time_limit": 3.0,
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color", "LANG": "C.UTF-8"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    print(f"cast: {out_path} duration={t:.1f}s steps={len(steps)}")


if __name__ == "__main__":
    main()
