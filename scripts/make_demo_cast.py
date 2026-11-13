#!/usr/bin/env python3
"""Generate the README demo cast from a live task-mode refinery pipeline run.

Reproducible: `python3 scripts/make_demo_cast.py [output.cast]`
Uses the installed `foundry` console script (override with FOUNDRY_BIN=...) or
falls back to `python -m hardonian_foundry.cli` with `PYTHONPATH=src`.
Runs in a throwaway working directory; no network, no model weights, synthetic
only. Long JSON output is elided on-screen with an explicit marker.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 100, 28
CAP_HEAD, CAP_TAIL = 4, 7


def _foundry_cmd() -> tuple[list[str], str]:
    configured = os.environ.get("FOUNDRY_BIN")
    if configured:
        return [configured], Path(configured).name
    if shutil.which("foundry"):
        return ["foundry"], "foundry"
    return [sys.executable, "-m", "hardonian_foundry.cli"], "foundry"


def _sub_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


def run(argv: list[str], cwd: str, env: dict[str, str]) -> tuple[str, int]:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    out = proc.stdout + proc.stderr
    return out.rstrip("\n"), proc.returncode


def elide(lines: list[str]) -> list[str]:
    if len(lines) <= CAP_HEAD + CAP_TAIL + 1:
        return lines
    hidden = len(lines) - CAP_HEAD - CAP_TAIL
    marker = f"... ({hidden} lines elided) ..."
    return lines[:CAP_HEAD] + [marker] + lines[-CAP_TAIL:]


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    out_path = Path(args[0] if args else "docs/assets/refinery-demo.cast")
    base_cmd, display_bin = _foundry_cmd()
    env = _sub_env()
    case_dir = REPO_ROOT / "examples/case-study-agent-tool-gate"
    steps: list[tuple[str, list[str], int, str]] = []

    with tempfile.TemporaryDirectory(prefix="foundry-demo-") as cwd:
        work = Path(cwd)
        task_dir = work / "task"
        keys_dir = work / "keys"
        keys_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

        # Prepare deterministic synthetic task-mode inputs without HF weights.
        for script_name, extra in (
            ("generate_dataset.py", ["--out-dir", str(task_dir)]),
            ("baseline_predict.py", ["--task-dir", str(task_dir), "--out", str(task_dir / "baseline.jsonl")]),
            ("gate_predict.py", ["--task-dir", str(task_dir), "--out", str(task_dir / "model.jsonl")]),
        ):
            subprocess.run(
                [sys.executable, str(case_dir / script_name), *extra],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
        manifest_path = task_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "agent-tool-gate-structural-v1",
                    "license_id": "Apache-2.0",
                    "provenance": "synthetic",
                    "data_origin": "synthetic",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        def invoke(cli_args: list[str], display_args: list[str] | None = None, tag: str = "") -> str:
            out, rc = run([*base_cmd, *cli_args], cwd, env)
            shown = display_args if display_args is not None else cli_args
            steps.append((f"{display_bin} {' '.join(shown)}", elide(out.splitlines()), rc, tag))
            return out

        invoke(["doctor"])
        invoke(
            ["refinery", "validate-dataset", "--task-spec", "task/spec.json"],
        )
        invoke(
            ["refinery", "preflight", "--task-spec", "task/spec.json", "--output-root", ".foundry/runs"],
        )
        run_out = invoke(
            [
                "refinery",
                "run",
                "--task-spec",
                "task/spec.json",
                "--baseline-predictions",
                "task/baseline.jsonl",
                "--model-predictions",
                "task/model.jsonl",
                "--output-root",
                ".foundry/runs",
            ]
        )
        match = re.search(r'"run_id":\s*"([^"]+)"', run_out)
        run_id = match.group(1) if match else "unknown"

        invoke(["refinery", "inspect", run_id, "--output-root", ".foundry/runs"])
        invoke(
            ["refinery", "promote", run_id, "--output-root", ".foundry/runs"],
            tag="promotion blocked — by design",
        )
        invoke(
            [
                "refinery",
                "attach-license",
                run_id,
                "--manifest",
                "task/manifest.json",
                "--output-root",
                ".foundry/runs",
            ]
        )
        invoke(["refinery", "keygen", "--out", "keys/sign.pem"])

        pub_proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from hardonian_foundry import signing; print(signing.public_key_hex(sys.argv[1]))",
                str(keys_dir / "sign.pem"),
            ],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        pub_hex = pub_proc.stdout.strip() if pub_proc.returncode == 0 else "0" * 64
        (task_dir / "keyring.json").write_text(
            json.dumps({"trusted_keys": [{"label": "demo", "public_key_hex": pub_hex}]}, indent=2) + "\n",
            encoding="utf-8",
        )
        invoke(
            [
                "refinery",
                "sign",
                run_id,
                "--key-file",
                "keys/sign.pem",
                "--output-root",
                ".foundry/runs",
            ]
        )
        invoke(
            [
                "refinery",
                "promote",
                run_id,
                "--output-root",
                ".foundry/runs",
                "--keyring",
                "task/keyring.json",
            ]
        )

    events: list[list[object]] = []
    t = 0.5
    for i, (cmd, lines, rc, tag) in enumerate(steps):
        events.append([t, "o", "\x1b[32m$ \x1b[0m" + cmd + "\r\n"])
        t += 0.8
        for line in lines:
            events.append([t, "o", line + "\r\n"])
            t += 0.045
        if rc != 0:
            label = tag or f"exit {rc}"
            events.append([t, "o", f"\x1b[31m[exit {rc}] {label}\x1b[0m\r\n"])
            t += 0.2
        events.append([t, "o", "\r\n"])
        t += 2.0 if i == len(steps) - 1 else 1.0

    header = {
        "version": 2,
        "width": WIDTH,
        "height": HEIGHT,
        "timestamp": int(time.time()),
        "idle_time_limit": 3.0,
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color", "LANG": "C.UTF-8"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(header) + "\n")
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    print(f"cast: {out_path} duration={t:.1f}s steps={len(steps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
