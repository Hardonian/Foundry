"""Artifact regression: installed wheel + sdist fixtures, no checkout or weights.

Run with the clean wheel environment's Python, from the extracted sdist. Negative
controls use disposable copies; the committed signed evidence is never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_ID = "run-20261111T031107Z-6a0895d1"
# Deliberately explicit: adding arbitrary JSON, keys or weights must not expand
# the distribution's fixture contract. Keep in sync with MANIFEST.in.
FIXTURES = (
    "attach_license.json", "baseline.jsonl", "data.jsonl", "inspect.json",
    "inspect_final.json", "keyring.json", "manifest.json", "model.jsonl",
    "predict_manifest.json", "promote_blocked.json", "promote_final.json",
    "run.json", "sign.json", "spec.json", "validate.json",
    "runs/evidence.jsonl", f"runs/{RUN_ID}/run.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def snapshot(task: Path) -> dict[str, str]:
    files = {p.relative_to(task).as_posix(): p for p in task.rglob("*") if p.is_file()}
    require(set(files) == set(FIXTURES), f"Unexpected/missing packaged fixtures: {sorted(set(files) ^ set(FIXTURES))}")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}


def cli(cwd: Path, action: str, runs: Path, expected: int, keyring: Path | None = None) -> dict:
    command = [sys.executable, "-I", "-m", "hardonian_foundry.cli", "refinery", action,
               RUN_ID, "--output-root", str(runs)]
    if keyring is not None:
        command.extend(["--keyring", str(keyring)])
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    require(proc.returncode == expected, f"{command}: exit {proc.returncode}, expected {expected}: {proc.stderr}")
    result = json.loads(proc.stdout)
    print(json.dumps({"command": command, "exit": proc.returncode, "result": result}, sort_keys=True))
    return result


def main() -> None:
    task = Path(__file__).resolve().parents[1] / "examples/case-study-exception-codes/task"
    before = snapshot(task)
    with tempfile.TemporaryDirectory(prefix="foundry-package-replay-") as td:
        scratch = Path(td)
        # -I ignores PYTHONPATH, user site and the checkout. Verify the resolved
        # module belongs to this environment rather than to an editable install.
        origin = subprocess.check_output(
            [sys.executable, "-I", "-c", "import hardonian_foundry; print(hardonian_foundry.__file__)"],
            cwd=scratch, text=True,
        ).strip()
        require(Path(sys.prefix).resolve() in Path(origin).resolve().parents, f"Not an installed wheel: {origin}")
        inspected = cli(scratch, "inspect", task / "runs", 0)
        require(inspected["provenance_status"] == "PROVENANCE_VERIFIED", "Signed fixture provenance failed")
        signed = cli(scratch, "promote", task / "runs", 0, task / "keyring.json")
        require(signed["status"] == "ALLOWED_REVIEW_REQUIRED" and signed["promoted"]
                and signed["review_required"] and not signed["missing_evidence"], "Signed fixture not review-gated")
        require(signed["evidence_summary"]["promotion_signature"] is not None, "Signature not verified")

        unsigned = scratch / "unsigned"
        shutil.copytree(task / "runs", unsigned)
        ledger = unsigned / "evidence.jsonl"
        rows = ledger.read_text(encoding="utf-8").splitlines(keepends=True)
        require(len(rows) == 3, "Expected run, license and signature entries")
        # Preserve the valid run + license prefix; remove only its signature.
        ledger.write_text("".join(rows[:-1]), encoding="utf-8")
        unsigned_inspect = cli(scratch, "inspect", unsigned, 0)
        require(unsigned_inspect["provenance_status"] == "PROVENANCE_VERIFIED", "Unsigned control broke the chain")
        blocked = cli(scratch, "promote", unsigned, 2, task / "keyring.json")
        require(not blocked["promoted"] and "cryptographic_promotion_signature" in blocked["missing_evidence"],
                "Unsigned evidence did not fail closed for the signature")

        mutated = scratch / "mutated"
        shutil.copytree(task / "runs", mutated)
        record = mutated / RUN_ID / "run.json"
        data = json.loads(record.read_text(encoding="utf-8"))
        data["evaluation"]["f1"] = 0.99
        record.write_text(json.dumps(data), encoding="utf-8")
        tampered = cli(scratch, "inspect", mutated, 1)
        require(tampered["provenance_status"] != "PROVENANCE_VERIFIED", "Mutated evidence was trusted")
        blocked = cli(scratch, "promote", mutated, 2, task / "keyring.json")
        require(not blocked["promoted"] and blocked["missing_evidence"], "Mutated evidence did not fail closed")
    require(snapshot(task) == before, "Replay modified the packaged fixture")
    print(json.dumps({"status": "PASS", "installed_module": origin, "fixture_sha256": before}, sort_keys=True))


if __name__ == "__main__":
    main()
