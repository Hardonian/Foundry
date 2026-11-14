"""Module entrypoint allowing `python -m hardonian_foundry` execution."""
from __future__ import annotations

from collections.abc import Sequence

from . import cli


def run(argv: Sequence[str] | None = None) -> int:
    """Execute the Foundry CLI and return its process exit code."""
    return cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(run())
