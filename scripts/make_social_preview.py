#!/usr/bin/env python3
"""Render docs/assets/social-preview.png (1280x640 GitHub social preview).

Pure PIL text rendering — no network, deterministic, reviewable.
Supports Linux, macOS, and Windows font fallback plus dynamic version and test-count
discovery.
Usage: python3 scripts/make_social_preview.py [output.png]
"""
from __future__ import annotations

import argparse
import re
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 640
BG, PANEL, FG, DIM, GREEN, RED, BLUE, AMBER = (
    "#0b0f14",
    "#11161d",
    "#e6edf3",
    "#8b949e",
    "#3fb950",
    "#f85149",
    "#58a6ff",
    "#d29922",
)

MONO_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/CascadiaMono.ttf",
    "C:/Windows/Fonts/cour.ttf",
)
MONO_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/courbd.ttf",
)
SANS_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
SANS_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def load_font(candidates: tuple[str, ...], size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Return the first loadable TrueType font from candidates, else PIL default."""
    for path_str in candidates:
        if Path(path_str).is_file():
            try:
                return ImageFont.truetype(path_str, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def discover_version() -> str:
    """Read Foundry version from package metadata or pyproject.toml."""
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from hardonian_foundry import __version__

        return str(__version__)
    except (ImportError, AttributeError):
        pyproject = REPO_ROOT / "pyproject.toml"
        if pyproject.is_file():
            match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
            if match:
                return match.group(1)
        return "0.3.0"


def discover_test_count() -> int:
    """Count discoverable tests under tests/ without executing them."""
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    tests_dir = REPO_ROOT / "tests"
    if not tests_dir.is_dir():
        return 0
    suite = unittest.defaultTestLoader.discover(str(tests_dir))
    return suite.countTestCases()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render GitHub social preview PNG.")
    parser.add_argument("output", nargs="?", default="docs/assets/social-preview.png")
    parser.add_argument("--version", default=None, help="override rendered version string")
    parser.add_argument("--test-count", type=int, default=None, help="override rendered test count")
    args = parser.parse_args(argv)

    version = args.version or discover_version()
    test_count = args.test_count if args.test_count is not None else discover_test_count()

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8, H], fill=GREEN)

    title = load_font(SANS_BOLD_CANDIDATES, 64)
    tag = load_font(SANS_CANDIDATES, 30)
    dimf = load_font(SANS_CANDIDATES, 22)
    mono = load_font(MONO_CANDIDATES, 22)
    mono_b = load_font(MONO_BOLD_CANDIDATES, 22)

    d.text((56, 48), f"Foundry v{version}", font=title, fill=FG)
    d.text((58, 138), "Evidence-first model lifecycle for AI agents.", font=tag, fill=BLUE)
    d.text(
        (58, 182),
        "Tamper-evident runs. Promotion fails closed. No model shipped on vibes.",
        font=dimf,
        fill=DIM,
    )

    px, py, pw, ph = 58, 246, 1164, 330
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=12, fill=PANEL, outline="#21262d")
    lines = [
        ("$ ", GREEN, "foundry refinery run --output-root .foundry/runs", FG),
        ("", None, '  "run_id": "run-20261111T003908Z-9902e004"', DIM),
        ("", None, '  "status": "COMPLETED"', DIM),
        ("$ ", GREEN, "foundry refinery promote run-20261111T003908Z-9902e004", FG),
        ("", None, '  "promoted": false,', RED),
        ("", None, '  "status": "BLOCKED"', RED),
        ("", None, '  "missing_evidence": [', AMBER),
        ("", None, '    "cryptographic_promotion_signature",', AMBER),
        ("", None, '    "model_licensing_manifest" ]', AMBER),
        ("[exit 2] ", RED, "promotion blocked — by design", DIM),
    ]
    y = py + 22
    for pre, pcol, body, bcol in lines:
        if pre:
            d.text((px + 26, y), pre, font=mono_b, fill=pcol)
            xoff = d.textlength(pre, font=mono_b)
        else:
            xoff = 0
        d.text((px + 26 + xoff, y), body, font=mono, fill=bcol)
        y += 30
    footer = (
        f"Apache-2.0 · offline · CPU-only research baseline · {test_count} tests · CI on every push"
        if test_count > 0
        else "Apache-2.0 · offline · CPU-only research baseline · CI on every push"
    )
    d.text((58, H - 42), footer, font=dimf, fill=DIM)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"wrote {out_path} (v{version}, {test_count} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
