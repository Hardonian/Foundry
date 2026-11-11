#!/usr/bin/env python3
"""Render docs/assets/social-preview.png (1280x640 GitHub social preview).

Pure PIL text rendering — no network, deterministic, reviewable.
Usage: python3 scripts/make_social_preview.py [output.png]
"""
from __future__ import annotations

import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
BG, PANEL, FG, DIM, GREEN, RED, BLUE, AMBER = (
    "#0b0f14", "#11161d", "#e6edf3", "#8b949e", "#3fb950", "#f85149", "#58a6ff", "#d29922",
)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/social-preview.png"
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8, H], fill=GREEN)

    title = ImageFont.truetype(SANS_B, 64)
    tag = ImageFont.truetype(SANS, 30)
    dimf = ImageFont.truetype(SANS, 22)
    mono = ImageFont.truetype(MONO, 22)
    mono_b = ImageFont.truetype(MONO_B, 22)

    d.text((56, 48), "Foundry", font=title, fill=FG)
    d.text((58, 138), "Evidence-first model lifecycle for AI agents.", font=tag, fill=BLUE)
    d.text((58, 182), "Tamper-evident runs. Promotion fails closed. No model shipped on vibes.",
           font=dimf, fill=DIM)

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
    d.text((58, H - 42), "Apache-2.0 · offline · CPU-only research baseline · 86 tests · CI on every push",
           font=dimf, fill=DIM)
    img.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
