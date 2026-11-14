"""promptfoo provider that red-teams the Foundry fail-closed licensing gate.

Stdlib-only; imports the gate from the repository ``src`` tree so the eval runs
without installing the package. The prompt is a JSON envelope
``{"manifest": {...}, "opts": {...}}`` where ``opts`` are keyword arguments for
``hardonian_foundry.licensing.evaluate``. The gate decision is returned as JSON
text for promptfoo assertions. The gate is deterministic and never promotes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def call_api(prompt, options, context):
    """Evaluate one licensing manifest envelope and return the decision as JSON."""
    from hardonian_foundry import licensing

    try:
        envelope = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        return {"output": json.dumps({"decision": "ERROR", "promotion_allowed": False})}
    if not isinstance(envelope, dict):
        return {"output": json.dumps({"decision": "ERROR", "promotion_allowed": False})}
    opts = envelope.get("opts") or {}
    if not isinstance(opts, dict):
        opts = {}
    result = licensing.evaluate(envelope.get("manifest"), **opts)
    return {"output": json.dumps(result)}
