#!/usr/bin/env bash
# End-to-end offline case study #3: Deterministic Reconciliation Policy Override (no HF weights required).
set -Eeuo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$CASE_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TASK="$CASE_DIR/task"

export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
FOUNDRY=("$PYTHON_BIN" -m hardonian_foundry.cli)

mkdir -p "$TASK/keys" && chmod 700 "$TASK/keys"

# 1. Generate deterministic synthetic dataset & spec (pins policy_predict.py SHA-256).
"$PYTHON_BIN" "$CASE_DIR/generate_dataset.py" --out-dir "$TASK"

# 2. Produce untrusted-suggestion baseline and deterministic policy-override predictions.
"$PYTHON_BIN" "$CASE_DIR/baseline_predict.py" --task-dir "$TASK" --out "$TASK/baseline.jsonl"
"$PYTHON_BIN" "$CASE_DIR/policy_predict.py" --task-dir "$TASK" --out "$TASK/model.jsonl"

# 3. Validate dataset & execute Foundry task-mode run.
"${FOUNDRY[@]}" refinery validate-dataset --task-spec "$TASK/spec.json" > "$TASK/validate.json"
RUN_JSON="$TASK/run.json"
"${FOUNDRY[@]}" refinery run --task-spec "$TASK/spec.json" \
  --manifest "$TASK/manifest.json" \
  --baseline-predictions "$TASK/baseline.jsonl" \
  --model-predictions "$TASK/model.jsonl" \
  --output-root "$TASK/runs" > "$RUN_JSON"

RUN_ID=$("$PYTHON_BIN" -c "import json, sys; print(json.load(open(sys.argv[1]))['run_id'])" "$RUN_JSON")
echo "RUN_ID=$RUN_ID"
"${FOUNDRY[@]}" refinery inspect "$RUN_ID" --output-root "$TASK/runs" > "$TASK/inspect.json"

# 4. Unsigned promotion must fail closed (exit 2).
set +e
"${FOUNDRY[@]}" refinery promote "$RUN_ID" --output-root "$TASK/runs" > "$TASK/promote_blocked.json"
BLOCKED_EXIT=$?
set -e
[ "$BLOCKED_EXIT" -eq 2 ] || { echo "FATAL: unsigned promote exited $BLOCKED_EXIT, expected 2"; exit 1; }

# 5. Generate signing key (if absent), sign run, and promote against keyring.
if [ ! -f "$TASK/keys/sign.pem" ]; then
  "${FOUNDRY[@]}" refinery keygen --out "$TASK/keys/sign.pem" > "$TASK/keys/keygen.json"
fi
PUB=$("$PYTHON_BIN" -c "import sys; from hardonian_foundry import signing; print(signing.public_key_hex(sys.argv[1]))" "$TASK/keys/sign.pem")
cat > "$TASK/keyring.json" <<KEYRING
{"trusted_keys": [{"label": "reconciliation-policy-case-study", "public_key_hex": "$PUB"}]}
KEYRING
"${FOUNDRY[@]}" refinery sign "$RUN_ID" --key-file "$TASK/keys/sign.pem" --output-root "$TASK/runs" > "$TASK/sign.json"
"${FOUNDRY[@]}" refinery promote "$RUN_ID" --output-root "$TASK/runs" --keyring "$TASK/keyring.json" > "$TASK/promote_final.json"
echo "=== reconciliation-policy case study complete ==="
