#!/usr/bin/env bash
# End-to-end case study: train -> finalize spec -> predict -> Foundry signed promotion.
# Deterministic inputs; every step logged. Adapter, runs, and keys stay local (gitignored).
set -Eeuo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$CASE_DIR/../.." && pwd)"
VENV="$HOME/venvs/foundry-train/bin"
# NOTE: deliberate hard default — the shell's HF_HOME (/mnt/g/...) is not writable
# for this user; the pinned snapshot lives under /home/scott/hf. Override with
# FOUNDRY_CASE_HF_HOME if the verified cache moves.
export HF_HOME="${FOUNDRY_CASE_HF_HOME:-/home/scott/hf}"
TASK="$CASE_DIR/task"
LOG="$CASE_DIR/case-study.log"
THREADS="${THREADS:-12}"

mkdir -p "$TASK/keys" && chmod 700 "$TASK/keys"
exec > >(tee -a "$LOG") 2>&1
echo "=== case study start $(date -u +%FT%TZ) ==="

# 1. Train the adapter on the frozen train split (fresh adapter each run).
rm -rf "$TASK/adapter"
"$VENV/python" "$CASE_DIR/train_lora.py" --hf-home "$HF_HOME" --task-dir "$TASK" \
  --output "$TASK/adapter" --epochs 3 --grad-accum 8 --threads "$THREADS"

# 2. Bind adapter provenance into the frozen spec (refuses if already finalized).
if grep -q '"0000000000' "$TASK/spec.json" 2>/dev/null || grep -q '0000000000000000' "$TASK/spec.json"; then
  "$VENV/python" "$CASE_DIR/finalize_spec.py" --task-dir "$TASK" --adapter-dir "$TASK/adapter"
else
  echo "spec already finalized; adapter hash must match this run:"
  grep adapter_sha256 "$TASK/spec.json" "$TASK/adapter/train_manifest.json"
fi

# 3. Score the frozen test split with the trained adapter (greedy, offline).
"$VENV/python" "$CASE_DIR/model_predict.py" --hf-home "$HF_HOME" --task-dir "$TASK" \
  --adapter-dir "$TASK/adapter" --out "$TASK/model.jsonl" --threads "$THREADS"

# 4. Foundry pipeline: validate -> run -> inspect -> promote (must BLOCK) -> license
#    -> keygen -> sign -> promote (must ALLOW with review required).
FOUNDRY=("$VENV/python" -m hardonian_foundry.cli)
"${FOUNDRY[@]}" refinery validate-dataset --task-spec "$TASK/spec.json" > "$TASK/validate.json"
RUN_JSON="$TASK/run.json"
"${FOUNDRY[@]}" refinery run --task-spec "$TASK/spec.json" \
  --baseline-predictions "$TASK/baseline.jsonl" --model-predictions "$TASK/model.jsonl" \
  --output-root "$TASK/runs" > "$RUN_JSON"
RUN_ID=$("$VENV/python" -c "import json;print(json.load(open('$RUN_JSON'))['run_id'])")
echo "RUN_ID=$RUN_ID"
"${FOUNDRY[@]}" refinery inspect "$RUN_ID" --output-root "$TASK/runs" > "$TASK/inspect.json"

set +e
"${FOUNDRY[@]}" refinery promote "$RUN_ID" --output-root "$TASK/runs" > "$TASK/promote_blocked.json"
BLOCKED_EXIT=$?
set -e
[ "$BLOCKED_EXIT" -eq 2 ] || { echo "FATAL: unsigned promote exited $BLOCKED_EXIT, expected 2"; exit 1; }

cat > "$TASK/manifest.json" <<'MANIFEST'
{
  "name": "exception-codes-lora",
  "license_id": "Apache-2.0",
  "provenance": "synthetic",
  "data_origin": "synthetic"
}
MANIFEST
"${FOUNDRY[@]}" refinery attach-license "$RUN_ID" --manifest "$TASK/manifest.json" \
  --output-root "$TASK/runs" > "$TASK/attach_license.json"

# v0.2.1 signing needs the optional [signing] extra; install idempotently.
"$VENV/bin/pip" show cryptography >/dev/null 2>&1 || "$VENV/bin/pip" install -q cryptography==50.0.2
if [ ! -f "$TASK/keys/sign.pem" ]; then
  "${FOUNDRY[@]}" refinery keygen --out "$TASK/keys/sign.pem" > "$TASK/keys/keygen.json"
fi
PUB=$("$VENV/python" "$CASE_DIR/show_pubkey.py" "$TASK/keys/sign.pem")
cat > "$TASK/keyring.json" <<KEYRING
{"trusted_keys": [{"label": "case-study", "public_key_hex": "$PUB"}]}
KEYRING
"${FOUNDRY[@]}" refinery sign "$RUN_ID" --key-file "$TASK/keys/sign.pem" \
  --output-root "$TASK/runs" > "$TASK/sign.json"

"${FOUNDRY[@]}" refinery promote "$RUN_ID" --output-root "$TASK/runs" \
  --keyring "$TASK/keyring.json" > "$TASK/promote_final.json"
echo "=== case study complete: $(grep -o '"status": "[A-Z_]*"' "$TASK/promote_final.json" | head -1) ==="
