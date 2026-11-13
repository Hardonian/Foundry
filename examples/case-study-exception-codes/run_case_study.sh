#!/usr/bin/env bash
# End-to-end case study: train -> finalize spec -> predict -> Foundry signed promotion.
# Deterministic inputs; every step logged. Adapter, runs, and keys stay local (gitignored).
# Optional overrides:
#   PROFILE=standard|adversarial-noisy|imbalanced  SEED=20261110
#   TASK_DIR=... (defaults to $CASE_DIR/task; override to avoid touching committed artifacts)
#   REPLAY_ONLY=1 (skip training and replay/verify existing committed signed evidence)
set -Eeuo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$CASE_DIR/../.." && pwd)"
VENV="${VENV:-$HOME/venvs/foundry-train/bin}"
PYTHON_BIN="${PYTHON_BIN:-$VENV/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

# NOTE: deliberate hard default — override FOUNDRY_CASE_HF_HOME if the verified cache moves.
export HF_HOME="${FOUNDRY_CASE_HF_HOME:-/home/scott/hf}"
TASK="${TASK_DIR:-$CASE_DIR/task}"
PROFILE="${PROFILE:-standard}"
SEED="${SEED:-20261110}"
LOG="$CASE_DIR/case-study.log"
THREADS="${THREADS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FOUNDRY=("$PYTHON_BIN" -m hardonian_foundry.cli)

if [ "${REPLAY_ONLY:-0}" = "1" ]; then
  RUN_ID="${RUN_ID:-run-20261111T031107Z-6a0895d1}"
  "${FOUNDRY[@]}" refinery inspect "$RUN_ID" --output-root "$TASK/runs"
  "${FOUNDRY[@]}" refinery promote "$RUN_ID" --output-root "$TASK/runs" --keyring "$TASK/keyring.json"
  exit 0
fi

mkdir -p "$TASK/keys" && chmod 700 "$TASK/keys"
exec > >(tee -a "$LOG") 2>&1
echo "=== case study start $(date -u +%FT%TZ) profile=$PROFILE seed=$SEED ==="

if [ ! -f "$TASK/data.jsonl" ] || [ "$PROFILE" != "standard" ] || [ "$SEED" != "20261110" ]; then
  "$PYTHON_BIN" "$CASE_DIR/generate_dataset.py" --out-dir "$TASK" --profile "$PROFILE" --seed "$SEED"
  "$PYTHON_BIN" "$CASE_DIR/baseline_predict.py" --task-dir "$TASK" --out "$TASK/baseline.jsonl"
fi

# 1. Train the adapter on the frozen train split (fresh adapter each run).
rm -rf "$TASK/adapter"
"$PYTHON_BIN" "$CASE_DIR/train_lora.py" --hf-home "$HF_HOME" --task-dir "$TASK" \
  --output "$TASK/adapter" --epochs 3 --batch-size "$BATCH_SIZE" --grad-accum 8 --threads "$THREADS"

# 2. Bind adapter provenance into the frozen spec (refuses if already finalized).
if grep -q '"0000000000' "$TASK/spec.json" 2>/dev/null || grep -q '0000000000000000' "$TASK/spec.json"; then
  "$PYTHON_BIN" "$CASE_DIR/finalize_spec.py" --task-dir "$TASK" --adapter-dir "$TASK/adapter"
else
  echo "spec already finalized; verifying adapter hash binding:"
  "$PYTHON_BIN" "$CASE_DIR/finalize_spec.py" --task-dir "$TASK" --adapter-dir "$TASK/adapter" --verify
fi

# 3. Score the frozen test split with the trained adapter (batched greedy, offline).
"$PYTHON_BIN" "$CASE_DIR/model_predict.py" --hf-home "$HF_HOME" --task-dir "$TASK" \
  --adapter-dir "$TASK/adapter" --out "$TASK/model.jsonl" --batch-size "$BATCH_SIZE" --threads "$THREADS"

# 4. Foundry pipeline: validate -> preflight -> run -> inspect -> promote (must BLOCK) -> license
#    -> keygen -> sign -> promote (must ALLOW with review required).
"${FOUNDRY[@]}" refinery validate-dataset --task-spec "$TASK/spec.json" > "$TASK/validate.json"
"${FOUNDRY[@]}" refinery preflight --task-spec "$TASK/spec.json" --output-root "$TASK/runs" > "$TASK/preflight.json"
RUN_JSON="$TASK/run.json"
"${FOUNDRY[@]}" refinery run --task-spec "$TASK/spec.json" \
  --baseline-predictions "$TASK/baseline.jsonl" --model-predictions "$TASK/model.jsonl" \
  --output-root "$TASK/runs" > "$RUN_JSON"
RUN_ID=$("$PYTHON_BIN" -c "import json, sys; print(json.load(open(sys.argv[1]))['run_id'])" "$RUN_JSON")
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

if [ ! -f "$TASK/keys/sign.pem" ]; then
  "${FOUNDRY[@]}" refinery keygen --out "$TASK/keys/sign.pem" > "$TASK/keys/keygen.json"
fi
PUB=$("$PYTHON_BIN" "$CASE_DIR/show_pubkey.py" "$TASK/keys/sign.pem")
cat > "$TASK/keyring.json" <<KEYRING
{"trusted_keys": [{"label": "case-study", "public_key_hex": "$PUB"}]}
KEYRING
"${FOUNDRY[@]}" refinery sign "$RUN_ID" --key-file "$TASK/keys/sign.pem" \
  --output-root "$TASK/runs" > "$TASK/sign.json"

"${FOUNDRY[@]}" refinery promote "$RUN_ID" --output-root "$TASK/runs" \
  --keyring "$TASK/keyring.json" > "$TASK/promote_final.json"
echo "=== case study complete: $(grep -o '"status": "[A-Z_]*"' "$TASK/promote_final.json" | head -1) ==="
