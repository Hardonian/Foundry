import argparse
import json
import sys
from pathlib import Path

from .baseline import dataset, evaluate, evaluate_agents, reconciliation, verify_trace


def _load_json_file(path_str, description):
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"{description} file not found: {path_str}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {description} file {path_str}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(
        prog="foundry",
        description="Offline synthetic review-gated baseline; not a trained model",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("demo")
    sub.add_parser("eval")
    sub.add_parser("policy-demo")

    provider = sub.add_parser("provider", help="opt-in synthetic loopback provider probe")
    provider.add_argument("--origin", required=True, help="numeric loopback HTTP origin with port")
    provider.add_argument("--model", required=True)

    registry = sub.add_parser("registry")
    registry.add_argument("operation", choices=["list", "promote"])
    registry.add_argument("model_id", nargs="?", default=None, help="optional model or run ID to evaluate for promotion")
    registry.add_argument("--output-root", default=None, help="path to runs directory")

    refinery = sub.add_parser("refinery", help="developer-facing refinery workflow")
    ref_sub = refinery.add_subparsers(dest="refinery_action", required=True)

    val_parser = ref_sub.add_parser("validate-dataset", help="validate a dataset against synthetic constraints")
    val_parser.add_argument("--data", default=None, help="path to JSON dataset file (default: built-in synthetic dataset)")
    val_parser.add_argument("--task-spec", default=None, help="path to task spec JSON (task mode: dataset resolved from spec)")

    pre_parser = ref_sub.add_parser("preflight", help="execute training preflight checks")
    pre_parser.add_argument("--data", default=None, help="path to JSON dataset file")
    pre_parser.add_argument("--manifest", default=None, help="path to model licensing manifest JSON")
    pre_parser.add_argument("--output-root", default=None, help="path to runs directory")

    run_parser = ref_sub.add_parser("run", help="execute end-to-end refinery vertical slice with evidence tracking")
    run_parser.add_argument("--data", default=None, help="path to JSON dataset file")
    run_parser.add_argument("--manifest", default=None, help="path to model licensing manifest JSON")
    run_parser.add_argument("--output-root", default=None, help="path to runs directory")
    run_parser.add_argument("--seed", type=int, default=42, help="deterministic split seed")
    run_parser.add_argument("--task-spec", default=None, help="task spec JSON (task mode)")
    run_parser.add_argument("--baseline-predictions", default=None, help="JSONL {id,label} baseline predictions (task mode)")
    run_parser.add_argument("--model-predictions", default=None, help="JSONL {id,label} candidate model predictions (task mode)")

    inspect_parser = ref_sub.add_parser("inspect", help="inspect run metadata and verify tamper-evident provenance")
    inspect_parser.add_argument("run_id", help="run identifier to inspect")
    inspect_parser.add_argument("--output-root", default=None, help="path to runs directory")

    promote_parser = ref_sub.add_parser("promote", help="evaluate fail-closed promotion gate for a tracked run")
    promote_parser.add_argument("run_id", help="run identifier to evaluate for promotion")
    promote_parser.add_argument("--output-root", default=None, help="path to runs directory")
    promote_parser.add_argument("--keyring", default=None, help="trusted keyring JSON for signature verification")

    license_parser = ref_sub.add_parser("attach-license", help="bind a licensing manifest decision into the run's evidence ledger")
    license_parser.add_argument("run_id", help="run identifier")
    license_parser.add_argument("--manifest", required=True, help="path to model licensing manifest JSON")
    license_parser.add_argument("--output-root", default=None, help="path to runs directory")

    sign_parser = ref_sub.add_parser("sign", help="sign the ledger head, binding a promotion attestation to the run")
    sign_parser.add_argument("run_id", help="run identifier")
    sign_parser.add_argument("--key-file", required=True, help="Ed25519 private key PEM (mode 0600)")
    sign_parser.add_argument("--output-root", default=None, help="path to runs directory")

    keygen_parser = ref_sub.add_parser("keygen", help="generate a new Ed25519 signing key (private key file must not exist)")
    keygen_parser.add_argument("--out", required=True, help="output path for the private key PEM")

    args = parser.parse_args()

    if args.command == "doctor":
        result = {"status": "ok", "runtime": "offline_cpu", "production_ready": False}
    elif args.command == "demo":
        result = {
            "reconciliation": reconciliation(dataset()[0]),
            "agent_allowed_proposal": verify_trace({
                "id": "synthetic-allow",
                "tool": "read_record",
                "arguments": {"record_id": "synthetic-record"},
            }),
            "agent_denied": verify_trace({
                "id": "synthetic-deny",
                "tool": "shell",
                "arguments": {},
            }),
            "evaluation": evaluate(),
        }
    elif args.command == "eval":
        result = {"reconciliation": evaluate(), "agent_verifier": evaluate_agents()}
    elif args.command == "policy-demo":
        from .synthetic_policy import assess
        result = assess({
            "id": "synthetic-policy-demo",
            "expected_minor": 100,
            "actual_minor": None,
            "expected_currency": "USD",
            "actual_currency": "USD",
        })
    elif args.command == "provider":
        from .provider import suggest
        try:
            result = suggest(dataset()[0], origin=args.origin, model=args.model)
        except ValueError as exc:
            print(f"Provider configuration error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if result["status"] == "UNAVAILABLE" else 0
    elif args.command == "registry":
        if args.operation == "list":
            result = {"models": [], "message": "No model registry is configured."}
        else:
            if args.model_id:
                from .refinery import evaluate_promotion
                promo = evaluate_promotion(args.model_id, output_root=args.output_root)
                print(json.dumps(promo, indent=2, sort_keys=True))
            print(
                "Promotion blocked: signed provenance, license, security and benchmark gates are not implemented.",
                file=sys.stderr,
            )
            return 2
    elif args.command == "refinery":
        from . import signing
        from .refinery import (
            attach_license,
            create_run,
            create_task_run,
            evaluate_promotion,
            inspect_run,
            load_keyring,
            run_preflight,
            sign_run,
            validate_dataset,
        )
        from .task_eval import (
            load_jsonl,
            load_predictor_dataset,
            validate_spec,
            validate_task_dataset,
        )
        try:
            if args.refinery_action == "validate-dataset":
                if args.task_spec:
                    spec = _load_json_file(args.task_spec, "task spec")
                    spec_errors = validate_spec(spec)
                    if spec_errors:
                        raise ValueError(f"invalid task spec: {'; '.join(spec_errors)}")
                    base_dir = Path(args.task_spec).resolve().parent
                    task_records, _ = load_predictor_dataset(spec, base_dir)
                    result = validate_task_dataset(task_records, spec)
                else:
                    records = _load_json_file(args.data, "dataset") if args.data else dataset()
                    result = validate_dataset(records)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0 if result["status"] == "VALID" else 1

            elif args.refinery_action == "preflight":
                records = _load_json_file(args.data, "dataset") if args.data else None
                manifest = _load_json_file(args.manifest, "manifest") if args.manifest else None
                result = run_preflight(dataset=records, output_root=args.output_root, model_manifest=manifest)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0 if result["status"] == "PASSED" else 1

            elif args.refinery_action == "run":
                if args.task_spec:
                    if not (args.baseline_predictions and args.model_predictions):
                        raise ValueError("task-mode run requires --baseline-predictions and --model-predictions")
                    spec = _load_json_file(args.task_spec, "task spec")
                    spec_errors = validate_spec(spec)
                    if spec_errors:
                        raise ValueError(f"invalid task spec: {'; '.join(spec_errors)}")
                    base_dir = Path(args.task_spec).resolve().parent
                    task_records, _ = load_predictor_dataset(spec, base_dir)
                    baseline_preds = load_jsonl(args.baseline_predictions)
                    model_preds = load_jsonl(args.model_predictions)
                    manifest = _load_json_file(args.manifest, "manifest") if args.manifest else None
                    result = create_task_run(
                        task_spec=spec,
                        dataset_records=task_records,
                        baseline_predictions=baseline_preds,
                        model_predictions=model_preds,
                        output_root=args.output_root,
                        model_manifest=manifest,
                    )
                else:
                    records = _load_json_file(args.data, "dataset") if args.data else None
                    manifest = _load_json_file(args.manifest, "manifest") if args.manifest else None
                    result = create_run(
                        output_root=args.output_root,
                        dataset=records,
                        model_manifest=manifest,
                        seed=args.seed,
                    )
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0

            elif args.refinery_action == "inspect":
                result = inspect_run(args.run_id, output_root=args.output_root)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0 if result["provenance_status"] == "PROVENANCE_VERIFIED" else 1

            elif args.refinery_action == "promote":
                keyring = load_keyring(args.keyring) if args.keyring else None
                result = evaluate_promotion(args.run_id, output_root=args.output_root, keyring=keyring)
                print(json.dumps(result, indent=2, sort_keys=True))
                if result["promoted"]:
                    print(
                        "Promotion ALLOWED (review required): evidence gates verified at the signed ledger head.",
                        file=sys.stderr,
                    )
                    return 0
                missing = result["missing_evidence"]
                print(
                    f"Promotion blocked: missing required evidence ({', '.join(missing)}).",
                    file=sys.stderr,
                )
                return 2

            elif args.refinery_action == "attach-license":
                manifest = _load_json_file(args.manifest, "manifest")
                result = attach_license(args.run_id, manifest=manifest, output_root=args.output_root)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0 if result["decision"] == "ALLOWED" else 1

            elif args.refinery_action == "sign":
                result = sign_run(args.run_id, private_key_path=args.key_file, output_root=args.output_root)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0

            elif args.refinery_action == "keygen":
                result = signing.generate_keypair(args.out)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"Refinery error: {exc}", file=sys.stderr)
            return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
