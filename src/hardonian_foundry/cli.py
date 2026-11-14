"""Hardonian Foundry CLI: offline baseline, provider gateway, registry, and Refinery workflows.

Exit codes:
  0  Success / verification passed / promotion ALLOWED_REVIEW_REQUIRED
  1  Validation, preflight, inspect, license, or provider health check failed
  2  Promotion BLOCKED, configuration/usage error, or security/runtime failure
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .baseline import dataset, evaluate, evaluate_agents, reconciliation, verify_trace
from .tenancy import TenantContext
from .types import EXIT_BLOCKED_OR_ERROR, EXIT_CHECK_FAILED, EXIT_OK


class _CliArgumentError(ValueError):
    """Raised on argparse parse failures so --json-errors can format usage errors."""


class _FoundryArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _CliArgumentError(message)


def _load_json_file(path_str: str, description: str) -> Any:
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"{description} file not found: {path_str}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {description} file {path_str}: {exc}") from exc


def _sanitize_for_json_output(value: Any) -> Any:
    # Redact only genuinely secret material. public_key_hex / signature_hex /
    # trusted_keys are PUBLIC verification artifacts, not secrets: keygen emits
    # public_key_hex so callers can build a keyring, and promote verifies against
    # it. Redacting them broke keygen -> keyring -> promote (rc=2). Private keys
    # are written to mode-0600 PEM files and never emitted as JSON.
    sensitive_fields = {
        "private_key", "private_key_hex", "private_key_pem", "private_bytes",
        "secret", "secret_key", "password", "token", "api_key",
    }
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in sensitive_fields:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = _sanitize_for_json_output(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_json_output(item) for item in value]
    return value


def _emit_json(payload: Any) -> None:
    print(json.dumps(_sanitize_for_json_output(payload), indent=2, sort_keys=True))


def _emit_error(message: str, *, exc: BaseException | None = None, json_errors: bool, exit_code: int) -> int:
    if json_errors:
        err_payload = {
            "error": message,
            "error_type": type(exc).__name__ if exc is not None else "CliError",
            "exit_code": exit_code,
        }
        print(json.dumps(err_payload, sort_keys=True), file=sys.stderr)
    else:
        print(message, file=sys.stderr)
    return exit_code


def _build_tenant_context(args: argparse.Namespace) -> TenantContext | None:
    tenant_id = getattr(args, "tenant_id", None)
    request_id = getattr(args, "request_id", None)
    if tenant_id is None and request_id is None:
        return None
    if not tenant_id:
        raise ValueError("--tenant-id is required when specifying tenant context")
    return TenantContext(tenant_id=tenant_id, request_id=request_id or "cli-req")


def _resolve_anchor_expectations(args: argparse.Namespace) -> tuple[str | None, int | None]:
    exp_head = getattr(args, "expected_head", None)
    exp_count = getattr(args, "expected_count", None)
    anchor_path = getattr(args, "anchor", None)
    if anchor_path:
        anchor_data = _load_json_file(anchor_path, "anchor")
        if not isinstance(anchor_data, dict):
            raise ValueError("anchor file must contain a JSON object")
        if exp_head is None:
            exp_head = anchor_data.get("head")
        if exp_count is None:
            exp_count = anchor_data.get("count")
    return exp_head, exp_count


def _handle_doctor(args: argparse.Namespace) -> int:
    from . import signing

    result: dict[str, Any] = {"status": "ok", "runtime": "offline_cpu", "production_ready": False}
    if getattr(args, "verbose", False):
        from .refinery import DEFAULT_RUNS_DIR

        py_ok = sys.version_info >= (3, 11)
        result.update({
            "version": __version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "signing_backend_available": signing.backend_available(),
            "signing_backend": signing.SIGNING_BACKEND,
            "default_runs_dir": DEFAULT_RUNS_DIR,
            "checks": [
                {"name": "python_version", "status": "PASSED" if py_ok else "FAILED"},
                {
                    "name": "signing_backend",
                    "status": "PASSED" if signing.backend_available() else "UNAVAILABLE",
                },
            ],
        })
    _emit_json(result)
    return EXIT_OK


def _handle_demo(_args: argparse.Namespace) -> int:
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
    _emit_json(result)
    return EXIT_OK


def _handle_eval(_args: argparse.Namespace) -> int:
    _emit_json({"reconciliation": evaluate(), "agent_verifier": evaluate_agents()})
    return EXIT_OK


def _handle_policy_demo(_args: argparse.Namespace) -> int:
    from .synthetic_policy import assess

    result = assess({
        "id": "synthetic-policy-demo",
        "expected_minor": 100,
        "actual_minor": None,
        "expected_currency": "USD",
        "actual_currency": "USD",
    })
    _emit_json(result)
    return EXIT_OK


def _handle_provider(args: argparse.Namespace, *, json_errors: bool) -> int:
    from .provider import CircuitBreaker, predict_batch, suggest
    from .task_eval import (
        load_jsonl,
        load_predictor_dataset,
        split_task_dataset,
        validate_spec,
    )

    action = getattr(args, "provider_action", None)
    if action == "predict":
        try:
            if args.task_spec:
                spec = _load_json_file(args.task_spec, "task spec")
                spec_errors = validate_spec(spec)
                if spec_errors:
                    raise ValueError(f"invalid task spec: {'; '.join(spec_errors)}")
                base_dir = Path(args.task_spec).resolve().parent
                if args.data:
                    all_records = load_jsonl(args.data)
                else:
                    all_records, _ = load_predictor_dataset(spec, base_dir)
                labels = list(spec["labels"])
                split_choice = args.split or "test"
                if split_choice == "all":
                    records = all_records
                else:
                    records = split_task_dataset(all_records, spec)[split_choice]
            else:
                if not args.data or not args.labels:
                    raise ValueError("provider predict requires either --task-spec or both --data and --labels")
                records = load_jsonl(args.data)
                labels = [lbl.strip() for lbl in args.labels.split(",") if lbl.strip()]

            breaker = CircuitBreaker(failure_threshold=args.failure_threshold)
            summary = predict_batch(
                records,
                labels=labels,
                origin=args.origin,
                model=args.model,
                timeout=args.timeout,
                allow_tls=args.allow_tls,
                allow_remote=args.allow_remote,
                auth_token_env=args.auth_token_env,
                ca_bundle=args.ca_bundle,
                circuit_breaker=breaker,
                fallback_label=args.fallback_label,
                out_path=args.out,
            )
        except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
            return _emit_error(f"Provider configuration error: {exc}", exc=exc, json_errors=json_errors, exit_code=EXIT_BLOCKED_OR_ERROR)

        # Omit verbose predictions list when writing to file to keep CLI output concise, unless no --out
        display = dict(summary)
        if args.out:
            display.pop("predictions", None)
        _emit_json(display)
        if summary["total"] > 0 and summary["predicted_count"] == 0:
            return EXIT_CHECK_FAILED
        return EXIT_OK

    # Default probe / `provider probe`
    if not getattr(args, "origin", None) or not getattr(args, "model", None):
        return _emit_error(
            "Provider configuration error: --origin and --model are required",
            json_errors=json_errors,
            exit_code=EXIT_BLOCKED_OR_ERROR,
        )
    try:
        result = suggest(
            dataset()[0],
            origin=args.origin,
            model=args.model,
            timeout=args.timeout,
            allow_tls=args.allow_tls,
            allow_remote=args.allow_remote,
            auth_token_env=args.auth_token_env,
            ca_bundle=args.ca_bundle,
        )
    except ValueError as exc:
        return _emit_error(f"Provider configuration error: {exc}", exc=exc, json_errors=json_errors, exit_code=EXIT_BLOCKED_OR_ERROR)

    _emit_json(result)
    return EXIT_CHECK_FAILED if result["status"] == "UNAVAILABLE" else EXIT_OK


def _handle_registry(args: argparse.Namespace, *, json_errors: bool) -> int:
    from .refinery import evaluate_promotion, list_runs, load_keyring

    try:
        tenant_ctx = _build_tenant_context(args)
        if args.operation == "list":
            runs = list_runs(output_root=args.output_root, tenant_context=tenant_ctx) if args.output_root else []
            _emit_json({"models": [], "runs": runs, "message": "No model registry is configured."})
            return EXIT_OK

        if not args.model_id:
            return _emit_error(
                "Promotion blocked: signed provenance, license, security and benchmark gates require a run_id.",
                json_errors=json_errors,
                exit_code=EXIT_BLOCKED_OR_ERROR,
            )

        keyring = load_keyring(args.keyring) if args.keyring else None
        exp_head, exp_count = _resolve_anchor_expectations(args)
        promo = evaluate_promotion(
            args.model_id,
            output_root=args.output_root,
            keyring=keyring,
            expected_head=exp_head,
            expected_count=exp_count,
            tenant_context=tenant_ctx,
        )
        _emit_json(promo)
        if promo["promoted"]:
            print(
                "Promotion ALLOWED (review required): evidence gates verified at the signed ledger head.",
                file=sys.stderr,
            )
            return EXIT_OK
        missing = promo["missing_evidence"]
        msg = f"Promotion blocked: missing required evidence ({', '.join(missing)})."
        if json_errors:
            _emit_error(msg, json_errors=True, exit_code=EXIT_BLOCKED_OR_ERROR)
        else:
            print(msg, file=sys.stderr)
        return EXIT_BLOCKED_OR_ERROR
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _emit_error(f"Registry error: {exc}", exc=exc, json_errors=json_errors, exit_code=EXIT_BLOCKED_OR_ERROR)


def _handle_refinery(args: argparse.Namespace, *, json_errors: bool) -> int:
    from . import artifact, signing
    from .refinery import (
        anchor_run,
        attach_license,
        create_run,
        create_task_run,
        diff_runs,
        evaluate_promotion,
        export_bundle,
        inspect_run,
        list_runs,
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
        action = args.refinery_action

        if action == "validate-dataset":
            val_res: Any
            if args.task_spec:
                spec = _load_json_file(args.task_spec, "task spec")
                spec_errors = validate_spec(spec)
                if spec_errors:
                    raise ValueError(f"invalid task spec: {'; '.join(spec_errors)}")
                base_dir = Path(args.task_spec).resolve().parent
                task_records, _ = load_predictor_dataset(spec, base_dir)
                val_res = validate_task_dataset(task_records, spec)
            else:
                records = _load_json_file(args.data, "dataset") if args.data else dataset()
                val_res = validate_dataset(records)
            _emit_json(val_res)
            return EXIT_OK if val_res["status"] == "VALID" else EXIT_CHECK_FAILED

        if action == "preflight":
            manifest = _load_json_file(args.manifest, "manifest") if args.manifest else None
            if args.task_spec:
                spec = _load_json_file(args.task_spec, "task spec")
                spec_errors = validate_spec(spec)
                if spec_errors:
                    raise ValueError(f"invalid task spec: {'; '.join(spec_errors)}")
                base_dir = Path(args.task_spec).resolve().parent
                if args.data:
                    task_records = load_jsonl(args.data)
                else:
                    task_records, _ = load_predictor_dataset(spec, base_dir)
                pre_res = run_preflight(
                    dataset=task_records,
                    output_root=args.output_root,
                    model_manifest=manifest,
                    task_spec=spec,
                )
            else:
                records = _load_json_file(args.data, "dataset") if args.data else None
                pre_res = run_preflight(dataset=records, output_root=args.output_root, model_manifest=manifest)
            _emit_json(pre_res)
            return EXIT_OK if pre_res["status"] == "PASSED" else EXIT_CHECK_FAILED

        if action == "run":
            tenant_ctx = _build_tenant_context(args)
            manifest = _load_json_file(args.manifest, "manifest") if args.manifest else None
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
                ood_preds = load_jsonl(args.ood_predictions) if args.ood_predictions else None
                run_res = create_task_run(
                    task_spec=spec,
                    dataset_records=task_records,
                    baseline_predictions=baseline_preds,
                    model_predictions=model_preds,
                    output_root=args.output_root,
                    model_manifest=manifest,
                    ood_challenge_predictions=ood_preds,
                    tenant_context=tenant_ctx,
                )
            else:
                records = _load_json_file(args.data, "dataset") if args.data else None
                run_res = create_run(
                    output_root=args.output_root,
                    dataset=records,
                    model_manifest=manifest,
                    seed=args.seed,
                    tenant_context=tenant_ctx,
                )
            _emit_json(run_res)
            return EXIT_OK

        if action == "inspect":
            tenant_ctx = _build_tenant_context(args)
            exp_head, exp_count = _resolve_anchor_expectations(args)
            insp_res = inspect_run(
                args.run_id,
                output_root=args.output_root,
                expected_head=exp_head,
                expected_count=exp_count,
                tenant_context=tenant_ctx,
            )
            _emit_json(insp_res)
            return EXIT_OK if insp_res["provenance_status"] == "PROVENANCE_VERIFIED" else EXIT_CHECK_FAILED

        if action == "promote":
            tenant_ctx = _build_tenant_context(args)
            exp_head, exp_count = _resolve_anchor_expectations(args)
            keyring = load_keyring(args.keyring) if args.keyring else None
            promo_res = evaluate_promotion(
                args.run_id,
                output_root=args.output_root,
                keyring=keyring,
                expected_head=exp_head,
                expected_count=exp_count,
                tenant_context=tenant_ctx,
            )
            _emit_json(promo_res)
            if promo_res["promoted"]:
                print(
                    "Promotion ALLOWED (review required): evidence gates verified at the signed ledger head.",
                    file=sys.stderr,
                )
                return EXIT_OK
            missing = promo_res["missing_evidence"]
            msg = f"Promotion blocked: missing required evidence ({', '.join(missing)})."
            if json_errors:
                return _emit_error(msg, json_errors=True, exit_code=EXIT_BLOCKED_OR_ERROR)
            print(msg, file=sys.stderr)
            return EXIT_BLOCKED_OR_ERROR

        if action == "attach-license":
            manifest = _load_json_file(args.manifest, "manifest")
            attach_res = attach_license(args.run_id, manifest=manifest, output_root=args.output_root)
            _emit_json(attach_res)
            return EXIT_OK if attach_res["decision"] == "ALLOWED" else EXIT_CHECK_FAILED

        if action == "sign":
            sign_res = sign_run(
                args.run_id,
                private_key_path=args.key_file,
                output_root=args.output_root,
                cosign=args.cosign,
            )
            _emit_json(sign_res)
            return EXIT_OK

        if action == "keygen":
            keygen_res = signing.generate_keypair(args.out)
            _emit_json(keygen_res)
            return EXIT_OK

        if action == "anchor":
            anchor_res = anchor_run(args.run_id, output_root=args.output_root, anchor_path=args.out)
            _emit_json(anchor_res)
            return EXIT_OK

        if action == "list":
            tenant_ctx = _build_tenant_context(args)
            runs = list_runs(
                output_root=args.output_root,
                tenant_context=tenant_ctx,
                status=args.status,
                task_id=args.task_id,
                verify_provenance=args.verify,
            )
            _emit_json({"runs": runs, "count": len(runs)})
            return EXIT_OK

        if action == "diff":
            tenant_ctx = _build_tenant_context(args)
            diff_res = diff_runs(
                args.run_id_a,
                args.run_id_b,
                output_root=args.output_root,
                tenant_context=tenant_ctx,
            )
            _emit_json(diff_res)
            return EXIT_OK

        if action == "export-bundle":
            tenant_ctx = _build_tenant_context(args)
            bundle_res = export_bundle(
                args.run_id,
                output_root=args.output_root,
                dest_path=args.out,
                fmt=args.format,
                tenant_context=tenant_ctx,
            )
            _emit_json(bundle_res)
            return EXIT_OK

        if action == "artifact":
            art_action = args.artifact_action
            if art_action == "add":
                add_res = artifact.add_artifact(
                    args.file,
                    store_dir=args.store_dir,
                    output_root=args.output_root,
                    name=args.name,
                    kind=args.kind,
                    run_id=args.run_id,
                )
                _emit_json(add_res)
                return EXIT_OK
            if art_action == "verify":
                ver_res = artifact.verify_artifact(
                    args.target,
                    expected_sha256=args.expected_sha256,
                    store_dir=args.store_dir,
                    output_root=args.output_root,
                )
                _emit_json(ver_res)
                return EXIT_OK if ver_res["verified"] else EXIT_CHECK_FAILED
            if art_action == "list":
                items = artifact.list_artifacts(
                    store_dir=args.store_dir,
                    output_root=args.output_root,
                    run_id=args.run_id,
                )
                _emit_json({"artifacts": items, "count": len(items)})
                return EXIT_OK

    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _emit_error(f"Refinery error: {exc}", exc=exc, json_errors=json_errors, exit_code=EXIT_BLOCKED_OR_ERROR)

    return EXIT_BLOCKED_OR_ERROR


def _add_tenant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tenant-id", default=None, help="tenant identifier for fail-closed scoping")
    parser.add_argument("--request-id", default=None, help="request identifier for tenant context")


def _add_anchor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-head", default=None, help="expected 64-char ledger head SHA-256")
    parser.add_argument("--expected-count", type=int, default=None, help="expected ledger entry count")
    parser.add_argument("--anchor", default=None, help="path to external anchor JSON created by `refinery anchor`")


def _add_provider_connection_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--origin", required=required, default=None, help="numeric loopback HTTP origin with port (or HTTPS with --allow-tls)")
    parser.add_argument("--model", required=required, default=None, help="model name")
    parser.add_argument("--timeout", type=float, default=3.0, help="per-request deadline in seconds (<= 10)")
    parser.add_argument("--allow-tls", action="store_true", help="opt-in to HTTPS provider origins")
    parser.add_argument("--allow-remote", action="store_true", help="opt-in to non-loopback HTTPS origins (requires --allow-tls)")
    parser.add_argument("--auth-token-env", default=None, help="env var name holding optional Bearer token")
    parser.add_argument("--ca-bundle", default=None, help="custom CA bundle PEM path for TLS")


def _build_parser() -> _FoundryArgumentParser:
    common = _FoundryArgumentParser(add_help=False)
    common.add_argument("--json-errors", action="store_true", default=argparse.SUPPRESS, help="emit errors as JSON on stderr")

    parser = _FoundryArgumentParser(
        prog="foundry",
        description="Offline synthetic review-gated baseline; not a trained model",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)
    doc = sub.add_parser("doctor", parents=[common], help="environment and self-health check")
    doc.add_argument("--verbose", action="store_true", help="include runtime, version, and signing backend details")

    sub.add_parser("demo", parents=[common], help="deterministic synthetic baseline demo")
    sub.add_parser("eval", parents=[common], help="evaluate synthetic baseline heuristics")
    sub.add_parser("policy-demo", parents=[common], help="deterministic reconciliation safety policy demo")

    # Provider parser supports both `provider --origin ... --model ...` and `provider predict ...`
    provider = sub.add_parser("provider", parents=[common], help="opt-in synthetic loopback provider probe & batch prediction")
    _add_provider_connection_args(provider, required=False)
    prov_sub = provider.add_subparsers(dest="provider_action", required=False)

    probe = prov_sub.add_parser("probe", parents=[common], help="probe provider with one synthetic reconciliation case")
    _add_provider_connection_args(probe, required=True)

    predict = prov_sub.add_parser("predict", parents=[common], help="generate task-mode {id,label} JSONL predictions with UNDETERMINED fallback")
    _add_provider_connection_args(predict, required=True)
    predict.add_argument("--task-spec", default=None, help="path to task spec JSON")
    predict.add_argument("--data", default=None, help="path to task JSONL dataset")
    predict.add_argument("--labels", default=None, help="comma-separated allowlist labels when --task-spec is omitted")
    predict.add_argument("--split", choices=["test", "validation", "train", "all"], default=None, help="partition to predict (default: test with --task-spec)")
    predict.add_argument("--out", default=None, help="output predictions JSONL path")
    predict.add_argument("--fallback-label", default="UNDETERMINED", help="fail-closed fallback label")
    predict.add_argument("--failure-threshold", type=int, default=5, help="circuit breaker failure threshold")

    # Registry parser
    registry = sub.add_parser("registry", parents=[common], help="model registry operations (fail-closed)")
    registry.add_argument("operation", choices=["list", "promote"])
    registry.add_argument("model_id", nargs="?", default=None, help="optional model or run ID to evaluate for promotion")
    registry.add_argument("--output-root", default=None, help="path to runs directory")
    registry.add_argument("--keyring", default=None, help="trusted keyring JSON for signature verification")
    _add_anchor_args(registry)
    _add_tenant_args(registry)

    # Refinery parser
    refinery = sub.add_parser("refinery", parents=[common], help="developer-facing refinery workflow")
    ref_sub = refinery.add_subparsers(dest="refinery_action", required=True)

    val_parser = ref_sub.add_parser("validate-dataset", parents=[common], help="validate a dataset against synthetic constraints")
    val_parser.add_argument("--data", default=None, help="path to JSON dataset file (default: built-in synthetic dataset)")
    val_parser.add_argument("--task-spec", default=None, help="path to task spec JSON (task mode: dataset resolved from spec)")

    pre_parser = ref_sub.add_parser("preflight", parents=[common], help="execute training preflight checks")
    pre_parser.add_argument("--data", default=None, help="path to JSON or JSONL dataset file")
    pre_parser.add_argument("--task-spec", default=None, help="path to task spec JSON")
    pre_parser.add_argument("--manifest", default=None, help="path to model licensing manifest JSON")
    pre_parser.add_argument("--output-root", default=None, help="path to runs directory")

    run_parser = ref_sub.add_parser("run", parents=[common], help="execute end-to-end refinery vertical slice with evidence tracking")
    run_parser.add_argument("--data", default=None, help="path to JSON dataset file")
    run_parser.add_argument("--manifest", default=None, help="path to model licensing manifest JSON")
    run_parser.add_argument("--output-root", default=None, help="path to runs directory")
    run_parser.add_argument("--seed", type=int, default=42, help="deterministic split seed")
    run_parser.add_argument("--task-spec", default=None, help="task spec JSON (task mode)")
    run_parser.add_argument("--baseline-predictions", default=None, help="JSONL {id,label} baseline predictions (task mode)")
    run_parser.add_argument("--model-predictions", default=None, help="JSONL {id,label} candidate model predictions (task mode)")
    run_parser.add_argument("--ood-predictions", default=None, help="optional JSONL OOD challenge predictions")
    _add_tenant_args(run_parser)

    inspect_parser = ref_sub.add_parser("inspect", parents=[common], help="inspect run metadata and verify tamper-evident provenance")
    inspect_parser.add_argument("run_id", help="run identifier to inspect")
    inspect_parser.add_argument("--output-root", default=None, help="path to runs directory")
    _add_anchor_args(inspect_parser)
    _add_tenant_args(inspect_parser)

    promote_parser = ref_sub.add_parser("promote", parents=[common], help="evaluate fail-closed promotion gate for a tracked run")
    promote_parser.add_argument("run_id", help="run identifier to evaluate for promotion")
    promote_parser.add_argument("--output-root", default=None, help="path to runs directory")
    promote_parser.add_argument("--keyring", default=None, help="trusted keyring JSON for signature verification")
    _add_anchor_args(promote_parser)
    _add_tenant_args(promote_parser)

    license_parser = ref_sub.add_parser("attach-license", parents=[common], help="bind a licensing manifest decision into the run's evidence ledger")
    license_parser.add_argument("run_id", help="run identifier")
    license_parser.add_argument("--manifest", required=True, help="path to model licensing manifest JSON")
    license_parser.add_argument("--output-root", default=None, help="path to runs directory")

    sign_parser = ref_sub.add_parser("sign", parents=[common], help="sign the ledger head, binding a promotion attestation to the run")
    sign_parser.add_argument("run_id", help="run identifier")
    sign_parser.add_argument("--key-file", required=True, help="Ed25519 private key PEM (mode 0600)")
    sign_parser.add_argument("--output-root", default=None, help="path to runs directory")
    sign_parser.add_argument("--cosign", action="store_true", help="add a co-signature with a distinct key for M-of-N quorum")

    keygen_parser = ref_sub.add_parser("keygen", parents=[common], help="generate a new Ed25519 signing key (private key file must not exist)")
    keygen_group = keygen_parser.add_mutually_exclusive_group(required=True)
    keygen_group.add_argument("--out", dest="out", help="output path for the private key PEM")
    keygen_group.add_argument("--key-file", dest="out", help="alias for --out")

    anchor_parser = ref_sub.add_parser("anchor", parents=[common], help="emit external ledger checkpoint anchor")
    anchor_parser.add_argument("run_id", nargs="?", default=None, help="optional run_id to verify before anchoring")
    anchor_parser.add_argument("--output-root", default=None, help="path to runs directory")
    anchor_parser.add_argument("--out", default=None, help="optional output path for anchor JSON")

    list_parser = ref_sub.add_parser("list", parents=[common], help="list tracked runs")
    list_parser.add_argument("--output-root", default=None, help="path to runs directory")
    list_parser.add_argument("--status", default=None, help="filter by run status")
    list_parser.add_argument("--task-id", default=None, help="filter by task_id")
    list_parser.add_argument("--verify", action="store_true", help="verify ledger provenance for each listed run")
    _add_tenant_args(list_parser)

    diff_parser = ref_sub.add_parser("diff", parents=[common], help="compare two tracked runs")
    diff_parser.add_argument("run_id_a", help="first run identifier")
    diff_parser.add_argument("run_id_b", help="second run identifier")
    diff_parser.add_argument("--output-root", default=None, help="path to runs directory")
    _add_tenant_args(diff_parser)

    bundle_parser = ref_sub.add_parser("export-bundle", parents=[common], help="export SHA-256-manifested verification bundle")
    bundle_parser.add_argument("run_id", help="run identifier to export")
    bundle_parser.add_argument("--out", required=True, help="destination directory or archive path")
    bundle_parser.add_argument("--format", choices=["dir", "zip", "tar", "tar.gz", "tgz"], default="dir", help="bundle output format")
    bundle_parser.add_argument("--output-root", default=None, help="path to runs directory")
    _add_tenant_args(bundle_parser)

    art_parser = ref_sub.add_parser("artifact", parents=[common], help="content-addressed artifact store operations")
    art_sub = art_parser.add_subparsers(dest="artifact_action", required=True)

    art_add = art_sub.add_parser("add", parents=[common], help="add a file to the SHA-256 artifact store")
    art_add.add_argument("file", help="source artifact file path")
    art_add.add_argument("--output-root", default=None, help="path to runs directory")
    art_add.add_argument("--store-dir", default=None, help="override artifact store directory")
    art_add.add_argument("--name", default=None, help="logical artifact name")
    art_add.add_argument("--kind", default="model_artifact", help="artifact kind")
    art_add.add_argument("--run-id", default=None, help="optional run_id association")

    art_verify = art_sub.add_parser("verify", parents=[common], help="verify an artifact by SHA-256 or file path")
    art_verify.add_argument("target", help="64-char sha256 digest or artifact file path")
    art_verify.add_argument("--expected-sha256", default=None, help="expected 64-char sha256 digest")
    art_verify.add_argument("--output-root", default=None, help="path to runs directory")
    art_verify.add_argument("--store-dir", default=None, help="override artifact store directory")

    art_list = art_sub.add_parser("list", parents=[common], help="list stored artifacts")
    art_list.add_argument("--output-root", default=None, help="path to runs directory")
    art_list.add_argument("--store-dir", default=None, help="override artifact store directory")
    art_list.add_argument("--run-id", default=None, help="filter by run_id")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    json_errors_requested = "--json-errors" in raw_argv
    parser = _build_parser()
    try:
        args = parser.parse_args(raw_argv)
    except _CliArgumentError as exc:
        return _emit_error(f"CLI usage error: {exc}", exc=exc, json_errors=json_errors_requested, exit_code=EXIT_BLOCKED_OR_ERROR)

    json_errors = bool(getattr(args, "json_errors", json_errors_requested))
    handlers = {
        "doctor": lambda: _handle_doctor(args),
        "demo": lambda: _handle_demo(args),
        "eval": lambda: _handle_eval(args),
        "policy-demo": lambda: _handle_policy_demo(args),
        "provider": lambda: _handle_provider(args, json_errors=json_errors),
        "registry": lambda: _handle_registry(args, json_errors=json_errors),
        "refinery": lambda: _handle_refinery(args, json_errors=json_errors),
    }
    return handlers[args.command]()


if __name__ == "__main__":
    sys.exit(main())
