import argparse
import json
import sys
from .baseline import dataset, digest, evaluate, reconciliation, split, verify_trace


def main():
    parser = argparse.ArgumentParser(prog="foundry", description="Offline synthetic review-gated baseline; not a trained model")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor"); sub.add_parser("demo"); sub.add_parser("eval"); sub.add_parser("policy-demo")
    registry = sub.add_parser("registry"); registry.add_argument("operation", choices=["list", "promote"])
    args = parser.parse_args()
    if args.command == "doctor": result = {"status":"ok", "runtime":"offline_cpu", "production_ready":False}
    elif args.command == "demo":
        result = {"reconciliation":reconciliation(dataset()[0]), "agent_allowed_proposal":verify_trace({"id":"synthetic-allow", "tool":"read_record", "arguments":{"record_id":"synthetic-record"}}), "agent_denied":verify_trace({"id":"synthetic-deny", "tool":"shell", "arguments":{}}), "evaluation":evaluate()}
    elif args.command == "eval": result = evaluate()
    elif args.command == "policy-demo":
        from .synthetic_policy import assess
        result = assess({"id": "synthetic-policy-demo", "expected_minor": 100,
                         "actual_minor": None, "expected_currency": "USD", "actual_currency": "USD"})
    elif args.operation == "list": result = {"models":[], "message":"No model registry is configured."}
    else:
        print("Promotion blocked: signed provenance, license, security and benchmark gates are not implemented.", file=sys.stderr); return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0

if __name__ == "__main__": sys.exit(main())
