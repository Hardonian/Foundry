## What changed and why

Root cause / motivation:

## Verification (required — claims without command output are not accepted)

- [ ] `PYTHONPATH=src python3 -m unittest discover -s tests` passes
- [ ] `PYTHONPATH=src python3 -m hardonian_foundry.cli doctor && ... demo` passes
- [ ] Refinery smoke passes (`validate-dataset`, `preflight`, `run`, `inspect`, `promote` still fails closed)
- [ ] `ruff check src tests` clean on touched files
- [ ] CI is green on the exact head commit

## Rollback path

How to revert cleanly if this regresses:

## Boundaries held

- [ ] No secrets, customer data, or model weights added
- [ ] No silent fallbacks introduced (errors stay loud, gates stay fail-closed)
- [ ] README "Honest status" still accurate (updated if not)
