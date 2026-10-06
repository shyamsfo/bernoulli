"""M9 use-case coverage: guardrails benchmarks.

Three datasets map to the "Guardrails" use-case card on the landing page:

- `wildguard_test` — WildGuardTest, response-harmfulness binary.
- `toxicchat` — ToxicChat, toxicity binary.
- `xstest` — XSTest, over-refusal detection.

Each benchmark folder owns its own loader, README, and results/.
See `benchmarks/README.md` methodology for metric definitions and
baseline conventions.
"""
