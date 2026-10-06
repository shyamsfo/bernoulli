# Bernoulli — Usage

Two interfaces, same engine. Pick your path:

- **[HTTP.md](HTTP.md)** — call Bernoulli over HTTP from any language. Server runs on `http://localhost:8000`. Start with `just serve` or `just docker-run`.
- **[CLI.md](CLI.md)** — call Bernoulli from a shell with `uv run bernoulli decide`. No server needed, model is loaded per invocation.

Both guides are organized around the three question types — **choice** (pick one of several), **binary** (yes/no), **rating** (score on an integer scale) — with small worked examples for each.

For realistic end-to-end patterns (support ticket triage, LLM output guardrail, content moderation, code review) and the design pitch for why Bernoulli instead of a prompt+parse pipeline, see **[`USECASES.md`](USECASES.md)**.
