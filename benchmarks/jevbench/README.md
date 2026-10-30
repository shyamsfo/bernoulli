# JevBench

Thin adapter that lets JevBench's official harness drive Bernoulli's `/v1/decide` endpoint, so our numbers land on their leaderboard under their rules. Getting listed is distribution; being measured on their rules is credibility.

**Current status**: scaffolding only. The three unknowns were resolved by a short web investigation on 2026-10-07; implementation plan below is now concrete.

## What we know (resolved 2026-10-07)

- **Upstream.** JevBench is hosted at [benchmarkheaven.com/jev-models](https://benchmarkheaven.com/jev-models) with the Python harness mirrored on GitHub — several identical copies: [`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench), [`architsinghai2/jevbench`](https://github.com/architsinghai2/jevbench), [`wayfind/jevbench`](https://github.com/wayfind/jevbench). All share the "JevBench v1" harness. Pick one as the canonical pin (we lean `fstandhartinger/jevbench` since that's the one surveyed first; swap if there's a reason to prefer another).
- **Adapter contract.** JevBench ships a Python harness we run ourselves. CLI: `python -m jevbench.cli run --tasks <dataset> --adapter <type> --model <name>`. Four adapter types shipped: `typesafe`, `openai-compatible`, `local`, `gradio`. Dataset is 242 contrastive decisions (v1.0) growing to 534 (v1.2), split into public + held-out cohorts. The harness enforces budget tracking via file-locked ledgers and emits structured JSON.
- **Submission flow.** [benchmarkheaven.com/submit](https://benchmarkheaven.com/submit) — a web form. Required fields: model name, at least one access method (GitHub / HF / public HTTPS API URL), contact email, benchmark selection (JevBench / ImageJevBench / AudioJevBench). Optional: encrypted API key, description. Free queue runs FIFO; paid 48h fast lane exists.
- **Scoring axes.** Intelligence, Calibration, Speed, Cost — weighted equally into a Capability Score. Current v1.4.2.2 leaderboard: Imajev-4B > Plumb-4B > Jev 1.13.0 (TypeSafe) at #3 / 63.29.
- **Our existing metrics line up.** 10-bin ECE, cost per 1k decisions, latency p50 — all produced by `benchmarks/common/{metrics,stability}.py`.

## Our submission approach: self-run first, then submit

Because the harness is self-runnable, we:

1. **Pin and vendor the harness** into `benchmarks/jevbench/upstream/` (git submodule — pinned to a specific commit of `fstandhartinger/jevbench`). Lets us reproduce.
2. **Write a Bernoulli adapter** in `benchmarks/jevbench/adapter.py` that fits one of JevBench's existing adapter slots. Two viable options:
   - **`local` adapter**: subclass JevBench's local-model adapter and shim calls into Bernoulli's `BernoulliHTTP` baseline (POST `/v1/decide`). Cleanest, zero third-party API dependency.
   - **`openai-compatible`**: add an OpenAI-chat-completions compatibility endpoint to `bernoulli.server` (text-and-parse path on top of the already-exposed `/v1/generate`). Lets JevBench use its built-in OpenAI adapter unchanged. More work but reusable for any OpenAI-compatible benchmark.
3. **Run the public 242-decision cohort** locally; capture results in `benchmarks/jevbench/results/<YYYY-MM-DD>.md`. Verify our Capability-Score vector is coherent before submission.
4. **Submit the web form** at benchmarkheaven.com/submit, pointing to our GitHub release + the committed results folder. "Model name: Bernoulli (Qwen2.5-VL-7B)."

The adapter is expected to be a short glue layer (~100 LoC) — our server already produces calibrated distributions and the metric computation lives in JevBench's harness. Most of the work is schema translation (JevBench's task format ↔ our `DecideRequest`) and ensuring the budget-ledger convention matches.

## Divergences to disclose in the submission

- **Backbone**: dev-tier `Qwen2.5-VL-7B-Instruct` (M4e step-up to 32B-AWQ is parked on AWS capacity). Flag so our Capability-Score isn't misread as production-tier.
- **Our pre-existing N=100 academic + use-case suite** is listed on `bernoulli.live/benchmarks.html` and uses the same metric conventions. JevBench is additive, not a replacement.

## Not here

- Adapter does not re-implement Bernoulli; it calls `/v1/decide` over HTTP.
- Adapter does not fit any baselines — JevBench is a comparison harness, not a training set.
- No per-task fine-tuned ceilings here; those stay with the per-benchmark subfolders in `benchmarks/academic/` and `benchmarks/guardrails/`.

## Layout once implemented

```
benchmarks/jevbench/
├── README.md             # this file
├── adapter.py            # BernoulliAdapter — glue to JevBench's local or openai-compatible slot
├── upstream/             # submodule or vendored pin of fstandhartinger/jevbench
├── run.sh                # one-liner that invokes jevbench.cli with our adapter
└── results/
    └── <YYYY-MM-DD>.md   # per-run report; captures model id, revision, Bernoulli commit, submission SHA
```
