# JevBench

Thin adapter that lets JevBench's official harness drive Bernoulli's `/v1/decide` endpoint, so our numbers land on their leaderboard under their rules. Getting listed is distribution; being measured on their rules is credibility.

**Current status**: scaffolding only. The adapter needs upstream info we don't have in-repo yet — see "What we don't know" below.

## What we know

- Jev lives at [typesafe.ai](https://typesafe.ai/) with System One concept docs at [docs.typesafe.ai/concepts/system-one](https://docs.typesafe.ai/concepts/system-one). This is the category Bernoulli competes in.
- Our metric conventions already line up with JevBench where published: **10-bin ECE** (headline), **cost per 1k decisions**, **stability** (reorder + reword). All three are already produced by `benchmarks/common/{metrics,stability}.py`.
- The eventual output is one `benchmarks/jevbench/results/<YYYY-MM-DD>.md` with the JevBench-mapped numbers plus a submission commit SHA.

## What we don't know (gating implementation)

1. **Upstream repo URL.** The milestone says "vendor or clone JevBench into `benchmarks/jevbench/upstream/`." We need the actual GitHub (or similar) URL. Once pinned, decide between git submodule vs vendored clone based on their license and release cadence.
2. **Adapter contract.** JevBench presumably defines an interface (Python class, HTTP endpoint, or CLI protocol) that a model-under-test implements. Shape unknown — could be:
   - A Python `Model` class with a `predict(prompt, options) -> distribution` method.
   - An HTTP endpoint JevBench POSTs to (which we already have via Bernoulli's `/v1/decide` — minor shape translation only).
   - A per-task CLI entry point.
3. **Submission flow.** Public leaderboard vs. closed submission form vs. PR to a leaderboard file. The method shapes how `benchmarks/jevbench/results/` is laid out and whether submissions are reversible.

Each of these is a short investigation on `typesafe.ai`, not a research project. Captured in `product/milestones.md` M10 as the next action.

## Implementation plan (once the above are answered)

```
benchmarks/jevbench/
├── README.md             # this file
├── adapter.py            # BernoulliAdapter(JevBenchModel) — one method per JevBench task kind
├── upstream/             # vendored JevBench (git submodule or pinned clone)
├── run.py                # thin driver: load JevBench suite, call adapter, write results/
├── submit.py             # formats + submits results per the JevBench flow
└── results/
    └── <YYYY-MM-DD>.md   # one per attempted submission; model/revision/commit pinned
```

The adapter itself is expected to be short — Bernoulli already produces calibrated distributions at an HTTP endpoint and the metric computation is reusable from `benchmarks/common/metrics.py`. Most of the work is schema translation (JevBench's input/output shape ↔ our `DecideRequest`/`DecideResponse`) and matching their reproducibility fields.

## Divergences to call out in the first submission

- **ECE bin count**: JevBench uses 10 bins (we match for the headline; 15-bin remains for continuity with pre-M8 reports).
- **Stability protocol**: we test up to 3 hand-authored reword stems per benchmark + full/sampled option permutations. If JevBench's protocol differs (different reword count, different permutation cap), document the gap in the submission footer.
- **Backbone**: our first attempt is likely on the dev-tier `Qwen2.5-VL-7B-Instruct` since the production-tier `g6e.xlarge` + 32B-AWQ step-up (M4e) is still parked on AWS capacity. Flag this prominently in the submission so numbers aren't misread as production-serving quality.

## Not here

- The adapter does not re-implement Bernoulli; it calls `/v1/decide` over HTTP (same path as the `BernoulliHTTP` baseline).
- The adapter does not fit any baselines — JevBench is a comparison harness, not a training set.
- No per-task fine-tuned ceilings here; those stay with the per-benchmark subfolders in `benchmarks/academic/` and `benchmarks/guardrails/`.
