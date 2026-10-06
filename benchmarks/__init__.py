"""External benchmark comparisons for Bernoulli.

See `benchmarks/README.md` for the layout and `product/milestones.md`
M8-M10 for the plan this package implements. Everything here is
outward-facing — publishable, reproducible comparisons against the
competing baselines and the Jev category.

- `benchmarks.common.dataset.BenchmarkExample` — the common loader output
- `benchmarks.common.metrics` — accuracy, macro-F1, ECE (10 + 15 bin), Brier, NLL, coverage, stability
- `benchmarks.common.baselines` — Bernoulli-via-HTTP, same-model generative, DeBERTa-zeroshot, BGE-m3+LR
- `benchmarks.<category>.<name>` — per-benchmark loaders producing BenchmarkExample streams
"""
