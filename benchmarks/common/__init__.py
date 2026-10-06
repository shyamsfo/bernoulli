"""Shared harness used by every benchmark in `benchmarks/`.

Three modules:

- `dataset` — the common `BenchmarkExample` type that every per-benchmark
  loader produces. Loaders live in `benchmarks.<category>.<name>` and
  expose `load(limit=None) -> Iterator[BenchmarkExample]`.
- `metrics` — accuracy, macro-F1, ECE (10 and 15 bin), Brier, NLL,
  coverage curves, stability score. Pure numpy, no scikit-learn.
- `baselines` — the four cross-benchmark baselines wired once and reused:
  Bernoulli-via-HTTP, same-model generative, DeBERTa-v3-zeroshot,
  BGE-m3 + logistic regression. Fine-tuned ceiling baselines are
  per-benchmark and live in each benchmark's subfolder.
"""
