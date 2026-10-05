"""Evaluation harness.

See vision_and_roadmap.md §7 Phase 2. The harness compares four configs
(raw / debiased / calibrated / generative-baseline) across a set of public
datasets.

- `evals.example.EvalExample` — the common (state, question, gold) format
- `evals.metrics` — accuracy, macro-F1, ECE (15 bins), Brier, NLL, latency
- `evals.datasets.*` — per-dataset loaders producing EvalExample streams
- `evals.run_eval` — driver: iterate dataset x config, compute metrics, write report
"""
