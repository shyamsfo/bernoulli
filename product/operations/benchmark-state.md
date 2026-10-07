# Benchmark state — where we are

Companion to [`benchmark-primer.md`](benchmark-primer.md). The primer explains **what** we measure and **why**. This doc is a dated, append-at-top log of **where we actually are** against those measurements.

Each entry has: numbers table, what's sellable, honest limitations, honest gaps, and what's next.

---

## 2026-10-07 — First academic sweep in; PAWS debias bug fixed

### Numbers (N=100 per benchmark, dev-tier backbone Qwen2.5-VL-7B on A10G)

| Benchmark | Bernoulli acc | Generative acc | BGE-m3+LR acc | Bernoulli NLL | Generative NLL | NLL ratio (B vs G) |
|---|---|---|---|---|---|---|
| SST-2 | 0.95 | 0.95 | 0.90 | **0.18** | 1.38 | 7.6× |
| AG News | **0.79** | 0.72 | 0.57 | **1.05** | 7.76 | 7.4× |
| TweetEval-emotion | 0.77 | 0.76 | 0.80 | **0.63** | 6.63 | 10.5× |
| Banking77 | 0.57 | 0.48 | **0.98** | **1.65** | 13.9 | 8.4× |
| PAWS (post-fix) | **0.83** | 0.82 | 0.56 | **0.41** | 4.97 | **12×** |
| arXiv post-cutoff | — | — | — | — | — | 2-row placeholder; non-finding |

Full per-benchmark reports at `benchmarks/academic/<name>/results/2026-10-07.md`. ECE and stability columns in those files too.

### What's sellable (and strengthened by the PAWS fix)

**The calibration thesis holds uniformly across every real academic benchmark.** Bernoulli NLL is 7–12× better than Generative at comparable or better accuracy. Five independent benchmarks, same story, including PAWS which is adversarial-by-construction.

The one-line pitch — *"same answer, dramatically better calibration"* — has five receipts. Example rows the landing page can safely quote:

- **SST-2**: Bernoulli and Generative both at 0.95; Bernoulli NLL 0.18 vs 1.38 (**7.6× better**).
- **PAWS**: Bernoulli 0.83 vs Generative 0.82; Bernoulli NLL 0.41 vs 4.97 (**12× better**), with 2.4× better ECE on an adversarial dataset.

Also sellable in context: Bernoulli either wins or ties on accuracy against the same-backbone generative baseline on 4 of 5 real benchmarks.

### Honest limitations (disclose, don't hide)

1. **Banking77 BGE-m3+LR dominates at 0.98 vs Bernoulli 0.57.** A trained encoder on 500 labeled examples beats zero-shot approaches on a 77-way intent task. **Not a Bernoulli flaw** — zero-shot ≠ fine-tuned. Honest positioning: *"if you have thousands of labeled examples for a fixed label space, train an encoder; if you don't, or your label space changes, Bernoulli."* The receipt actually supports the pitch when framed correctly.

2. **Banking77 reorder stability = 0.13.** Chunked scoring makes top-1 sensitive to which labels fall in which chunk — architectural (parking-lot item). Any Banking77 citation needs a methodology footnote; cannot be quoted as a stability number without the caveat.

3. **BGE-m3+LR at 500 training examples is uneven.** SST-2 0.90, TweetEval 0.80, AG News 0.57, PAWS 0.56. 500 examples is enough for easy sentiment but not for 4-way topic or adversarial binary. If we cite the "did we need an LLM?" comparison, we have to disclose the training-set-size assumption.

### Honest gaps (what's worse than we want)

1. **No DeBERTa column anywhere.** The strongest open zero-shot encoder is missing from the leaderboard entirely. CPU latency on 4-way+ is pathological (hours per benchmark × multiple benchmarks); GPU is held by the Bernoulli server. **Single biggest gap in the comparability story.** Needs a GPU-scheduling fix or overnight CPU batch.

2. **Zero M9 use-case numbers.** All five loaders built, none swept. The use-case cards on `bernoulli.live` have no numeric backing. **For a product pitched on use cases, this is the gap that matters most.** Academic benchmarks are prerequisite; use-case receipts are what sell product.

3. **N=100 is small.** Headline numbers in a 5% band could swing 2–3 points at N=500. Not misleading in direction but not publishable without the sample-size note. Running at N=500 is a few hours.

4. **arXiv post-cutoff is 2 placeholder rows.** The anti-contamination claim can't be made with fake examples. M8 task 10b (`make_dataset.py` on the real arXiv API) must land before any "model couldn't have memorized" story.

5. **No fine-tuned ceiling for any benchmark.** Each benchmark's README lists a target (`distilbert-sst-2`, `twitter-roberta-emotion`, etc.) — none wired. Can't position Bernoulli against task-specific state-of-the-art.

6. **Dev-tier backbone (Qwen2.5-VL-7B on g5.xlarge).** M4e step-up to 32B-AWQ on g6e.xlarge is parked on capacity. Any serving-latency or accuracy number carries a "dev-tier backbone" caveat.

### Where we are, honestly

| Audience | Can we talk to them with current numbers? |
|---|---|
| A technical peer / collaborator | **Yes.** 5-benchmark calibration story is clean and reproducible. |
| Public landing-page results section | **Not yet.** Need DeBERTa column, M9 use-case numbers, N=500 scale-up, arXiv anti-contamination result. |
| JevBench submission (M10) | **No.** Still missing the three unknowns about their contract AND two of the three required deliverables (use-case coverage, N=500 run). |

### What's next (biggest leverage)

1. **M9 use-case sweep** (loaders exist, server works, GPU-contention fixed via `/v1/generate`). ~1–2h at N=100. Delivers numeric backing for the three use-case cards (guardrails / triage / rating). **Highest marginal value right now.**
2. Scale academic suite to N=500 — hours of GPU time; strengthens every number already in the table.
3. Fix the DeBERTa GPU-contention story (either shared-GPU scheduling or overnight CPU batch).
4. Run `make_dataset.py` for arXiv post-cutoff (M8 10b). Needs `uv add arxiv`.
5. Fill `benchmarks/README.md` leaderboard summary (M8 13), now unblocked.
