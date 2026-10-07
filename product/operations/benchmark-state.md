# Benchmark state — where we are

Companion to [`benchmark-primer.md`](benchmark-primer.md). The primer explains **what** we measure and **why**. This doc is a dated, append-at-top log of **where we actually are** against those measurements.

Each entry has: numbers table, what's sellable, honest limitations, honest gaps, and what's next.

---

## 2026-10-07 (evening) — arXiv post-cutoff real data in; anti-contamination claim backed

### What landed

- `make_dataset.py` ran on the dev box: 200 arXiv papers across cs.CL / cs.CV / math.PR / econ.EM, publication dates **2026-09-18 to 2026-10-06**, every one strictly after `BACKBONE_CUTOFF = 2025-01-01` (and well after any plausible Qwen2.5-VL training cutoff).
- First benchmark run on the real data (N=100 test).

### Numbers

| baseline | acc | ECE (10) | NLL | note |
|---|---|---|---|---|
| **Bernoulli** | **0.95** | **0.047** | **0.21** | wins on acc, calibration, NLL |
| Generative | 0.92 | 0.080 | 2.21 | NLL **10.7× worse** |
| BGE-m3 + LR | 0.94 | 0.509 | 0.87 | acc close; overconfident on 100 train → ECE blown out |

### Why this matters

Every other benchmark in the suite could, in principle, be in Qwen2.5-VL's pretraining corpus. arXiv post-cutoff can't be — the papers didn't exist yet. This row answers the sharpest skeptic question — *"are you just scoring on data the model memorised?"* — with a clear no.

Bernoulli hits 0.95 accuracy on data the model demonstrably hasn't seen, with the calibration gap intact (10.7× NLL ratio). The anti-contamination claim is now backed.

### What this moves

- `benchmarks/README.md` leaderboard — arXiv row fills in with real numbers.
- `web/benchmarks.html` detail page — arXiv row replaces the "placeholder pending 10b" footnote.
- `web/index.html` summary strip — new 6th row *"arXiv post-cutoff · papers the model can't have seen"* with the 0.95 / 0.92 / 10.7× cells. Takes the public-page thesis from "9 benchmarks confirm" to "9 benchmarks including anti-contamination confirm."
- The caveats section drops one bullet (arXiv placeholder) permanently.

### What's still outstanding

- WildGuardTest gated on HF.
- No DeBERTa column.
- N=100 is small.
- Dev-tier backbone (M4e capacity).

---

## 2026-10-07 (afternoon) — First M9 use-case sweep in

### M9 numbers (N=100, dev-tier Qwen2.5-VL-7B on A10G, g5.xlarge)

| Benchmark | Bernoulli acc | Generative acc | BGE-m3+LR acc | Bernoulli NLL | Generative NLL | NLL ratio |
|---|---|---|---|---|---|---|
| ToxicChat | 0.55 | 0.56 | **0.67** | 1.07 | 12.2 | 11× |
| XSTest | 0.68 | **0.79** | — | 0.53 | 5.80 | 11× |
| CLINC150-OOS | 0.46 | 0.37 | **0.77** | 0.95 | 17.4 | **18×** |
| Yelp 1-5 stars | **0.58** | 0.55 | 0.48 | **0.91** | 12.4 | 14× |

Per-benchmark extras (post `__init__` re-export fix):

- **XSTest**: Bernoulli `false_refusal_rate = 0.000`, `unsafe_recall = 0.36`. Generative `false_refusal_rate = 0.120`, `unsafe_recall = 0.70`. Bernoulli *never over-refuses* but under-flags unsafe prompts.
- **CLINC150-OOS**: Bernoulli `oos_auroc = 0.49` (random), `oos_recall = 0.52`. Generative `oos_auroc = 0.55`, `oos_recall = 0.87`. BGE+LR `oos_auroc = 0.75` (**best**) but `oos_recall = 0.00` — probability discriminates while argmax doesn't.
- **Yelp**: Bernoulli `MAE = 0.44` (**best**), `off_by_one = 0.95`. Generative `MAE = 0.50`, off-by-one 0.95. BGE+LR `MAE = 0.68`, off-by-one 0.95. All within one star for 95% of examples.

Full reports: `benchmarks/{guardrails/toxicchat, guardrails/xstest, triage/clinc150_oos, ratings/yelp_stars}/results/2026-10-07.md`.

**Not yet landed**: WildGuardTest. HF dataset `allenai/wildguardmix` is gated — needs one-time license acceptance on HF + `HF_TOKEN` env var. See the WildGuardTest README for the two-step unblock.

### Sellable additions

**Calibration gap confirmed on use-case benchmarks too.** Four M9 benchmarks each show Bernoulli NLL 11–18× better than Generative. Same thesis, different domain. Combined with the five academic benchmarks, that's **nine independent receipts** for *"same answer, dramatically better calibration."*

**XSTest's 0% false-refusal rate is a genuinely sellable product property.** For the guardrails use case, "will never over-refuse a safe prompt" is a hard thing to guarantee from a generative LLM. Pair with an explicit escalation path for the long tail (unsafe_recall of 0.36 says we miss ~2/3 of actually-unsafe prompts, so Bernoulli alone is *not* a complete safety classifier — it's a cheap first-pass that trades recall for zero false refusals).

**Yelp's MAE 0.44** beats Generative (0.50) and BGE+LR (0.68). For 5-way ordinal rating, Bernoulli's ability to use the full distribution (expected = Σ p_i · i) matters more than argmax — this is where the "typed decisions, not single tokens" story earns its keep.

### Honest limitations

1. **BGE-m3+LR dominates when training data is in-distribution**: ToxicChat (0.67 > 0.55), CLINC in_scope (1.00). It loses on Yelp (0.48 < 0.58) because 5-way ordinal regression on short review text isn't fit by linear LR, and because 500 examples × 5 classes = sparse per-class signal.
2. **XSTest unsafe_recall = 0.36** is low. Bernoulli has a "never refuse" prior post-debias-fix. Not disqualifying, but means Bernoulli-as-sole-guardrail is wrong framing.
3. **CLINC150-OOS oos_recall for Bernoulli = 0.52** = coin flip. The "none of these" probability story needs more work; probably prompt framing + maybe stem rewording.

### Still-outstanding gaps

- **WildGuardTest** stuck on HF gating (user action).
- **DeBERTa column** across M9 — same CPU-latency pathological issue as M8.
- **M9 landing-page cross-reference** (per-card link to the matching `results/latest.md`) — not yet done. Next natural task.
- **N=500 scale-up** on both academic and use-case suites. Hours of GPU time; not yet done.
- **arXiv post-cutoff real snapshot** (M8 task 10b).

### Where we stand

- **Internal technical-peer conversation**: can show 9 benchmarks with the calibration-gap story. Credible.
- **Public landing page**: still missing WildGuardTest, DeBERTa column, and the per-card cross-reference. One day's work on cross-reference + WildGuard unblock closes most of the gap.
- **JevBench submission**: still need the three JevBench unknowns + N=500 scale-up.

---

## 2026-10-07 (morning) — First academic sweep in; PAWS debias bug fixed

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
