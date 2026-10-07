# PAWS: Bernoulli below random — investigation

**Status:** **resolved 2026-10-07**. Root cause: `reverse` debias on
`BinaryQuestion` compounds the model's Yes-token prior instead of
canceling positional bias — the labels (`Yes` / `No`) are themselves
semantic. Fix: short-circuit debias to a no-op whenever the question is
binary. Post-fix PAWS N=100: Bernoulli accuracy **0.83** (was 0.37),
ECE 0.076 (was 0.176), NLL 0.41 (was 0.68) — Bernoulli now narrowly
beats Generative on PAWS while being 12× better calibrated.

## The observation

First real-N sweep across all 6 academic benchmarks returned a result for PAWS that doesn't fit:

| Baseline | Accuracy | macro-F1 | ECE (10) | NLL | Stability (reword) |
|---|---|---|---|---|---|
| **Bernoulli** | **0.37** | 0.270 | 0.176 | 0.680 | 1.000 |
| Generative | **0.82** | 0.800 | 0.180 | 4.97 | 0.890 |
| BGE-m3+LR | 0.56 | 0.510 | 0.009 | 0.692 | 1.000 |

Snapshot: [`benchmarks/academic/paws/results/2026-10-07.md`](../../benchmarks/academic/paws/results/2026-10-07.md).

Two things are wrong at the same time:

1. **Bernoulli is below random on a binary task** (random = 0.5). The predicted top-1 is wrong more often than a coin would be. Low accuracy with calibrated distributions is a signal: the model is confidently wrong in a consistent direction.
2. **Generative crushes Bernoulli** on the same examples at 0.82 — a 45-point gap on the same backbone, same prompt-stem framing. Same-model generative on PAWS is the strongest result in the entire sweep for the generative baseline. That's not noise.

Reword stability of 1.000 for Bernoulli says the direction holds across rewords. So this isn't a brittle-prompt artefact at the surface level; it's something stable in how Bernoulli (the logit-read path) is reading paraphrase pairs on this corpus.

## Hypotheses (ordered by current plausibility)

**H1 — Yes/No label semantics are inverted or scrambled.** Bernoulli's binary path reads single-token logits for ` Yes` / ` No` at the answer position. PAWS gold maps `label == 1` → `"Yes"` (paraphrase) and `label == 0` → `"No"`. If somewhere between the loader and the scorer the Yes/No logits get swapped (or if the chat template adds a prefix like "Answer: Yes/No" that shifts token positions), we'd see exactly this: calibrated but wrong-direction output. Supporting evidence: 0.37 ≈ 1 − 0.63, which is close to what accuracy would look like if the top-1 were systematically flipped from a correct model.

**H2 — The system prompt framing fights the task.** The binary system prompt renders options as `", ".join(options_for(question))` which for binary is `"Yes, No"`. For paraphrase detection, the implicit framing "answer Yes if these are paraphrases" may be undermined by the backbone's prior over `Yes`/`No` in a two-sentence context. The paraphrase question stem lives in `benchmarks/academic/paws/loader.py :: QUESTION.prompt` — the actual text matters.

**H3 — Reverse-debias is actively hurting on PAWS.** Debias averages the raw-order and reverse-order forward passes. For binary, "reverse" just swaps Yes/No in the option list, which should wash out any position bias. If the raw pass is already correct and the reverse pass is strongly biased, averaging could flip the top-1. Testable by rerunning with `BERNOULLI_DEFAULT_DEBIAS=none`.

**H4 — PAWS is adversarial by construction.** The dataset was built to fool lexical-overlap shortcuts. Qwen2.5-VL-7B may genuinely be bad at this. But the generative baseline running the same backbone hitting 0.82 is strong evidence against H4 as the primary cause — the model knows the task; the issue is in how Bernoulli reads it.

## First-cut investigation plan

Minimum viable set to find the cause, in order of cheapness:

1. **Eyeball ~10 examples.** Run `uv run bernoulli decide --state ...` against the first 10 PAWS test rows and print both the gold label and the predicted distribution. If `Yes`/`No` are swapped, we'll see it immediately — gold=`Yes` with high `P(No)` and vice versa. Minutes.

2. **Compare with and without debias.** Set `BERNOULLI_DEFAULT_DEBIAS=none` and rerun PAWS with --limit 100. If accuracy flips to ~0.63, H3 is confirmed. Minutes once the server is up.

3. **Compare framings.** Add a reword stem to PAWS that uses explicit label words ("paraphrase" / "not a paraphrase") rather than Yes/No, run the stability test on just that stem. If accuracy jumps, H2 is at least partly implicated.

4. **Instrument the single-token logits.** Add a `print()` (or temporary structured log) in `HFScorer.score` showing the raw logits for the two label tokens on the first few PAWS examples. See whether logit magnitudes match a correct model being flipped by the decode step, or a model that just has `Yes` ≫ `No` across the board.

Stop as soon as the smoking gun shows up. If H1 or H3 is right, the fix is one-liner; if H2, it's a per-benchmark prompt note.

## What this blocks

- **M8 task 13 (fill leaderboard summary).** Can't publish the comparison table with an unexplained below-random row for the headline mechanism. Either resolve the anomaly or add a footnote that clearly disclaims the PAWS Bernoulli number pending investigation.
- **Any public claim about Bernoulli on binary questions.** The use-case cards on `bernoulli.live` lean on binary via "Should the model refuse?" framing. XSTest / WildGuard sweeps will likely surface or rule out the same issue. Run those (M9) before quoting binary numbers publicly.

## What this does not block

- The calibration-gap story across the four Choice benchmarks (SST-2 / AG News / TweetEval / Banking77). Those four give the pitch its receipts regardless of PAWS.

## Updates

<!-- Append-only: date — observation / next step. Oldest first. -->

- **2026-10-07** — Doc created from the first N=100 sweep. No investigation yet.
- **2026-10-07** — **Step 1 complete (eyeball).** Ran 10 PAWS examples through the direct `HFScorer` + `decide()` path (no HTTP, no stability sweep):
  - **Bernoulli predicted `Yes` on all 10 examples.** 4 golds were `Yes` (correct), 6 were `No` (wrong). Accuracy = 4/10 = 0.40, matching the N=100 sweep's 0.37 within noise.
  - **P(Yes) range: [0.501, 0.637].** Every single prediction is just barely above the 0.5 threshold. The model never confidently says `No`.
  - **H1 (label inversion) is ruled out.** If `Yes`/`No` were swapped at the scorer level, we would see `P(Yes) < 0.5` for `Yes`-gold examples. Instead every row has `P(Yes) > 0.5` regardless of gold. The semantics are correct; the problem is a **Yes-biased prior** on the paraphrase prompt.
  - **H2 (prompt framing) moves to primary.** The current prompt — "Does Sentence 2 convey the same meaning as Sentence 1?" — may be priming the model toward Yes just by presenting two sentences in a comparison frame. Several of the actually-not-paraphrase examples are near-identical sentences with one word changed (number swap, place swap), which a lenient reading would call "same-ish". The dataset is adversarial by construction, and our framing seems to lean lenient.
  - **H3 (reverse debias flipping) still worth checking.** For `BinaryQuestion` the debias path has to do something — either swap `Yes`/`No` semantically or no-op. If it's semantically swapping and the model has a position bias at the token level, the average could inflate P(Yes). Testable next.
  - **Next step: H3 test** — rerun PAWS N=100 with `BERNOULLI_DEFAULT_DEBIAS=none`. If accuracy flips to ~0.63, debias is the culprit; if accuracy stays at ~0.37, debias is not involved and H2 is the remaining hypothesis.

- **2026-10-07** — **Step 2 complete (H3 CONFIRMED).** 50 PAWS examples through the direct scorer, both configs side-by-side:
  - `debias=reverse`: accuracy **0.380** (matches the N=100 sweep)
  - `debias=none`: accuracy **0.860**
  - 32/50 (64%) of examples flip between the two configs.
  - **Root cause:** For `BinaryQuestion` the "labels" are the semantic tokens `Yes` / `No` themselves. When reverse debias renders the second pass as `Yes) No / No) Yes`, the model doesn't do the mental gymnastics of mapping label-token-at-position-0 to the semantically-swapped option — it just outputs the token that matches its semantic conviction. Result: the model's Yes-token prior gets **doubled** in the sum instead of **canceled** by position-flipping, because position-flipping doesn't exist in a prompt where the labels are themselves the semantics.
  - For `ChoiceQuestion` with letter labels (A, B, C, ...) this doesn't happen — letters are arbitrary tokens with no inherent semantic prior.

- **2026-10-07** — **Fix landed + verified.** `bernoulli/debias.py` short-circuits `reverse`/`cyclic` to `none` whenever `question` is a `BinaryQuestion`. Position bias on binary is a real problem but requires a different mechanism (e.g. stem rewording) — not label reordering. Tests updated: `test_binary_short_circuits_reverse_to_single_pass` + `test_binary_short_circuits_cyclic_too`. The pre-existing `test_binary_question_uses_yes_no_tokens` was encoding the buggy behavior and has been replaced.

  **Post-fix PAWS N=100** (full sweep rerun):

  | baseline | acc | ECE (10) | NLL | note |
  |---|---|---|---|---|
  | **bernoulli** | **0.83** | **0.076** | **0.41** | was 0.37 / 0.176 / 0.68 pre-fix |
  | generative | 0.82 | 0.180 | 4.97 | unchanged (debias doesn't apply) |
  | bge-m3-lr | 0.56 | 0.009 | 0.69 | unchanged |

  Bernoulli narrowly beats Generative on PAWS, with ECE **2.4× better** and NLL **12× better**. The calibration-gap story now holds uniformly across all 5 real academic benchmarks (ignoring the 2-row arXiv placeholder).

  **Blast radius of this bug:**
  - All pre-fix `BinaryQuestion` results in-tree. In academic/, that's just PAWS; the first-sweep PAWS result file was overwritten.
  - M9 use-case benchmarks that use `BinaryQuestion` (WildGuardTest, ToxicChat, XSTest, CLINC150-OOS). None have been swept yet, so no corrupted results exist — but any pre-fix smoke numbers I may have mentioned should be ignored.
  - Historical `evals/reports/boolq.md` (pre-M8 snapshot, binary): would be affected, but it's an explicitly-dropped benchmark and the snapshot stays frozen as historical record. Noted in `benchmarks/README.md`'s "What is deliberately not here" section.

  **Takeaway**: calibrated-probability pipelines need per-question-kind defense. Reverse-order debias works for Choice but is semantically wrong for Binary. Rating uses letter labels so it's fine. Future question kinds should get this question explicitly during review.
