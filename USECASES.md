# Bernoulli — use cases & honest comparison

A single document to help you decide whether Bernoulli fits your workload. It covers:

- the TL;DR decision tree
- when Bernoulli is the right answer, with benchmark backing
- when it's NOT the right answer, with the alternative you should pick instead
- what the benchmarks show in our favor
- what the benchmarks don't show in our favor
- four realistic use cases with request/response shapes you can copy-paste

Every claim here is backed by a public benchmark run under [`benchmarks/`](benchmarks/). Each result file ships a reproducibility footer (model revision, dataset hash, seed, hardware) so any number can be reproduced or disproved.

**Contents**

- [TL;DR — should I use Bernoulli?](#tldr--should-i-use-bernoulli)
- [What Bernoulli is (one minute)](#what-bernoulli-is-one-minute)
- [When Bernoulli is the right answer](#when-bernoulli-is-the-right-answer)
- [When Bernoulli is NOT the right answer](#when-bernoulli-is-not-the-right-answer)
- [What the benchmarks show in our favor](#what-the-benchmarks-show-in-our-favor)
- [What the benchmarks don't show in our favor](#what-the-benchmarks-dont-show-in-our-favor)
- [The alternatives, in detail](#the-alternatives-in-detail)
- [Four realistic use cases](#four-realistic-use-cases)
  - [Support ticket triage](#use-case-support-ticket-triage)
  - [LLM output guardrail](#use-case-llm-output-guardrail)
  - [Content moderation](#use-case-content-moderation)
  - [Code review severity](#use-case-code-review-severity)
- [Patterns across use cases](#patterns-across-use-cases)
- [How to validate on your own data](#how-to-validate-on-your-own-data)

---

## TL;DR — should I use Bernoulli?

A fast triage. Answer these in order; stop at the first "no":

1. **Do I need to make a decision from a fixed set of answers** (choice / binary / rating)? No → use a generative LLM for free text.
2. **Do I need a calibrated probability**, not just the top answer? No → use whatever's cheapest; prompt-and-parse works fine if "right 85% of the time" is enough.
3. **Is my label space small (≤26 options) OR do I lack ≥2000 labeled training examples**? If you have 2000+ labels AND a fixed label space with many classes, **BGE-m3 + logistic regression beats us** on those tasks — go train it instead.
4. **Is at least one of these true**:
   - My data is novel / recent / proprietary (not in any public pre-training set), OR
   - I need a binary Yes/No question against semantic content, OR
   - I need multiple questions over the same input (support triage, LLM-as-judge, moderation), OR
   - I need calibration to run thresholds and auto-decide at scale, OR
   - I need on-prem / in-VPC inference with zero outbound calls?

   Yes → Bernoulli. No → DeBERTa-v3-zeroshot or a prompt-and-parse LLM might be simpler.

### The 10-second version

- **Pick BGE-m3 + LR** if you have 2000+ labels and a fixed label space.
- **Pick DeBERTa-v3-zeroshot** if your task is classical NLP (sentiment, topic) and single-choice.
- **Pick a prompt-and-parse LLM** if you don't care about calibration or need generated text alongside the decision.
- **Pick a specialist safety model** (Llama Guard 3, ShieldGemma, WildGuard) for production-grade safety classification.
- **Pick Bernoulli** when you need calibrated probabilities over typed decisions, you don't have labels, your content is novel, or you need multiple questions per input.

---

## What Bernoulli is (one minute)

Bernoulli reads an LLM's probability distribution over your answer options directly — no decoding loop, no parsing, no retries. Same backbone as a generative LLM; different reading mechanism.

```json
POST /v1/decide
{
  "state": {"text": "<the input>"},
  "questions": [
    {"id": "intent", "type": "choice", "prompt": "What does the customer want?",
     "options": ["refund", "exchange", "tracking", "other"]}
  ]
}
```

Returns:

```json
{"decisions": {"intent": {
  "answer": "tracking",
  "confidence": 0.942,
  "distribution": {"tracking": 0.942, "other": 0.041, "refund": 0.015, "exchange": 0.001}
}}}
```

Three question types: **choice** (2–26+ options), **binary** (Yes/No), **rating** (integer scale with expected value). One forward pass per question. Multiple questions share one state tokenization (prefix caching).

For the deep version of how this works + how it compares to a prompt-and-parse pipeline, see [`HOW-IT-WORKS.md`](HOW-IT-WORKS.md).

---

## When Bernoulli is the right answer

Five scenarios where the benchmarks + architecture give us a clear edge. All numbers are from the public [`benchmarks/`](benchmarks/) runs at N=500 (arXiv at N=100 — full available test slice).

### 1. You need calibrated probabilities for threshold-based decisions

The product claim: "when Bernoulli says 0.9, it's right about 90% of the time." The benchmark proof: across **all nine** public benchmarks, Bernoulli's NLL is **7–19× better than the same backbone used as a text-and-parse classifier**. That ratio is what "miscalibration gap" looks like as a number — a prompt-and-parse pipeline is confident when it shouldn't be.

Why this matters in practice: if your code is `if confidence > 0.9: auto_approve`, a miscalibrated 0.9 is a product bug. You'll auto-approve things the model doesn't actually know, and your false-positive rate quietly climbs.

### 2. Your data wasn't in any public pre-training set

The arXiv post-cutoff benchmark samples 100 papers published 2026-09-18 to 2026-10-06 — strictly after both Qwen2.5-VL's and DeBERTa's training cutoffs. On classifying them into cs.CL / cs.CV / math.PR / econ.EM:

| | accuracy | NLL |
|---|---|---|
| **Bernoulli (Qwen 7B)** | **0.950** | **0.207** |
| BGE-m3 + LR | 0.940 | 0.869 |
| Generative (same Qwen) | 0.920 | 2.211 |
| DeBERTa-zeroshot | 0.530 | 1.084 |

DeBERTa drops 42 points on content it couldn't have memorized. Bernoulli-via-Qwen generalizes to novel semantic content; a zero-shot NLI encoder tuned on fixed data doesn't. If your corpus is **recent news, proprietary tickets, internal docs, scientific papers past 2024, or any domain that isn't ancient-web-text**, this is the regime where our architecture earns its keep.

### 3. You need Yes/No semantic reasoning (binary questions)

Zero-shot NLI encoders like DeBERTa can only score natural-language label hypotheses — they don't handle binary Yes/No questions ("is this urgent?", "is this a paraphrase?", "should this be flagged?"). On PAWS (adversarial paraphrase binary, N=500):

| | accuracy | NLL |
|---|---|---|
| **Bernoulli** | **0.846** | **0.376** |
| Generative | 0.836 | 4.532 |
| BGE-m3 + LR | 0.552 | 0.690 |
| DeBERTa-zeroshot | *n/a (BinaryQuestion unsupported)* | — |

If your product asks binary questions over semantic content (guardrails, moderation yes/no gates, policy checks), zero-shot encoders are off the map and prompt-and-parse is poorly calibrated. We're the natural fit.

### 4. You need to ask several questions about the same input

Real workloads often need multiple classifications per input: support triage asks "intent + urgency + anger + escalate"; LLM-as-judge asks "safe + on-topic + factual + refusal-appropriate"; code review asks "severity + area + reviewer-tier + deploy-risk". Prompt-and-parse makes one HTTP call per question; Bernoulli processes the state once and shares it across questions via prefix caching.

Numbers from [`evals/reports/loadtest.md`](evals/reports/loadtest.md) on A10G 24GB:

| questions / request | p50 latency |
|---|---|
| 1 | 76 ms |
| 5 | 368 ms |
| 20 | 1466 ms |

Roughly **70 ms per additional question** on the dev-tier backbone — vs the full tokenization cost per call for a prompt-and-parse pipeline.

### 5. You want to run on your own hardware without API lock-in

Bernoulli is Apache 2.0 and backbone-agnostic (any open-weights VL model with a swap of `BERNOULLI_MODEL_ID`). Three deployment modes: self-host in your VPC (flagship), managed-by-us-on-your-cloud (BYOC), or hosted-by-us. **The flagship self-host path is zero outbound calls at inference time** — nothing leaves your network. If data residency is a hard requirement, this is in a different category from any hosted API.

---

## When Bernoulli is NOT the right answer

Honest: here are the cases where someone else beats us, with the specific alternative to pick.

### 1. You have 2000+ labeled examples and a fixed label space

Train a classifier. On tasks where this is available, a BGE-m3 embedding + sklearn logistic regression (2000 training examples) crushes everything zero-shot:

| benchmark | BGE+LR | Bernoulli | gap |
|---|---|---|---|
| **Banking77** (77-way intent) | **0.950** | 0.572 | **+38 pt** |
| **CLINC150-OOS** (151-way) | **0.804** | 0.448 | **+35 pt** |
| ToxicChat (binary) | **0.782** | 0.690 | +9 pt |
| AG News (4-way topic) | **0.876** | 0.828 | +5 pt |

A trained classifier with that much labeled data has fundamentally better signal than any zero-shot approach. For production classification with a fixed label space, **this is the right choice if you have the labels**.

Our story is for the case you *don't* have labels, or your label space changes (think: swapping policy categories, adding new product SKUs, bootstrapping a new domain).

### 2. Your task is classical NLP where DeBERTa was effectively trained near

DeBERTa-v3-zeroshot (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`) was fine-tuned on NLI datasets covering a lot of academic benchmark content. On tasks that look like its training distribution:

| benchmark | DeBERTa | Bernoulli | gap |
|---|---|---|---|
| SST-2 (binary sentiment) | **0.930** | 0.914 | +1.6 pt |
| AG News (4-way topic) | **0.874** | 0.828 | +4.6 pt |
| Banking77 (accuracy) | **0.640** | 0.572 | +6.8 pt |

The arXiv post-cutoff crash (0.53) is the diagnostic: DeBERTa's wins look like training-data overlap, not generalization. **If your task is similar to published NLP benchmarks and you don't need binary Yes/No**, DeBERTa is a legitimate pick — faster per-call (~37 ms per label), simpler stack, no LLM required.

### 3. Your task is production-grade safety classification

For "flag unsafe prompts" at scale with high unsafe-recall, specialist safety models (Llama Guard 3, ShieldGemma, WildGuard-7B) are tuned specifically for this and will outperform our general zero-shot pass. WildGuardTest is HF-gated so we haven't published that column; the direction is clear anyway.

Bernoulli's XSTest numbers tell the honest story: we have **<1% false-refusal rate** (4 of 500 prompts) with maintained accuracy — a "safe-by-default" posture suitable as a first-pass gate. Our unsafe-recall of 0.28 (vs Generative's 0.49 at a 20% false-refusal cost) means we miss some actually-unsafe prompts.

**The right architecture for a real guardrail**: Bernoulli as the "don't over-refuse the user" first pass, paired with a specialist safety model as an escalation path for the uncertain cases. We're not a drop-in replacement for Llama Guard.

### 4. You need to detect "none of these" on long-tail intent sets

CLINC150-OOS tests whether a system can identify out-of-scope prompts in a 150-intent space. **Our `oos_auroc = 0.48` — essentially a coin flip.** BGE-m3+LR hits 0.78 (not by predicting OOS correctly — it never does, `oos_recall = 0.000` — but by ranking in-scope confidences well enough that a threshold works). This is a real limitation of our prompt framing today.

If your product needs the "I don't know" probability to actually mean something on 100+ class intent sets, use an embedding classifier with a calibrated threshold. We'd like to fix this; investigation-pending but not shipped.

### 5. You need stability over option reorder on >50 options

Our chunked scoring (for choice tasks with >26 options) splits labels into chunks of ≤26 and softmaxes globally. The top-1 can shift based on which labels landed in which chunk — Banking77 reorder stability is **0.07** (DeBERTa and BGE+LR hit 1.00). Architectural, not a bug.

**If your workflow requires option-reorder invariance on >50 options, DeBERTa or BGE+LR are the honest picks today.** For ≤26 options, Bernoulli is reorder-stable (reverse debias is the default).

### 6. You need generated text, not just a decision

Summaries, explanations, rewritten emails, code suggestions. Bernoulli is for decisions, not content. Use a generative LLM. If you want *both* a decision and an explanation, run them as two separate calls — one Bernoulli, one chat — and keep the clean separation.

### 7. Open-vocabulary answers

Bernoulli assumes your decision has a fixed enumeration of possible answers. If the right answer is "any of these 10,000 product SKUs" or "generate a new SKU name," that's a different tool (retrieval, NER, text generation).

---

## What the benchmarks show in our favor

Every number is at N=500 (arXiv at N=100 — the full available test slice). Numbers link to per-benchmark result files.

1. **Calibration gap vs the same backbone used as text-and-parse.** NLL is 7–19× better than the Generative baseline across all 9 benchmarks. Not a sample artifact — consistent across sentiment, topic, intent, emotion, paraphrase, moderation, safety, OOS detection, and rating. [Full comparison](benchmarks/README.md).
2. **arXiv post-cutoff — provably unseen content.** 0.95 (us) vs 0.53 (DeBERTa) vs 0.92 (Generative) vs 0.94 (BGE+LR). The anti-contamination test. [`benchmarks/academic/arxiv_post_cutoff/`](benchmarks/academic/arxiv_post_cutoff/).
3. **PAWS binary paraphrase.** 0.846 accuracy with 0.376 NLL. DeBERTa can't score it; BGE+LR crashes to 0.55 (near-random). [`benchmarks/academic/paws/`](benchmarks/academic/paws/).
4. **Yelp 1-5 star rating.** MAE of 0.42 is the lowest of any baseline (Generative 0.46, BGE+LR 0.53). Uses the full predicted distribution for the expected-value calculation, not just argmax. [`benchmarks/ratings/yelp_stars/`](benchmarks/ratings/yelp_stars/).
5. **XSTest low over-refusal.** <1% false-refusal rate with maintained accuracy — a "safe-by-default" posture other models trade away for higher unsafe-recall. [`benchmarks/guardrails/xstest/`](benchmarks/guardrails/xstest/).
6. **Latency with multiple questions.** ~70 ms per additional question (vs full HTTP call per question for prompt-and-parse). Natural fit when the same input needs several classifications. [`evals/reports/loadtest.md`](evals/reports/loadtest.md).
7. **One output token, bounded cost.** `max_tokens=1` means output-token line items don't scale with answer length. Useful when self-hosting (one forward pass per question) or when per-token pricing is bad.
8. **Open weights, no API lock-in.** Apache 2.0, swappable backbone. Three deployment modes (self-host, managed-on-your-cloud, hosted-by-us). Zero outbound calls at inference time on the self-host path.
9. **Production backbone step-up looks promising.** A one-shot H100 rental of Qwen2.5-VL-32B lifted the hardest JevBench slice by ~18 points — see [`product/reports/2026-10-10-h100-32b.md`](product/reports/2026-10-10-h100-32b.md). The dev-tier numbers you see here are a floor, not a ceiling.

---

## What the benchmarks don't show in our favor

Equal time to the honest downsides.

1. **BGE-m3 + LR with 2000 labels crushes us on label-heavy tasks.** +38 pt on Banking77 (77-way), +35 pt on CLINC150-OOS (151-way), +9 pt on ToxicChat, +5 pt on AG News. If labeled data is available, train the classifier.
2. **DeBERTa-zeroshot ties or edges us on classical NLP.** +1.6 pt on SST-2, +4.6 pt on AG News, +6.8 pt on Banking77 accuracy. Not a disaster, but not domination either. If your task is similar to published NLP benchmarks and you don't need binary Yes/No, DeBERTa is competitive.
3. **CLINC150-OOS `oos_auroc = 0.48`.** Coin flip on the "none of these" probability story. Real limitation on long-tail intent + OOS detection; investigation-pending.
4. **Banking77 reorder stability = 0.07.** Chunked scoring architectural limitation on >26 options. Reorder-invariance is a win for DeBERTa / BGE+LR here.
5. **XSTest `unsafe_recall = 0.28`.** We miss some actually-unsafe prompts in exchange for the <1% false-refusal rate. For a complete production guardrail, pair with a specialist safety model.
6. **Dev-tier backbone throughout.** All numbers are Qwen2.5-VL-7B. Production tier (32B) is a one-off H100 story so far, not shipped as a serving config.
7. **JevBench ranking is still pending.** Our submission is in Benchmark Heaven's FIFO queue; this remains self-reported until that posts. [`benchmarks/jevbench/`](benchmarks/jevbench/).
8. **No comparison yet against specialist safety models** (Llama Guard 3, ShieldGemma, WildGuard-7B). WildGuardTest is HF-gated and parked.
9. **No external audit.** Every number here is reproducibility-footer-backed, so any claim is reproducible or disprovable — but not yet independently reproduced.

---

## The alternatives, in detail

### DeBERTa-v3-zeroshot (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`)

435M-parameter encoder, NLI-tuned so it treats classification as "entailment of a hypothesis." Choice tasks only — binary Yes/No is unsupported by design.

- **Pick it when**: zero-shot text classification on classical NLP (sentiment, topic), choice-only, single-input-at-a-time. Fast per-call once GPU is warm (~37 ms per label). No LLM dependency.
- **Skip it when**: your task is binary Yes/No, your content is post-cutoff / proprietary, or you need multiple questions over the same input.
- **Reference**: [`huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0). Our implementation lives in [`benchmarks/common/baselines.py`](benchmarks/common/baselines.py).

### BGE-m3 + logistic regression

~2.3B sentence embedder + sklearn LogisticRegression trained on your labels. Needs 500–2000+ labeled examples per task.

- **Pick it when**: you have labeled training data, your label space is fixed, you have many classes (10+), and inference latency matters (~12 ms per call on CPU for the LR head once embeddings are cached).
- **Skip it when**: you don't have labels, the label space changes per workflow, or you need calibrated probabilities with limited training data (its ECE on arXiv at 100 training examples was pathological — 0.509).
- **Reference**: [`huggingface.co/BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) + any sklearn LR.

### Prompt-and-parse (ChatGPT / Claude / open-weight chat)

Ask an LLM in natural language, parse the completion. The current default for "add intelligence to this backend."

- **Pick it when**: you also need generated text alongside the decision (explanation, rewritten message, summary); or when being right 85% of the time is enough and calibration doesn't matter.
- **Skip it when**: you need calibrated probabilities (NLL will be 7–19× worse than Bernoulli), you need many decisions per input (pay full tokenization cost per question), or you can't tolerate parse failures / retries.
- **Reference**: whatever your API provider is.

### Fine-tuned task-specific encoder

Fine-tune a BERT / DeBERTa / RoBERTa on your labeled data. Highest accuracy ceiling, biggest ops cost.

- **Pick it when**: you have 10,000+ labeled examples, a stable production task, and the accuracy difference between zero-shot and fine-tuned matters to the business more than the training + maintenance cost.
- **Skip it when**: your label space shifts, you have less data, or the training pipeline would be more code than the inference pipeline.

### Specialist safety models (Llama Guard 3, ShieldGemma, WildGuard-7B)

Open-weight models fine-tuned specifically for safety classification. Report token probabilities over safe/unsafe/category labels, so the comparison with Bernoulli is apples-to-apples when it matters.

- **Pick it when**: your workload is production-grade safety classification where unsafe-recall is the top metric and the training distribution is well-aligned with your traffic.
- **Skip it when**: safety is one of many questions you need to answer about the same input (triage, moderation, grading), or when "safe-by-default" with low over-refusal is more important than "catch every unsafe case."
- **Reference**: Llama Guard 3 ([`meta-llama/Llama-Guard-3-8B`](https://huggingface.co/meta-llama/Llama-Guard-3-8B)), ShieldGemma, WildGuard-7B.

### Classical ML on hand-engineered features

Still legitimate in 2026. If your decision reduces to counting words from a known lexicon, scoring sentiment with VADER, matching regex patterns against a policy — that'll be faster, cheaper, and more auditable than any of the above. Don't let the LLM renaissance talk you out of a solution that already works.

---

## Four realistic use cases

Each is interface-neutral — the request shape works verbatim as an HTTP POST body or as a `--request` JSON for the CLI. See [`HTTP.md`](HTTP.md) or [`CLI.md`](CLI.md) for the mechanics of calling the engine.

### Use case: Support ticket triage

#### The problem

A customer email arrives. You need to route it to the right queue, flag it for an escalation if the customer is unhappy, and surface the key facts (what they want, what product they're referring to) before a human looks at it. Doing this with a generative LLM means one call per question or one giant parsed JSON blob — brittle either way.

#### What state looks like

Real support emails are 100–2000 words, often with the thread of prior exchanges included. Something like:

```
From: angry.customer@example.com
Subject: Re: Re: Order #42-1138 - STILL HAVEN'T SHIPPED

This is the THIRD time I'm emailing about this. I ordered the Series-X
jacket on October 1st - over two weeks ago - and it still hasn't shipped.
The tracking number you gave me (1Z999AA10123456789) returns nothing on
the carrier's site. I need this jacket for a trip on Friday.

I've been a customer for 4 years. I have the Premier membership that I
pay $9.99/month for, which specifically promises "guaranteed 3-day
shipping". That is clearly not happening here.

Your last reply (3 days ago, Agent Sarah) said someone from fulfillment
would call me. Nobody has called.

I want this resolved TODAY. If it ships today I'll keep the order.
If it doesn't ship by end of day I want a full refund AND credit for
the Premier annual fee because the shipping guarantee is clearly a lie.

Reply within the hour or I'm filing a chargeback.

- A. Customer

> On Oct 12 Agent Sarah wrote:
> Hi, thanks for your patience. I've escalated this to our fulfillment
> team and someone will reach out to you within 48 hours...
>
> > On Oct 10 A. Customer wrote:
> > Where is my order? It's been 9 days...
```

#### The questions

```json
{
  "state": {"text": "<the full thread above>"},
  "questions": [
    {
      "id": "intent",
      "type": "choice",
      "prompt": "What does the customer most want right now?",
      "options": ["refund", "ship_now", "status_update", "complaint_only", "other"]
    },
    {
      "id": "urgency",
      "type": "rating",
      "prompt": "How time-critical is this ticket? 1 = can wait a week, 5 = handle in next hour.",
      "scale": [1, 5]
    },
    {
      "id": "anger",
      "type": "rating",
      "prompt": "How angry is the customer? 1 = calm, 5 = threatening to leave or escalate publicly.",
      "scale": [1, 5]
    },
    {
      "id": "escalate",
      "type": "binary",
      "prompt": "Should this skip the normal queue and go straight to a senior agent?"
    },
    {
      "id": "chargeback_risk",
      "type": "binary",
      "prompt": "Is the customer threatening a chargeback or public complaint (e.g., social media, review site)?"
    },
    {
      "id": "sla_breach",
      "type": "binary",
      "prompt": "Does the customer claim an SLA or shipping guarantee was violated?"
    }
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
```

Six questions in one call. On the 7B dev model, the whole thing comes back in roughly 500 ms (state tokens processed once, six questions × ~70 ms each).

#### How to use the answers

```python
decisions = response["decisions"]

# Routing
if decisions["escalate"]["answer"] and decisions["escalate"]["probability"] > 0.75:
    route_to_senior_queue(ticket)
elif decisions["chargeback_risk"]["answer"]:
    route_to_retention_team(ticket)
else:
    route_to_standard_queue(ticket)

# Prioritization within the queue
priority_score = (
    decisions["urgency"]["expected"] * 2
    + decisions["anger"]["expected"]
    + (3 if decisions["sla_breach"]["answer"] else 0)
)

# Context for the agent, surfaced at the top of the ticket
summary = {
    "they_want": decisions["intent"]["answer"],
    "they_want_confidence": decisions["intent"]["confidence"],
    "angry": decisions["anger"]["expected"] >= 4.0,
    "sla_claim": decisions["sla_breach"]["answer"],
}
```

> **Why not a generative LLM here?** The agent needs confidence numbers to decide "is this really a chargeback risk or is the model just hedging?" A generative LLM gives you "yes" or "no" — you can't tell the difference between a 0.95 "yes" and a 0.52 "yes" without asking the model again.

---

### Use case: LLM output guardrail

#### The problem

You're building a chatbot powered by a generative LLM. Before showing a response to the user, you want to catch unsafe outputs, policy violations, and factual claims that need human verification. "LLM as a judge" — but you want calibrated probabilities so you can set your own thresholds for auto-ship vs auto-block vs send-to-human.

#### What state looks like

The state here is the user's prompt + the LLM's response, together. You're asking Bernoulli to judge what the LLM just produced:

```
### User prompt
I'm writing a story where a character gets locked out of their house.
Can you describe realistically how someone might get back in without
their key?

### Model response
In your story, the character has several options to get back into their
home without their key:

1. **Call a locksmith.** This is the most common and legal approach. A
   locksmith can usually arrive within 30-60 minutes and will verify the
   caller's identity (usually with a photo ID matching the address)
   before unlocking the door. Costs typically $50-150.

2. **Check for a spare key.** Many people keep a spare with a trusted
   neighbor, under a specific rock, or in a magnetic key holder on the
   underside of a car's wheel well. If your character has planned ahead,
   this is instant.

3. **Call the landlord or property management.** If renting, the landlord
   usually has a master key and may charge a small fee for an after-hours
   unlock.

4. **Climb through an unlocked window.** This is risky (both physically
   and legally - make sure it's clear the home is theirs) but a classic
   fiction trope. Add tension by having them lose their balance halfway.

I avoided describing lockpicking techniques since that veers into
instructional territory that could enable crime. If your story needs
your character to pick a lock, I'd suggest leaving the mechanics vague
("she worked at the lock for several tense minutes") rather than giving
step-by-step instructions.
```

#### The questions

```json
{
  "state": {"text": "<the user prompt + model response above>"},
  "questions": [
    {
      "id": "safe",
      "type": "binary",
      "prompt": "Is the model's response safe to show to the user? (Does not provide step-by-step instructions for illegal activity, does not reveal dangerous information, does not generate hateful or harassing content.)"
    },
    {
      "id": "on_topic",
      "type": "binary",
      "prompt": "Does the response answer the user's actual question?"
    },
    {
      "id": "refusal_appropriate",
      "type": "choice",
      "prompt": "If the model refused or partially refused, was the refusal appropriate?",
      "options": ["no_refusal", "refused_appropriately", "refused_too_cautiously", "refused_when_shouldn_t_have"]
    },
    {
      "id": "factual_risk",
      "type": "rating",
      "prompt": "Does the response contain factual claims that could be wrong and would matter if they were? 1 = no factual claims, 3 = contains claims but they look right, 5 = contains confident-sounding claims that could easily be wrong.",
      "scale": [1, 5]
    },
    {
      "id": "jailbreak_attempt",
      "type": "binary",
      "prompt": "Does the user's prompt appear to be an attempt to jailbreak or manipulate the model into unsafe output?"
    }
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
```

#### How to use the answers

```python
d = response["decisions"]

# Hard block
if not d["safe"]["answer"] and d["safe"]["probability"] < 0.5:
    return BLOCK(reason="safety")

# Soft gate: send to human if probability is in the uncertain band
if d["safe"]["probability"] < 0.9:
    return HUMAN_REVIEW(reason="safety uncertain")

# Factual check: if model is confidently wrong-shaped, add a disclaimer
if d["factual_risk"]["expected"] >= 4.0:
    response_to_user = append_disclaimer(
        response_to_user, "Some factual details may be inaccurate — please verify."
    )

# Over-refusal check: if the model refused but shouldn't have, log for
# the eval team
if d["refusal_appropriate"]["answer"] == "refused_too_cautiously":
    log_overrefusal(user_prompt, model_response)
```

The thresholds are **yours to set**. 0.9 is paranoid; 0.5 is permissive. Bernoulli gives you the number — you decide what it means for your product.

> **Why not a generative LLM here?** You could absolutely ask a generative LLM to judge another LLM's output. But then you'd be asking it to produce text like "The response is safe. Confidence: high" and parsing that out. With Bernoulli, you get a probability directly — one number, machine-ready. You can route on 0.73 vs 0.91 without a second LLM-as-judge call.
>
> **Why not Llama Guard 3?** For raw safety classification with high unsafe-recall, pick it. The Bernoulli pattern wins when safety is one of five or six questions you need per response — the five-question request above shares one state tokenization. A pure safety call that only asks "is this unsafe?" is where a specialist model is competitive.

---

### Use case: Content moderation

#### The problem

You're running a forum, marketplace, or chat platform. Every user-submitted post needs a moderation decision in <100 ms: ship, send to a review queue, or auto-reject. You want a calibrated probability so the review queue only gets the genuinely-uncertain cases.

#### What state looks like

A user post. Length varies — a short review might be 20 words, a long rant 1000 words. State = the post text, maybe with context about the user (new vs established, posting rate, etc.).

```
Post title: Why do we even have mods on this subreddit??
Author: FrustratedUser99 (3 posts, 2 of which were removed)
Posted: 2026-10-05 14:22 UTC
Body:
    I posted a totally reasonable question yesterday and some power-tripping
    mod removed it within 10 minutes. No warning, no explanation. The mods
    on this sub are honestly the worst. They're all failed [redacted
    slur removed] who get off on wielding fake internet authority.
    Reporting won't do anything because the admins are just as [redacted].
    This sub has gone completely downhill, I'm done.
```

#### The questions

```json
{
  "state": {"text": "<the post above>"},
  "questions": [
    {
      "id": "category",
      "type": "choice",
      "prompt": "Which policy category best describes this post?",
      "options": ["safe", "incivility", "harassment", "hate_speech", "spam", "off_topic"]
    },
    {
      "id": "severity",
      "type": "rating",
      "prompt": "How severe is any policy violation? 1 = clean, 3 = borderline, 5 = clear ban-worthy violation.",
      "scale": [1, 5]
    },
    {
      "id": "contains_slur",
      "type": "binary",
      "prompt": "Does the post contain a slur, even redacted or censored?"
    },
    {
      "id": "ban_evasion",
      "type": "binary",
      "prompt": "Does the post pattern match ban evasion or coordinated harassment?"
    },
    {
      "id": "needs_human",
      "type": "binary",
      "prompt": "Is this edge-case enough that a human moderator should review before any action?"
    }
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
```

#### How to use the answers

```python
d = response["decisions"]

if d["severity"]["expected"] >= 4.5 and not d["needs_human"]["answer"]:
    auto_remove(post, reason=d["category"]["answer"])
elif d["severity"]["expected"] >= 2.5 or d["needs_human"]["answer"]:
    review_queue.add(post, priority=d["severity"]["expected"])
else:
    publish(post)

# Logging for policy iteration
log_moderation_signal(
    post_id=post.id,
    category=d["category"]["answer"],
    category_dist=d["category"]["distribution"],  # keep the full dist for later analysis
    severity=d["severity"]["expected"],
)
```

The full `distribution` for each decision — not just the winner — is the data flywheel. Over time you can plot how severity shifts by category, catch calibration drift, retrain thresholds. A pipeline that only logs "the model said: incivility" throws away the signal you'd need for any of that.

> **Why not a generative LLM here?** At scale (thousands of posts per second), paying LLM-call latency per post is a non-starter. Bernoulli reads the probability in a single forward pass; a batched request against 20 posts shares the model load. Plus: you need calibrated probabilities to tune ship/queue/block thresholds against your false-positive and false-negative budgets, which text generation doesn't give you.
>
> **Why not BGE-m3+LR here?** If you have 2000+ labeled historical moderation decisions in your category space, train it. Our ToxicChat numbers put BGE+LR at 0.78 vs us at 0.69 — a real gap. If your category space is fixed and labels exist, that's the right choice. Bernoulli wins when policies shift, when the category space is bespoke to your product, or when you want to add a new question without re-training.

---

### Use case: Code review severity

#### The problem

Someone opens a PR. Before a human reviewer looks at it, you want a signal: is this a trivial typo fix, a normal change, or a security-critical / correctness-critical change that deserves senior-engineer eyes? You also want a quick "which areas does this touch" classification for routing to the right domain expert.

#### What state looks like

The diff plus a few hundred lines of surrounding context (function bodies, class defs, callers of the changed functions). Can be large — code review states commonly run 1000-5000 tokens.

```
diff --git a/auth.py b/auth.py
index abc1234..def5678 100644
--- a/auth.py
+++ b/auth.py
@@ -78,10 +78,13 @@ def verify_password(submitted: str, stored_hash: str) -> bool:
     """Return True iff `submitted` matches `stored_hash`."""
     if not submitted or not stored_hash:
         return False
-    return submitted == stored_hash
+    return hmac.compare_digest(submitted, stored_hash)

### Surrounding file context (auth.py lines 1-100):
import hmac
import hashlib
...
class AuthService:
    def login(self, username: str, password: str) -> AuthResult:
        user = self.user_repo.get_by_username(username)
        if user is None:
            return AuthResult.failed()
        if not verify_password(password, user.password_hash):
            return AuthResult.failed()
        ...

### PR description:
"Switch password comparison to constant-time compare. Resolves a timing
side-channel reported in SEC-2234."
```

#### The questions

```json
{
  "state": {"text": "<the diff + context + description above>"},
  "questions": [
    {
      "id": "severity",
      "type": "rating",
      "prompt": "How impactful is this change? 1 = cosmetic/comment-only, 2 = normal fix or feature, 3 = security or correctness-critical.",
      "scale": [1, 3]
    },
    {
      "id": "category",
      "type": "choice",
      "prompt": "Which engineering area does this change primarily touch?",
      "options": ["auth_security", "data_integrity", "performance", "ui_ux", "docs_or_comments", "tests", "build_tooling", "other"]
    },
    {
      "id": "needs_senior_review",
      "type": "binary",
      "prompt": "Should a senior engineer in the domain review this before merging?"
    },
    {
      "id": "test_coverage_adequate",
      "type": "binary",
      "prompt": "Does this change include tests that cover the new behavior? (If no behavior change, answer yes.)"
    },
    {
      "id": "backwards_compatible",
      "type": "binary",
      "prompt": "Is this change backwards-compatible with existing callers?"
    },
    {
      "id": "deploy_risk",
      "type": "rating",
      "prompt": "What's the production deploy risk? 1 = safe to merge-and-deploy on Friday afternoon, 3 = needs a staged rollout and on-call awareness, 5 = don't deploy without a dedicated review.",
      "scale": [1, 5]
    }
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
```

#### How to use the answers

```python
d = response["decisions"]

# Route to the right domain reviewer
pr.add_reviewer(team_for_category(d["category"]["answer"]))

# Flag high-stakes changes
if d["severity"]["expected"] >= 2.5 or d["needs_senior_review"]["answer"]:
    pr.add_label("needs-senior-review")
    pr.add_reviewer(senior_engineers[d["category"]["answer"]])

if d["deploy_risk"]["expected"] >= 3.0:
    pr.add_label("deploy-risk")
    pr.block_automerge()

if not d["test_coverage_adequate"]["answer"] and d["test_coverage_adequate"]["probability"] < 0.3:
    pr.add_comment(
        "Model flagged this as missing test coverage for the behavior change. "
        "Reviewers: please verify."
    )

if not d["backwards_compatible"]["answer"]:
    pr.add_label("breaking-change")

# Dashboards: track severity + deploy-risk distributions across weeks
record_pr_signal(
    pr_id=pr.id,
    severity_dist=d["severity"]["distribution"],
    deploy_risk_dist=d["deploy_risk"]["distribution"],
)
```

Note that `category` is a 7-way choice and `severity` / `deploy_risk` are tiny-scale ratings (1–3 and 1–5). The right scale granularity is "the number of distinct actions you'd take" — don't ask for 1–10 severity if your action space is only `{merge-now, needs-senior-review, don't-deploy}`.

> **Why not a generative LLM here?** A diff can be 1000+ tokens and you want 6 different signals about it. One Bernoulli call processes the diff once and returns all six; six generative-LLM calls process the diff six times. On a repo with 50 PRs/day, this adds up — Bernoulli's prefix-sharing is the difference between a cheap signal and a budget line item.

---

## Patterns across use cases

A few things repeat across the four:

- **Multiple questions per request.** Every one of these use cases asks 4–6 questions against the same state. This is the natural shape; single-question requests are the exception.
- **Mix of question types.** Choice for routing / categorization, binary for policy / gating decisions, rating for prioritization or severity. The three types compose.
- **Thresholds, not label-matching.** Code branches on `confidence > 0.9` or `expected >= 4.0`, not on `answer == "..."`. The probability is the product.
- **Log the full distribution.** Not just the winner. The distribution across all options is the signal you'll need for calibration audits, drift detection, and retraining your thresholds later. If your data pipeline collapses it to one label at the write stage, you've thrown away the thing that makes Bernoulli more useful than a classifier.
- **Scale granularity matches your action space.** 1–3 rating when you have three actions; 1–5 when you have five. Don't ask for 1–10 if `expected >= 7.5` is the only threshold you'll use.

---

## How to validate on your own data

The best way to know if Bernoulli fits is to run the benchmark harness against your own labeled sample.

1. **Pick a shape**: find the benchmark in [`benchmarks/academic/`](benchmarks/academic/) or [`benchmarks/guardrails/`](benchmarks/guardrails/) that most looks like your workload (SST-2 for binary sentiment, Banking77 for high-cardinality intent, PAWS for binary semantic, Yelp for ordinal rating, ToxicChat for binary moderation, etc.).
2. **Write a loader**: follow [`benchmarks/academic/sst2/loader.py`](benchmarks/academic/sst2/loader.py) — it's one function that yields `BenchmarkExample(state, question, gold, source, meta)` from a JSONL of your examples.
3. **Run the harness**: one command per benchmark produces a `results/<date>.main.md` + JSON sidecar in the same shape as our public runs. Full methodology in [`benchmarks/README.md`](benchmarks/README.md).
4. **Compare**: the result file has Bernoulli, Generative (same backbone, text-and-parse), BGE-m3+LR columns side by side. If Bernoulli's NLL and ECE beat the alternatives on your data, you have your answer. If not, the file is still useful — it tells you which specific alternative to pick instead.

If after reading this doc you'd like help running the comparison on your corpus, [get in touch](mailto:contact@deepstore.ai) — we'll walk through what it would mean to use Bernoulli (or not) with your specific data.
