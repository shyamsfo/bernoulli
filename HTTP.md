# Bernoulli — HTTP guide

<!-- If you edit this, consider whether CLI.md needs the same edit. -->

Everything you need to call Bernoulli over HTTP. If you're calling it from a shell instead of a service, see [`CLI.md`](CLI.md) — same engine, same request shape, just no HTTP envelope.

**Prereq.** The server is running at `http://localhost:8000` (or wherever you pointed it). Start it with `just serve` (dev / HFScorer) or `just docker-build && just docker-run` (vLLM in Docker). `curl` is enough to call it.

---

## Which kind of question do I want?

Bernoulli answers three kinds of questions. Pick the one that matches your problem:

| You want to…                                 | Use      |
|----------------------------------------------|----------|
| pick exactly one of several named things     | **choice** |
| get a yes/no answer                          | **binary** |
| score something on an integer scale (e.g. 1–5) | **rating** |

Every request has a `state` (the thing you want to reason about — text today, images in a future release) and a list of `questions`. You can ask multiple questions at once — Bernoulli answers all of them against the same state in one engine call.

---

## Choice

**Use when** you have 2–128 named options and want the model to pick one. The response tells you which option won, how confident the model is, and the full probability distribution across all options.

Request fields that matter:
- `prompt` — the question you want answered
- `options` — the list of possible answers, as strings

### Example — customer intent

Suppose a support ticket just came in and you want to classify what the customer wants.

```bash
curl -sS -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "I ordered a jacket two weeks ago and nothing has arrived. Where is my package??"},
    "questions": [{
      "id": "intent",
      "type": "choice",
      "prompt": "What does the customer want?",
      "options": ["refund", "exchange", "tracking", "other"]
    }]
  }'
```

Response:

```json
{
  "decisions": {
    "intent": {
      "type": "choice",
      "answer": "tracking",
      "confidence": 0.942,
      "distribution": {
        "refund": 0.015,
        "exchange": 0.001,
        "tracking": 0.942,
        "other": 0.041
      }
    }
  },
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "calibration_version": null,
  "latency_ms": 76
}
```

- **`answer`** is the winning option.
- **`confidence`** is the probability the model assigned to the winner. Here, 94%.
- **`distribution`** is the full breakdown — route "tracking" to a specific workflow with high confidence, but if `distribution["refund"]` had been meaningful (say > 0.2), you'd probably want a human in the loop.

### Example — content moderation

Suppose you're classifying forum posts before publishing them.

```bash
curl -sS -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "GET RICH QUICK!!! CLICK HERE TO MAKE $5000/DAY WORKING FROM HOME!!!"},
    "questions": [{
      "id": "category",
      "type": "choice",
      "prompt": "Which content category best describes this post?",
      "options": ["safe", "spam", "abuse", "off_topic"]
    }]
  }'
```

Response (abridged):

```json
{
  "decisions": {
    "category": {
      "type": "choice",
      "answer": "spam",
      "confidence": 0.981,
      "distribution": {"safe": 0.001, "spam": 0.981, "abuse": 0.012, "off_topic": 0.006}
    }
  },
  ...
}
```

Branch on `confidence` — if it's above 0.9 and `answer == "spam"`, auto-reject. If it's between 0.5 and 0.9, send to the review queue. Below that, publish and move on. **This is the whole point of returning a calibrated distribution instead of just a label — you can set thresholds that reflect your risk tolerance.**

---

## Binary

**Use when** the answer is yes or no. The response gives you a boolean plus the probability of "yes".

Request fields that matter:
- `prompt` — the yes/no question

### Example — "is this urgent?"

Suppose you're triaging a support queue and want to pull urgent items to the top.

```bash
curl -sS -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "I keep getting charged for a subscription I canceled 3 months ago. My next payment is tomorrow. Please help!!"},
    "questions": [{
      "id": "urgent",
      "type": "binary",
      "prompt": "Is this urgent? (Does it need handling in the next 24 hours?)"
    }]
  }'
```

Response:

```json
{
  "decisions": {
    "urgent": {
      "type": "binary",
      "answer": true,
      "probability": 0.887
    }
  },
  ...
}
```

- **`answer`** is `true` if `probability >= 0.5`, otherwise `false`.
- **`probability`** is the model's probability that the answer is "yes". A `probability` of 0.52 is technically "yes" but barely; 0.95 is a clear yes.

### Example — guardrail check

Suppose you want to check whether an LLM's output follows a policy before showing it to a user.

```bash
curl -sS -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "I cannot advise on how to pick a lock, as that is often associated with illegal activity. If you are locked out, please contact a professional locksmith."},
    "questions": [{
      "id": "safe",
      "type": "binary",
      "prompt": "Is the response safe to show to the user? (Refuses the harmful request politely without revealing dangerous info?)"
    }]
  }'
```

Response:

```json
{
  "decisions": {
    "safe": {"type": "binary", "answer": true, "probability": 0.952}
  },
  ...
}
```

This is the "LLM as a judge" pattern — but because Bernoulli reads the probability directly rather than parsing a yes/no out of generated text, you get a probability you can threshold on. Set `probability >= 0.9` to auto-ship, `0.5-0.9` to log-and-ship, `< 0.5` to block.

---

## Rating

**Use when** you want a score on an integer scale. The response gives you the full probability distribution over scale points plus the expected value (what you'd report if you had to collapse it to a single number).

Request fields that matter:
- `prompt` — the question
- `scale` — the scale as `[low, high]`, inclusive. Width must be ≤ 9 (so `[1, 9]` is the biggest).

### Example — customer anger (1–5)

Suppose you want to triage an email based on how upset the customer sounds.

```bash
curl -sS -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "I AM FURIOUS. THIS IS THE THIRD TIME I HAVE EMAILED YOU. UNACCEPTABLE."},
    "questions": [{
      "id": "anger",
      "type": "rating",
      "prompt": "On a scale of 1 (calm) to 5 (furious), how angry does this customer sound?",
      "scale": [1, 5]
    }]
  }'
```

Response:

```json
{
  "decisions": {
    "anger": {
      "type": "rating",
      "expected": 4.65,
      "distribution": {"1": 0.01, "2": 0.03, "3": 0.08, "4": 0.26, "5": 0.62}
    }
  },
  ...
}
```

- **`expected`** is `Σ probability × value` — the probability-weighted average score. Here, 4.65 out of 5.
- **`distribution`** gives you the full picture: 88% of the probability mass is on 4 or 5. You can threshold on `expected` for routing (e.g., `expected >= 4.0` → escalate to a human), or inspect the distribution if you need to see uncertainty.

### Example — code review severity (1–3)

Suppose you want a quick rough-grade on a code change before a human looks at it.

```bash
curl -sS -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "diff --git a/auth.py b/auth.py\n-    if password == stored_pw:\n+    if hmac.compare_digest(password, stored_pw):"},
    "questions": [{
      "id": "severity",
      "type": "rating",
      "prompt": "How impactful is this change? 1 = cosmetic, 2 = normal fix, 3 = security/correctness-critical.",
      "scale": [1, 3]
    }]
  }'
```

Response:

```json
{
  "decisions": {
    "severity": {
      "type": "rating",
      "expected": 2.78,
      "distribution": {"1": 0.03, "2": 0.16, "3": 0.81}
    }
  },
  ...
}
```

The small scale (1–3) is handy when you only need rough bins. Anything ≤ 9 wide works; use the right granularity for your decision, not more.

---

## Asking multiple questions in one request

Put them all in the `questions` array. Bernoulli answers them against the same state in one engine call — much cheaper than issuing N separate HTTP requests, because the state prompt is tokenized once and the engine batches the forward passes.

```bash
curl -sS -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "Customer email: Where is my package? I ordered it two weeks ago and nothing has arrived. This is unacceptable service."},
    "questions": [
      {
        "id": "intent",
        "type": "choice",
        "prompt": "What does the customer want?",
        "options": ["refund", "exchange", "tracking", "other"]
      },
      {"id": "urgent", "type": "binary", "prompt": "Is this urgent?"},
      {"id": "anger", "type": "rating", "prompt": "How angry is the customer?", "scale": [1, 5]}
    ]
  }'
```

Response: one `decisions` entry per question, keyed by the `id` you supplied.

```json
{
  "decisions": {
    "intent": {"type": "choice", "answer": "tracking", "confidence": 0.942, "distribution": {...}},
    "urgent": {"type": "binary", "answer": true,       "probability": 0.947},
    "anger":  {"type": "rating", "expected": 4.30,     "distribution": {...}}
  },
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "calibration_version": null,
  "latency_ms": 368
}
```

On the current dev box (A10G 24GB, Qwen2.5-VL-7B), expect ~70 ms per added question with the default reverse debias.

---

## Request options

Tweak scoring per request via the `options` field:

```json
"options": {
  "debias": "reverse",
  "calibrated": true
}
```

- **`debias`** — the model has a mild position bias: all else equal, it slightly prefers whichever option appears first. Bernoulli corrects by rerunning the request with permuted option orders and averaging:
  - `"none"` — single forward pass. Cheapest. Use when you don't care about calibrated probabilities or you have ≤ 2 options and position doesn't matter.
  - `"reverse"` — **default.** One extra pass with options reversed. 2× cost, catches most of the bias.
  - `"cyclic"` — one pass per cyclic shift (k passes for k options, capped at 6). Most thorough, most expensive.
- **`calibrated`** — if the server was started with `BERNOULLI_CALIBRATION_PATH` set to a Calibration JSON, setting `calibrated: true` applies the fitted temperature per question type. Noticeably sharper probabilities on tasks the model is overconfident on. If no calibration file is loaded, this is a no-op and `response.calibration_version` is `null`.

---

## The other endpoints

```bash
# liveness
curl http://localhost:8000/healthz
# → {"status":"ok"}

# the loaded model
curl http://localhost:8000/v1/models
# → {"models":[{"id":"Qwen/Qwen2.5-VL-7B-Instruct","revision":"cc594898..."}]}
```

---

## Response fields at a glance

Every `/v1/decide` response is shaped the same way:

```json
{
  "decisions": {
    "<question_id>": { ... Decision ... }
  },
  "model": "<HF id>",
  "calibration_version": "<ISO date or null>",
  "latency_ms": 368
}
```

The `Decision` inside varies by question type — `ChoiceDecision` has `answer`/`confidence`/`distribution`, `BinaryDecision` has `answer`/`probability`, `RatingDecision` has `expected`/`distribution`.

---

## Troubleshooting

- **422 Unprocessable Entity** — the request payload didn't validate. Common causes: missing `prompt`, missing `options` for a choice, `scale` where `high <= low`, unknown fields (we use `extra='forbid'` on all models). The error body shows which field.
- **501 Not Implemented** — you sent images in the state. Image modality lands in M6; right now `state.images` returns 501.
- **Timeouts** — first call after a cold start includes model load (~2 min) and Triton JIT compile (~30 s). Subsequent calls are tens-to-hundreds of ms. Warm the server with a dummy request before benchmarking.
- **Max option count** — `choice` goes up to 128 options; above 26 the server uses chunked scoring (one engine call per 26-option chunk).
- **Rating scale width** — must be ≤ 9. Internally mapped to the 9 single-token labels A–I.

See [`CLAUDE.md`](CLAUDE.md) for operational caveats on the dev box (vLLM memory, zombie engine processes, Docker storage).
