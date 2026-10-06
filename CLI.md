# Bernoulli — CLI guide

<!-- If you edit this, consider whether HTTP.md needs the same edit. -->

Everything you need to call Bernoulli from a shell. If you're calling it from a service or across a network, see [`HTTP.md`](HTTP.md) — same engine, same request shape, just wrapped in HTTP.

**Prereq.** The `bernoulli` CLI is installed in your project venv (`uv sync` on the dev box or `pip install -e .[scorer-hf]` elsewhere). The CLI loads the model into memory on each invocation, so expect a ~2 min warm-up on the first call. For a long-running interactive loop, use the [HTTP server](HTTP.md) instead and let the model stay resident.

Run `uv run bernoulli decide --help` for the full option list. The two shapes:

```bash
# Form A — single question, with state + question in separate files
uv run bernoulli decide --state state.txt --question question.json

# Form B — one JSON blob with the full request
uv run bernoulli decide --request request.json
```

Form A is handy for ad-hoc runs; Form B is handy when you're scripting a batch or want to send multiple questions at once.

---

## Which kind of question do I want?

Bernoulli answers three kinds of questions. Pick the one that matches your problem:

| You want to…                                 | Use      |
|----------------------------------------------|----------|
| pick exactly one of several named things     | **choice** |
| get a yes/no answer                          | **binary** |
| score something on an integer scale (e.g. 1–5) | **rating** |

Every request has a `state` (the thing you want to reason about — text today, images in a future release) and a list of `questions`. You can ask multiple questions at once — Bernoulli answers all of them against the same state in one engine call.

> Examples below are deliberately small so you can copy-paste and run them. For realistic use cases that combine multiple question types against a document-sized state (support thread, LLM output, code diff + context), see [`USECASES.md`](USECASES.md).

---

## Choice

**Use when** you have 2–128 named options and want the model to pick one. The response tells you which option won, how confident the model is, and the full probability distribution across all options.

Question fields that matter:
- `prompt` — the question you want answered
- `options` — the list of possible answers, as strings

### Example — customer intent

Suppose a support ticket just came in and you want to classify what the customer wants.

```bash
cat > /tmp/state.txt <<'EOF'
I ordered a jacket two weeks ago and nothing has arrived. Where is my package??
EOF

cat > /tmp/question.json <<'EOF'
{
  "id": "intent",
  "type": "choice",
  "prompt": "What does the customer want?",
  "options": ["refund", "exchange", "tracking", "other"]
}
EOF

uv run bernoulli decide --state /tmp/state.txt --question /tmp/question.json
```

Output:

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
- **`distribution`** is the full breakdown. Pipe it into `jq '.decisions.intent.distribution'` to pick off specific entries, or branch on `confidence` to decide whether to accept the answer automatically or kick to a human.

### Example — content moderation

Suppose you're pre-filtering forum posts.

```bash
cat > /tmp/state.txt <<'EOF'
GET RICH QUICK!!! CLICK HERE TO MAKE $5000/DAY WORKING FROM HOME!!!
EOF

cat > /tmp/question.json <<'EOF'
{
  "id": "category",
  "type": "choice",
  "prompt": "Which content category best describes this post?",
  "options": ["safe", "spam", "abuse", "off_topic"]
}
EOF

uv run bernoulli decide --state /tmp/state.txt --question /tmp/question.json | jq .decisions.category
```

Output:

```json
{
  "type": "choice",
  "answer": "spam",
  "confidence": 0.981,
  "distribution": {"safe": 0.001, "spam": 0.981, "abuse": 0.012, "off_topic": 0.006}
}
```

Branch on `confidence` — if it's above 0.9 and `answer == "spam"`, auto-reject. If it's between 0.5 and 0.9, send to the review queue. Below that, publish and move on. **This is the whole point of returning a calibrated distribution instead of just a label — you can set thresholds that reflect your risk tolerance.**

---

## Binary

**Use when** the answer is yes or no. The response gives you a boolean plus the probability of "yes".

Question fields that matter:
- `prompt` — the yes/no question

### Example — "is this urgent?"

Suppose you're triaging a support queue and want to pull urgent items to the top.

```bash
cat > /tmp/state.txt <<'EOF'
I keep getting charged for a subscription I canceled 3 months ago. My next payment is tomorrow. Please help!!
EOF

cat > /tmp/question.json <<'EOF'
{
  "id": "urgent",
  "type": "binary",
  "prompt": "Is this urgent? (Does it need handling in the next 24 hours?)"
}
EOF

uv run bernoulli decide --state /tmp/state.txt --question /tmp/question.json | jq .decisions.urgent
```

Output:

```json
{"type": "binary", "answer": true, "probability": 0.887}
```

- **`answer`** is `true` if `probability >= 0.5`, otherwise `false`.
- **`probability`** is the model's probability that the answer is "yes". A `probability` of 0.52 is technically "yes" but barely; 0.95 is a clear yes.

### Example — guardrail check

Suppose you want to check whether an LLM's output follows a policy before showing it to a user.

```bash
cat > /tmp/state.txt <<'EOF'
I cannot advise on how to pick a lock, as that is often associated with illegal activity. If you are locked out, please contact a professional locksmith.
EOF

cat > /tmp/question.json <<'EOF'
{
  "id": "safe",
  "type": "binary",
  "prompt": "Is the response safe to show to the user? (Refuses the harmful request politely without revealing dangerous info?)"
}
EOF

uv run bernoulli decide --state /tmp/state.txt --question /tmp/question.json | jq .decisions.safe
```

Output:

```json
{"type": "binary", "answer": true, "probability": 0.952}
```

This is the "LLM as a judge" pattern — but because Bernoulli reads the probability directly rather than parsing a yes/no out of generated text, you get a probability you can threshold on. Pipe it into your deploy script: `probability >= 0.9` → auto-ship, `0.5-0.9` → log-and-ship, `< 0.5` → block.

---

## Rating

**Use when** you want a score on an integer scale. The response gives you the full probability distribution over scale points plus the expected value (what you'd report if you had to collapse it to a single number).

Question fields that matter:
- `prompt` — the question
- `scale` — the scale as `[low, high]`, inclusive. Width must be ≤ 9 (so `[1, 9]` is the biggest).

### Example — customer anger (1–5)

Suppose you want to triage an email based on how upset the customer sounds.

```bash
cat > /tmp/state.txt <<'EOF'
I AM FURIOUS. THIS IS THE THIRD TIME I HAVE EMAILED YOU. UNACCEPTABLE.
EOF

cat > /tmp/question.json <<'EOF'
{
  "id": "anger",
  "type": "rating",
  "prompt": "On a scale of 1 (calm) to 5 (furious), how angry does this customer sound?",
  "scale": [1, 5]
}
EOF

uv run bernoulli decide --state /tmp/state.txt --question /tmp/question.json | jq .decisions.anger
```

Output:

```json
{
  "type": "rating",
  "expected": 4.65,
  "distribution": {"1": 0.01, "2": 0.03, "3": 0.08, "4": 0.26, "5": 0.62}
}
```

- **`expected`** is `Σ probability × value` — the probability-weighted average score. Here, 4.65 out of 5.
- **`distribution`** gives you the full picture: 88% of the probability mass is on 4 or 5. You can threshold on `expected` for routing (e.g., `jq '.decisions.anger.expected >= 4.0'` to escalate to a human), or inspect the distribution if you need to see uncertainty.

### Example — code review severity (1–3)

Suppose you want a rough grade on a code change.

```bash
cat > /tmp/state.txt <<'EOF'
diff --git a/auth.py b/auth.py
-    if password == stored_pw:
+    if hmac.compare_digest(password, stored_pw):
EOF

cat > /tmp/question.json <<'EOF'
{
  "id": "severity",
  "type": "rating",
  "prompt": "How impactful is this change? 1 = cosmetic, 2 = normal fix, 3 = security/correctness-critical.",
  "scale": [1, 3]
}
EOF

uv run bernoulli decide --state /tmp/state.txt --question /tmp/question.json | jq .decisions.severity
```

Output:

```json
{"type": "rating", "expected": 2.78, "distribution": {"1": 0.03, "2": 0.16, "3": 0.81}}
```

The small scale (1–3) is handy when you only need rough bins. Anything ≤ 9 wide works; use the right granularity for your decision, not more.

---

## Asking multiple questions in one request

Pass a JSON **array** of questions instead of a single object. Bernoulli answers them against the same state in one engine call — much cheaper than running `bernoulli decide` three times, because the state prompt is tokenized once and the engine batches the forward passes.

```bash
cat > /tmp/state.txt <<'EOF'
Customer email: Where is my package? I ordered it two weeks ago and nothing has arrived. This is unacceptable service.
EOF

cat > /tmp/questions.json <<'EOF'
[
  {
    "id": "intent",
    "type": "choice",
    "prompt": "What does the customer want?",
    "options": ["refund", "exchange", "tracking", "other"]
  },
  {"id": "urgent", "type": "binary", "prompt": "Is this urgent?"},
  {"id": "anger", "type": "rating", "prompt": "How angry is the customer?", "scale": [1, 5]}
]
EOF

uv run bernoulli decide --state /tmp/state.txt --question /tmp/questions.json
```

Output: one `decisions` entry per question, keyed by the `id` you supplied.

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

## Full-request form

For scripting a batch, build a complete `DecideRequest` JSON and pass it with `--request`:

```bash
cat > /tmp/request.json <<'EOF'
{
  "state": {"text": "Customer email: Where is my package? ..."},
  "questions": [
    {"id": "intent", "type": "choice", "prompt": "What does the customer want?",
     "options": ["refund", "exchange", "tracking", "other"]}
  ],
  "options": {"debias": "reverse", "calibrated": true}
}
EOF

uv run bernoulli decide --request /tmp/request.json
```

This form takes a `options` block too, which the shorthand (`--state` + `--question`) doesn't. See below for what the knobs do.

---

## Request options

Tweak scoring per request via the `options` block (only available in the `--request` form):

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
- **`calibrated`** — if a Calibration JSON is on disk and `BERNOULLI_CALIBRATION_PATH` env var points at it, setting `calibrated: true` applies the fitted temperature per question type. Noticeably sharper probabilities on tasks the model is overconfident on. If no calibration file is loaded, this is a no-op and `calibration_version` is `null`.

---

## Config

The CLI reads from env, prefix `BERNOULLI_` (see `bernoulli/config.py` for the full list). The ones you'll touch most:

| var | effect |
|---|---|
| `BERNOULLI_MODEL_ID` | HF hub id of the backbone VLM. Override the default via `--model <id>` on the CLI. |
| `BERNOULLI_SCORER` | `hf` (transformers, dev) or `vllm` (production). `vllm` is much faster but takes ~2 min to warm up. |
| `BERNOULLI_MAX_MODEL_LEN` | Context limit. On a 24 GB GPU running vLLM, set to `8192`. |
| `BERNOULLI_CALIBRATION_PATH` | Path to a Calibration JSON for `calibrated=true` requests. |

---

## Response fields at a glance

Every `bernoulli decide` output is shaped the same way:

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

Pipe into `jq` to pick off the bits you need:

```bash
# just the answer
uv run bernoulli decide --state s.txt --question q.json | jq -r .decisions.intent.answer

# only route when confidence is high enough
confidence=$(uv run bernoulli decide --state s.txt --question q.json | jq .decisions.intent.confidence)
if [ "$(echo "$confidence > 0.9" | bc)" = "1" ]; then route; else queue; fi
```

---

## Troubleshooting

- **Validation error (pydantic)** — the request payload didn't validate. Common causes: missing `prompt`, missing `options` for a choice, `scale` where `high <= low`, unknown fields (we use `extra='forbid'` on all models). The error output shows which field.
- **`NotImplementedError: image states land in M4`** — you passed images in the state. Image modality lands in M6; right now `state.images` raises.
- **First call is slow** — includes model load (~2 min on HFScorer, longer on vLLM with Triton JIT). Subsequent calls in the same process are fast, but the CLI reloads on each invocation. For repeated calls, use the HTTP server (`just serve`) and keep the model resident.
- **Max option count** — `choice` goes up to 128 options; above 26 the engine uses chunked scoring (one engine call per 26-option chunk).
- **Rating scale width** — must be ≤ 9. Internally mapped to the 9 single-token labels A–I.

See [`CLAUDE.md`](CLAUDE.md) for operational caveats on the dev box.
