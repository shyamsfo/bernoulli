# Jev-via-OpenRouter demo — build brief

**This is a prompt doc for a separate Claude Code instance.** Open a fresh Claude Code session in this directory and point it at this file.

## Your goal

Build a web app ("jev-demo") that lets a visitor try **Jev** — TypeSafe AI's commercial decision model — through the OpenRouter API. The app must mirror the shape of an existing Gradio app that does the same thing for an open-source alternative. The user has already seen that reference app and liked the UX.

**Reference app**: [https://shyamsfo-bernoulli-demo.hf.space](https://shyamsfo-bernoulli-demo.hf.space)
Try it in a browser to see the look-and-feel. Three tabs, pre-canned examples, probability bar output, one "Show the API request" accordion per tab.

**What is Jev**: a hosted decision model from TypeSafe AI. Takes unstructured state + typed questions, returns calibrated probabilities. Available via the OpenRouter API at provider slug `typesafe/jev` (verify the current slug on [openrouter.ai/models](https://openrouter.ai/models) when you start — model names drift).

**Why this demo exists**: the reference Bernoulli app shows what the technique looks like end-to-end for an open-source model. This demo surfaces the same experience for the paid, hosted alternative — a side-by-side invitation for users who want to compare them.

---

## 1 — What the app does

Three tabs, one per question type. The user picks a tab, types or picks an example, clicks "Decide", and sees a probability distribution over the typed answer options.

### Tab 1 — Binary (Yes / No)

User inputs: `state` (text block, ~5 lines), `question` (text block, ~2 lines).
Output: `{Yes: 0.72, No: 0.28}` rendered as a two-bar probability chart with labels.

### Tab 2 — Choice (one of N options)

User inputs: `state`, `question`, `options` (comma-separated list, 2–10 options expected).
Output: `{option_a: 0.52, option_b: 0.33, option_c: 0.15}` rendered as an N-bar chart in canonical order.

### Tab 3 — Rating (ordinal)

User inputs: `state`, `question`, `scale_low` (int, default 1), `scale_high` (int, default 5).
Output: `{1: 0.02, 2: 0.08, 3: 0.56, 4: 0.28, 5: 0.06}` rendered as an N-bar chart over the integer scale.

### Dev-affordance panel (per tab)

Each tab has a collapsible "Show the API request" accordion below the output. Inside:
- A `curl` command showing exactly what you POSTed to OpenRouter (full body inline so copy-paste works).
- The raw JSON response OpenRouter returned.

This makes the app pedagogically useful: a developer can see the full API contract in one glance.

---

## 2 — OpenRouter API basics

OpenRouter speaks OpenAI's chat-completions API shape. One POST, Bearer-auth with an API key you keep server-side.

```bash
curl -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "typesafe/jev",       # ← verify current slug
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "response_format": {"type": "json_schema", "json_schema": {...}}
  }'
```

Jev's actual API shape (prompt + response contract) is documented at [docs.typesafe.ai/concepts/system-one](https://docs.typesafe.ai/concepts/system-one). **Read that before writing the handlers** — it tells you how to frame the state + question + options so Jev produces its typed output.

### Auth

- Store the OpenRouter API key as a server-side env var (`OPENROUTER_API_KEY`). **Never ship it to the browser.**
- The frontend talks to your Python backend; your backend talks to OpenRouter.

---

## 3 — Suggested stack

Pick based on your taste; any of these works:

**Option A — FastAPI + vanilla HTML/CSS/JS** (my recommendation)
- FastAPI backend at `/api/decide/{binary,choice,rating}` endpoints
- Single `index.html` with three `<div>` tabs, vanilla JS for the fetch calls
- CSS from scratch or Tailwind if you like
- One file per layer; easy to reason about

**Option B — Flask or Starlette** (if you prefer those)
- Same shape, different framework

**Option C — Gradio Blocks** (if you want to skip HTML/CSS entirely)
- You'd basically be porting the reference Space but swapping the backend from `/v1/decide` → OpenRouter
- Easiest path if you're OK with Gradio's look-and-feel

Option A is what the user had in mind when they framed this as "python + html + javascript or whatever web app."

---

## 4 — Pre-canned examples (verbatim from the reference app — reuse these)

Keep these as a "Try an example" dropdown or button grid in each tab. They were chosen to showcase typical decision shapes.

### Binary examples

```python
BINARY_EXAMPLES = [
    {
        "state": "A customer email: 'I ordered a blender three weeks ago and it still hasn't arrived. I want my money back and I'm never buying from you again.'",
        "question": "Does this message express strong frustration with the company?",
    },
    {
        "state": "A paper's abstract: 'We propose a novel attention mechanism that reduces transformer memory from O(n²) to O(n log n) by exploiting a hierarchical token structure.'",
        "question": "Is this abstract about transformer efficiency rather than application performance?",
    },
    {
        "state": "A one-sentence incident report: 'The staging database replica fell behind primary by 12 hours overnight and had to be rebuilt from scratch.'",
        "question": "Should this be flagged as a production-impacting incident?",
    },
]
```

### Choice examples

```python
CHOICE_EXAMPLES = [
    {
        "state": "The Lakers beat the Celtics 112-104 in overtime last night behind LeBron's 42-point effort.",
        "question": "Which news category does this story belong to?",
        "options": ["world", "sports", "business", "sci/tech"],
    },
    {
        "state": "A user message: 'I lost my credit card at the restaurant last night. Can you help me block it?'",
        "question": "Which support category does this ticket fall into?",
        "options": ["lost_card", "fraud_dispute", "balance_inquiry", "general_info", "other"],
    },
    {
        "state": "An abstract: 'We present a self-supervised method for learning robust visual representations from unlabeled video via temporal contrastive loss.'",
        "question": "Which arXiv category?",
        "options": ["cs.CL", "cs.CV", "cs.LG", "cs.RO"],
    },
]
```

### Rating examples

```python
RATING_EXAMPLES = [
    {
        "state": "The food was fine, service was slow but friendly, pretty good value for the price.",
        "question": "How many stars would this review likely give (1 = terrible, 5 = perfect)?",
        "scale_low": 1,
        "scale_high": 5,
    },
    {
        "state": "An incident: One user cannot download a PDF attachment, but opening it in the browser works.",
        "question": "Rate the severity (1 = cosmetic, 2 = workaround available, 3 = many users blocked, 4 = data loss).",
        "scale_low": 1,
        "scale_high": 4,
    },
    {
        "state": "A code review comment: 'This looks fine but the variable name could be clearer. Not a blocker.'",
        "question": "Rate the urgency (1 = ship it, 2 = optional nitpick, 3 = should fix, 4 = must fix before merge).",
        "scale_low": 1,
        "scale_high": 4,
    },
]
```

---

## 5 — UI copy (reuse verbatim)

### Intro / hero markdown (replace the backbone line with the Jev model slug)

```markdown
# Jev decision model demo

**Give it a piece of state and a typed question. Get a probability per option.**
Not a chat. Not a classifier. One API call per decision — the whole distribution
comes back, calibrated.

Backbone: `typesafe/jev` via [OpenRouter](https://openrouter.ai).

Each decision expands into a **"Show the API request"** panel with the exact
`curl` + response JSON — copy-paste to reproduce.

> **Session cap applies** to keep API costs bounded on this demo. Reach your
> cap? Refresh the page.
```

### Per-tab explainers

**Binary**: "The simplest decision: is this one thing true or not? The probability on 'Yes' is what you'd branch on. A well-calibrated 0.72 means ~72% of similar prompts should turn out to be 'Yes'."

**Choice**: "Pick one of N labels, with a probability for each. The full distribution matters — if the top option is 0.4 and the second is 0.35, that's not a confident decision, and your branching logic should know."

**Rating**: "Ordinal rating — the 1-to-N kind. The whole distribution comes back, so you can compute an expected rating (Σ p_i · i) or just read the probability mass near the top of the scale directly."

---

## 6 — Non-goals (don't do these)

- **No auth** on the frontend. The OpenRouter key is server-side only.
- **No persistence**. Each session is stateless beyond a request counter.
- **No streaming**. The decide path is one-shot; return the full distribution in one response.
- **No image path**. Text-only; same as the reference app.
- **Don't ship Bernoulli's "native probs vs verbalized probs" positioning** on this demo. The point of this project is to show Jev working, not to pitch an alternative. The reference app has its own positioning on that.

---

## 7 — Honest technical note to carry in the explainer

OpenRouter's Jev endpoint is a **text-generation API** that returns verbalized JSON. The model emits a JSON object with its probability distribution, which your backend parses. This differs from a "native-probs" implementation (where logits over label tokens are read directly from a running model) — but for a demo audience the UX is identical, and Jev's calibration is reportedly good.

If a visitor asks "how does this compare to [open-source alternative]", you can link out to [bernoulli.live/benchmarks](https://www.bernoulli.live/benchmarks.html) which carries side-by-side benchmark numbers.

---

## 8 — Suggested deliverable layout

```
jev-demo/
├── README.md             # how to run + deploy
├── pyproject.toml        # uv-managed deps: fastapi, httpx, python-dotenv
├── .env.example          # OPENROUTER_API_KEY= (gitignored real .env)
├── src/
│   ├── __init__.py
│   ├── app.py            # FastAPI app, three /api/decide/* endpoints
│   ├── openrouter.py     # client for OpenRouter chat/completions
│   └── schemas.py        # pydantic request/response models
├── static/
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── tests/
    └── test_openrouter.py  # mock the OpenRouter response, verify parsing
```

---

## 9 — Success criteria

The demo is done when:

1. You can click "Try an example" on any tab, submit, and see a probability bar chart in <5 s.
2. Opening the "Show the API request" accordion reveals a copy-pasteable curl that reproduces the result.
3. All three tabs work on both desktop and mobile browsers.
4. The OpenRouter API key is server-side-only (verify by viewing the deployed page source — no key should be visible).
5. The deployed URL survives a page refresh and a 10-minute idle period.

Don't over-polish the CSS. A clean two-column layout per tab (inputs on the left, output + accordion on the right) matching the reference app is enough.

---

## 10 — Where to publish

The user plans to deploy this separately from the reference Bernoulli app. Suggested paths:

- **Fly.io** — small Python app, free tier covers casual traffic
- **Render** — similar
- **Vercel** — if you go with a Next.js version instead of FastAPI (your call)
- **The user's own box** — if they prefer self-hosted

Pick whichever has the fastest "git push → URL" loop. The user can swap infra later if needed.

---

## Reference links to include in the final README of your deliverable

- Reference Gradio app (what we're mirroring the UX of): https://shyamsfo-bernoulli-demo.hf.space
- Reference app source (Gradio Python, for context — NOT a copy target): https://huggingface.co/spaces/shyamsfo/bernoulli-demo/blob/main/app.py
- Benchmark comparisons (for the honest-technical-note link): https://www.bernoulli.live/benchmarks.html
- OpenRouter model catalog: https://openrouter.ai/models
- Jev product docs: https://docs.typesafe.ai/concepts/system-one

Good luck. Ask the user if the OpenRouter model slug differs from `typesafe/jev` when you start — it's the one thing that may have drifted.
