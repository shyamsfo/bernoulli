# Bernoulli HTTP API — reference

A compact, self-contained reference for the Bernoulli HTTP API. Target reader: a developer (or another LLM agent) implementing a client against the production endpoint without needing to read the rest of the repo.

For a tutorial walkthrough with worked examples per question type, read [`HTTP.md`](HTTP.md) instead. For deeper architecture / "why this instead of a chat endpoint", read [`HOW-IT-WORKS.md`](HOW-IT-WORKS.md).

**Audience split:**
- Writing a client → **this file is enough**.
- Deciding whether to use Bernoulli → [`USECASES.md`](USECASES.md).
- Operating the server → [`CLI.md`](CLI.md) + the top-level `justfile`.

## Table of contents

- [Base URL, versioning, and transport](#base-url-versioning-and-transport)
- [Authentication](#authentication)
- [Endpoints at a glance](#endpoints-at-a-glance)
- [Request schema — `/v1/decide`](#request-schema--v1decide)
- [Response schema — `/v1/decide`](#response-schema--v1decide)
- [Other endpoints](#other-endpoints)
- [Error codes](#error-codes)
- [Rate limits and SLAs](#rate-limits-and-slas)
- [Worked example](#worked-example)
- [Client snippets](#client-snippets)
- [Differences from Jev's API](#differences-from-jevs-api)
- [Versioning policy](#versioning-policy)

---

## Base URL, versioning, and transport

| | |
|---|---|
| **Production** | `https://bernoulli-api.nuwire.dev` |
| **Staging** | `https://bernoulli-api-staging.nuwire.dev` (same backend as prod; swap-mechanism test only) |
| **Local dev** | `http://localhost:8000` after `just serve-start` (daily/prod path) or `just dev serve-local` (local uvicorn) |
| **Protocol** | HTTPS 1.1 / 2 (ssd2 nginx terminates TLS → plain HTTP to the backend) |
| **Media type** | `application/json`, UTF-8 |
| **Max request body** | 32 MB (nginx `client_max_body_size`) |
| **Timeout** | 60 s request-to-response end-to-end |
| **Versioning** | URL-pinned (`/v1/`). Breaking changes get `/v2/`; see [Versioning policy](#versioning-policy) |

All endpoints are hosted by a single FastAPI service. The HTTPS layer runs on a separate TLS-termination box (ssd2) and reverse-proxies to the backend's `:8000`; this is operationally invisible to clients.

---

## Authentication

API-key authentication via the `Authorization` header on every endpoint **except** `/healthz`.

```
Authorization: Bearer <api-key>
```

- **Key format**: opaque 64-character hex string (SHA-256-length — 256 bits of entropy per key).
- **Key provisioning**: out-of-band (contact the operator; no self-service enrollment).
- **Multiple keys**: the server accepts any key from its configured set. Clients should treat the key as a shared secret, not a per-user identity.
- **Rotation**: no key rotation API today. Rotation requires an operator restart.
- **401 responses**:
  - `{"detail": "missing bearer token"}` — no `Authorization` header, or wrong scheme (not `Bearer`).
  - `{"detail": "invalid api key"}` — key not in the server's configured set.

Keys are **not** scoped to specific endpoints or question types; each key grants full access.

---

## Endpoints at a glance

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/healthz` | No | Liveness probe. Returns `{"status": "ok"}` when the model is loaded. |
| `GET` | `/v1/models` | Yes | Which backbone + revision is loaded. |
| `POST` | `/v1/decide` | Yes | **The main endpoint.** State + typed questions → calibrated probabilities. |
| `POST` | `/v1/generate` | Yes | Baseline/comparison path. Same request shape as `/v1/decide`, but produces its "distribution" by generating a short text completion and parsing it (1.0/0.0 point mass on parsed answer; uniform on parse failure). Only works when the loaded scorer has a `.generate()` method — `BERNOULLI_SCORER=hf` yes, `BERNOULLI_SCORER=vllm` no (returns 501). |

---

## Request schema — `/v1/decide`

### `DecideRequest`

```json
{
  "state": State,                      // required
  "questions": [Question, ...],        // required; min 1
  "options": DecideOptions             // optional; defaults below
}
```

Unknown fields → **422 Unprocessable Entity**. The pydantic model has `extra="forbid"` throughout.

### `State`

```json
{
  "text": string,                      // optional; default ""
  "images": [string, ...]              // optional; default []; max 8 items
}
```

- `text` is the full input the model reasons over. Length is bounded by the backbone's context window minus the question prompt + options (~32k tokens for Qwen2.5-VL-7B minus overhead).
- `images` are **not supported** in the current serving config (text-only backbone). Sending a non-empty list produces a 422. Image support lands with the multimodal milestone.

### `Question` — discriminated union by `type`

Every question requires `id`, `type`, and `prompt`. Type-specific fields follow.

#### `ChoiceQuestion`

```json
{
  "id": string,                        // echoed in the response's "decisions" dict as the key
  "type": "choice",
  "prompt": string,                    // the question text shown to the model
  "options": [string, string, ...]     // required; 2 ≤ len ≤ 128
}
```

- Option strings are matched verbatim; preserve case + whitespace.
- Options > 26 are automatically chunked server-side (letters A-Z are the raw token alphabet; chunks of 26 are stitched via a global softmax). Chunked scoring has a known limitation on option-reorder stability — see `benchmarks/README.md`.

#### `BinaryQuestion`

```json
{
  "id": string,
  "type": "binary",
  "prompt": string                     // phrase so a Yes/No answer is natural
}
```

No `options` field. The implied options are `Yes` and `No` as single tokens on the backbone's tokenizer; the response's `answer` is `true` / `false`.

#### `RatingQuestion`

```json
{
  "id": string,
  "type": "rating",
  "prompt": string,
  "scale": [low, high]                 // 2-tuple of ints, inclusive, len <= 9
}
```

- `scale: [1, 5]` means the model picks among 1, 2, 3, 4, 5.
- `high - low + 1` must be between 2 and 9 (we map ratings onto letters A-I; digits are two tokens on Qwen's tokenizer so letters are the invariant).
- Response gives `expected = Σ p_i · i` (the posterior mean) + the full distribution over the integers.

### `DecideOptions`

```json
{
  "debias": "none" | "reverse" | "cyclic",   // default "reverse"
  "calibrated": boolean                      // default true
}
```

- `"reverse"` (default): double-pass with reversed option order; cancels left-position bias on choice questions. 2× forward passes per question. Yes/No binary questions internally skip debias (reversing Yes/No is semantic, not positional).
- `"cyclic"`: for N ≤ 6 options, averages all N cyclic shifts. More thorough, N× forward passes.
- `"none"`: single pass, fastest, less calibrated.
- `calibrated`: whether to apply the per-question-type temperature from the loaded calibration file (none by default; set `BERNOULLI_CALIBRATION_PATH` on the server to enable). When set to `true` but no calibration is loaded, this is silently a no-op.

---

## Response schema — `/v1/decide`

### `DecideResponse`

```json
{
  "decisions": { "<question-id>": Decision, ... },  // keyed by the request's question.id
  "model": string,                                   // e.g. "Qwen/Qwen2.5-VL-7B-Instruct"
  "calibration_version": string | null,              // ISO date of the loaded calibration, if any
  "latency_ms": integer                              // server-side wall-clock, excludes network
}
```

The `decisions` dict preserves one entry per question in the request. Keys are the caller-supplied `question.id`. Order is **not** guaranteed to match request order — key your lookup by `id`.

### `Decision` — discriminated union by `type`

The type field in each decision matches the type of the question it answers.

#### `ChoiceDecision`

```json
{
  "type": "choice",
  "answer": string,                        // the winning option (argmax of distribution)
  "confidence": float,                     // = distribution[answer]; in [0, 1]
  "distribution": {                        // probability per option string; sums to 1.0 ± 1e-6
    "<option-1>": float,
    "<option-2>": float,
    ...
  }
}
```

#### `BinaryDecision`

```json
{
  "type": "binary",
  "answer": boolean,                       // true iff P(Yes) > 0.5
  "probability": float                     // P(answer == true); in [0, 1]
}
```

Note the asymmetry from choice: there's no `distribution` field because binary is a single scalar. If you want `P(No)`, compute `1 - probability`.

#### `RatingDecision`

```json
{
  "type": "rating",
  "expected": float,                       // Σ p_i · i; between scale[0] and scale[1]
  "distribution": {                        // probability per integer label (keys are string-cast ints)
    "<low>": float,
    ...,
    "<high>": float
  }
}
```

- Distribution keys are the integer labels serialized as strings (`"1"`, `"2"`, ...) because JSON objects require string keys.
- `expected` is the useful signal most of the time; `distribution` lets you compute other summaries (median, mode, confidence intervals).

---

## Other endpoints

### `GET /healthz`

```
GET /healthz
→ 200 {"status": "ok"}
```

Returns 200 **only after the model is loaded in VRAM.** During cold start (model loading, ~4-5 min for HFScorer, ~90 s for VLLMScorer), this endpoint is unreachable (connection refused or 502 from the TLS terminator).

Use for liveness probes and load-balancer health checks. No auth required.

### `GET /v1/models`

```
GET /v1/models
→ 200 {"models": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", "revision": "cc59489..."}]}
```

Reports which backbone is loaded. Returns exactly one model (we don't multi-host). The `revision` is a HuggingFace commit SHA; two servers with the same `id` + `revision` are byte-identical model weights.

### `POST /v1/generate`

Same request shape as `/v1/decide`, same response shape. Different *mechanism*: this endpoint produces its "distribution" by generating a short text completion and parsing it (1.0/0.0 point mass on parsed answer; uniform on parse failure). It exists as a **comparison baseline**, not a production path — use it when you want to measure the gap between prompt-and-parse and native-logit scoring on the same model.

**Only works when the server has `BERNOULLI_SCORER=hf`** (HFScorer has `.generate()`). On `BERNOULLI_SCORER=vllm` the endpoint returns 501 with a descriptive message.

---

## Error codes

| Status | Shape | When |
|---|---|---|
| **200** | `DecideResponse` | Success. |
| **401** | `{"detail": "missing bearer token"}` | No `Authorization` header or wrong auth scheme. |
| **401** | `{"detail": "invalid api key"}` | Bearer token not in server's configured key set. |
| **422** | `{"detail": [FastAPI validation detail]}` | Request violates the pydantic schema (missing field, wrong type, unknown field due to `extra="forbid"`, option count out of range, scale too wide, etc.). The `detail` is an array of per-field errors in FastAPI's standard shape. |
| **501** | `{"detail": "Loaded scorer (VLLMScorer) has no .generate(); use BERNOULLI_SCORER=hf for the generative baseline path."}` | Only from `/v1/generate` on a non-HF scorer. |
| **504** | *HTML (from the nginx TLS-terminator)* | Backend didn't respond within the TLS terminator's timeout. Usually means the backend is cold-starting (model still loading) or the ClouDNS A record for the internal name is pointing at a dead backend. Retry in ~90 s; if persistent, operator issue. |
| **5xx** | *FastAPI default error page* | Unexpected server error. Rare. Capture the response body and share with the operator. |

**Do not retry on 401 or 422.** They indicate a client bug, not transient failure.
**Retry on 504** with exponential backoff; cold-start can take minutes.

---

## Rate limits and SLAs

**None configured right now.** This is a single-tenant dev-tier endpoint; the operator and callers coordinate out-of-band. The server will accept as many concurrent requests as the hardware can handle (A10G 24 GB → one in-flight request processed cleanly, additional requests queued).

**Realistic latency** on the current `BERNOULLI_SCORER=hf` setup, Qwen 7B, A10G 24 GB:
- 1 question / request: **~70-150 ms** (p50 single-call), 300-500 ms first-call cold (CUDA JIT warmup).
- 5 questions / request: **~368 ms** (p50).
- 20 questions / request: **~1466 ms** (p50).
- Marginal cost per additional question on the same state: **~70 ms** (state prefill is shared via prefix caching on vLLM; HFScorer re-processes but still saves tokenization time).

Full load numbers in [`evals/reports/loadtest.md`](evals/reports/loadtest.md).

**No published SLA**. The box is manually brought up each dev morning and brought down each evening. During off-hours, the public endpoint returns 504 from the TLS terminator (backend not reachable). This is a development-tier affordance, not a production service.

---

## Worked example

### Request

```bash
curl -X POST https://bernoulli-api.nuwire.dev/v1/decide \
  -H "Authorization: Bearer $BERNOULLI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"text": "Where is my package? I ordered it two weeks ago and nothing has arrived. This is unacceptable."},
    "questions": [
      {"id": "intent", "type": "choice", "prompt": "What does the customer want?",
       "options": ["refund", "tracking", "exchange", "other"]},
      {"id": "urgent", "type": "binary", "prompt": "Is this urgent?"},
      {"id": "anger", "type": "rating", "prompt": "How angry is the customer? 1 = calm, 5 = furious.", "scale": [1, 5]}
    ],
    "options": {"debias": "reverse", "calibrated": true}
  }'
```

### Response (annotated)

```json
{
  "decisions": {
    "intent": {
      "type": "choice",
      "answer": "tracking",
      "confidence": 0.942,
      "distribution": {
        "tracking": 0.942,
        "other":    0.041,
        "refund":   0.015,
        "exchange": 0.002
      }
    },
    "urgent": {
      "type": "binary",
      "answer": true,
      "probability": 0.947        // probability that answer is true
    },
    "anger": {
      "type": "rating",
      "expected": 4.30,            // posterior mean of the rating distribution
      "distribution": {
        "1": 0.003,
        "2": 0.034,
        "3": 0.091,
        "4": 0.409,
        "5": 0.463
      }
    }
  },
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "calibration_version": null,
  "latency_ms": 124
}
```

Branch on these in your code:

```python
d = response["decisions"]

if d["intent"]["confidence"] < 0.5:
    route_to_human(ticket)                    # model doesn't know
elif d["intent"]["answer"] == "refund":
    route_to_billing(ticket)
elif d["urgent"]["probability"] > 0.9:
    priority_queue.add(ticket, score=d["anger"]["expected"])
```

The point: your code branches on **confidence** and **expected value**, not just the argmax `answer`. That's what the native-logit path gives you that a prompt-and-parse LLM does not.

---

## Client snippets

### Python (requests)

```python
import requests, os

BASE = "https://bernoulli-api.nuwire.dev"
KEY  = os.environ["BERNOULLI_API_KEY"]

def decide(state_text, questions, debias="reverse", calibrated=True):
    r = requests.post(
        f"{BASE}/v1/decide",
        headers={"Authorization": f"Bearer {KEY}"},
        json={
            "state": {"text": state_text},
            "questions": questions,
            "options": {"debias": debias, "calibrated": calibrated},
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()
```

### TypeScript (fetch)

```typescript
const BASE = "https://bernoulli-api.nuwire.dev";

async function decide(
  stateText: string,
  questions: Array<Record<string, unknown>>,
  apiKey: string,
) {
  const r = await fetch(`${BASE}/v1/decide`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      state: { text: stateText },
      questions,
      options: { debias: "reverse", calibrated: true },
    }),
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}
```

### Salesforce Apex

```apex
public class BernoulliClient {
    private static final String BASE = 'https://bernoulli-api.nuwire.dev';

    public static Map<String, Object> decide(String stateText, List<Map<String, Object>> questions) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint(BASE + '/v1/decide');
        req.setMethod('POST');
        req.setHeader('Authorization', 'Bearer ' + System.Label.BERNOULLI_API_KEY);
        req.setHeader('Content-Type', 'application/json');
        req.setBody(JSON.serialize(new Map<String, Object>{
            'state'     => new Map<String, Object>{ 'text' => stateText },
            'questions' => questions,
            'options'   => new Map<String, Object>{ 'debias' => 'reverse', 'calibrated' => true }
        }));
        req.setTimeout(30000);

        HttpResponse res = new Http().send(req);
        if (res.getStatusCode() != 200) {
            throw new CalloutException('Bernoulli ' + res.getStatusCode() + ': ' + res.getBody());
        }
        return (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
    }
}
```

**Salesforce setup prerequisites:**
1. **Remote Site Settings**: add `https://bernoulli-api.nuwire.dev` (and the staging URL if used).
2. **Named Credential** (recommended): store the API key as a Named Credential's authentication header; reference via `callout:BernoulliAPI` instead of hard-coded `System.Label`.

---

## Differences from Jev's API

Bernoulli's API is **not drop-in compatible with Jev's.** We implement the same technique (native-logit scoring over typed questions) but our wire format is independently designed. Specifically:

| Dimension | Bernoulli | Jev (approximate, see Jev docs for current) |
|---|---|---|
| Primary endpoint | `POST /v1/decide` | different path |
| Request top-level | `{state, questions, options}` | different field names |
| Question types | `choice` / `binary` / `rating` | `choice` / `noul` / `score` (per the JevBench harness vocab) |
| Response | discriminated-union `Decision` keyed by caller `id` | different schema |
| Auth | `Authorization: Bearer <key>` | may differ |
| Schema strictness | `extra="forbid"` → 422 on unknown fields | may differ |

**If you need Jev-shaped wire compatibility** (e.g., for a client written against Jev's SDK), that's a tracked future feature — see M14 in [`product/milestones.md`](product/milestones.md). The adapter would be ~0.5-1 day of work. Not current.

Design rationale: our schema is pydantic-strict and uses discriminated unions so clients get clean 422 errors on malformed input, rather than silent coercion or ambiguous success. Jev's schema wasn't designed around that strictness. Choosing our own was a deliberate call to prioritize machine-friendly correctness over mimicry.

---

## Versioning policy

- Base path `/v1/` is URL-pinned. Breaking changes get a new `/v2/`.
- **Breaking changes** = removing a field, renaming a field, changing a field's type, changing the semantics of a value.
- **Non-breaking changes** = adding a new field to a response (clients should ignore unknown fields in responses, even though the server rejects unknown fields in requests), adding a new question type, adding a new endpoint, adding a new error code.
- **Deprecation**: an endpoint or field marked for removal continues working for at least 90 days with a `Deprecation` response header pointing at the replacement. (We have not deprecated anything yet.)

The pydantic schemas in `bernoulli/types.py` are the authoritative source of the wire format. If this doc and that file disagree, the file wins — open an issue so this doc is corrected.
