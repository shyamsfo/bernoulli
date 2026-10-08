# How Bernoulli Works

**Audience**: experienced Python developer familiar with HuggingFace Transformers, LLM tooling (LiteLLM, LangChain, Instructor, BAML), and the usual LLM-serving concerns (tokenization, batching, bf16 vs fp16, KV cache). You should be able to read `bernoulli/*.py` after this doc and have the whole architecture make sense.

If you want API shapes (not architecture), read [`HTTP.md`](HTTP.md) + [`CLI.md`](CLI.md).
If you want to just run it, read [`USAGE.md`](USAGE.md).

---

## 1 — The one-paragraph version

Bernoulli is **a thin orchestration layer around a single HuggingFace model that turns typed questions into calibrated probability distributions**. For every question it runs **exactly one forward pass**, reads the logits at the "answer" position (where the model would start generating), restricts them to a known set of label tokens (A..Z for multi-way choice, Yes/No for binary, A..I for rating scales), softmaxes those into probabilities, and returns them. No text is generated. No JSON is parsed. No retries, no repair. The "LLM part" of the system is one `model(**inputs)` call per forward pass; everything else is prompt construction, token-ID bookkeeping, debiasing, and calibration — all pure NumPy and Python.

If that paragraph already makes sense: skim §2 for terminology, then jump to §4 for the end-to-end walkthrough.

---

## 2 — The core idea in a bit more detail

An LLM's forward pass produces, at every token position, a `(vocab_size,)` vector of logits — unnormalized log-probabilities over the vocabulary of "what the next token should be." Normal LLM use selects a token (via sampling, argmax, beam, etc.) and continues. **Bernoulli stops right there.**

Given a prompt that ends with `Answer:` (we deliberately end the assistant turn there — see §3), the logits at the final position tell you *for every token in the vocabulary, how likely the model thinks it is to appear next*. Most of those tokens are irrelevant — "cat", "the", "<|endoftext|>". But a tiny subset — the single tokens corresponding to our answer labels (` A`, ` B`, ` C`, ` Yes`, ` No`, etc.) — are what the model is actually "choosing between."

So the whole technique is:

1. Build a prompt whose last token sets up the model to emit a label next.
2. One forward pass. Grab the logits at that position.
3. Index into the logit vector at the known token-IDs of our labels. That gives a tiny `(num_labels,)` vector.
4. Softmax it. That's your probability distribution over the labels.

This is **native probabilities** — the probability of each option is read directly from the model's own internal probability space, not inferred from text the model generated. By contrast, pipelines like LiteLLM-with-JSON-schema or Instructor get probabilities by **asking the model to write them out**, then parsing the text — JevBench explicitly flags those as `probs_source="verbalized"` because they're a self-report by the model, not a measurement of it.

Three concrete consequences of native probs:

- **Dramatically better calibration.** When the model is uncertain between two options, the softmax reflects that: 0.52 vs 0.48. Text-generated probs tend to collapse into point masses (model writes `{"A": 1.0}` for its argmax even when it's barely leaning that way). On our benchmarks, NLL is **7-40× better** than the same backbone used as a text generator on the same inputs.
- **One forward pass per decision.** No autoregressive generation loop. On a 7B model this is ~90 ms vs seconds for a generated-and-parsed answer.
- **No parse failures.** The labels are known tokens; there's nothing to parse. "The model refused to answer in valid JSON" is not a failure mode that can happen.

### Why does this work at all?

The forward pass of a decoder-only LLM computes `P(next_token | context)` for every token. The model has been trained to put mass on whatever token is appropriate next given the preceding text. If the preceding text is a question + its options + `Answer:`, the appropriate next token *is* one of the options. The model's probability distribution over that restricted set is the actual decision we want.

The caveat is that we need the labels to **each be a single token** on the backbone's tokenizer. "A", "B", "C" are usually single tokens on modern tokenizers (BPE with a space prefix: ` A`, ` B`, ` C`). "Yes" and "No" also are. **Digits 1-9 with a space prefix are two tokens on Qwen2.5's tokenizer** (" " + "1"), which is why we use letters A-I for rating scales instead of digits (see `bernoulli/labels.py`'s module docstring).

This single-token requirement is the thing that constrains which models work out of the box. More on that in §6.

---

## 3 — Everything else is bookkeeping

The forward pass is simple. Everything *around* the forward pass is where the actual care lives:

### Prompt construction

Chat template, system prompt, state-first/question-last layout, assistant prefill with "Answer:". `bernoulli/prompt.py`.

Why state-first/question-last? When the same state (a document, a ticket, a review) is used for multiple questions, the state's tokens form a shared prefix. **vLLM's prefix caching reuses the KV cache for that prefix**, so questions after the first one on the same state are much cheaper. (This matters a lot in the ~145 ms per-question figure on the landing page — it assumes 20 questions over one state. Each marginal question is ~70 ms after the first.)

Why `"Answer:"` as the assistant prefill? We want the next token the model emits to be a *label*, not punctuation or a space. Prefilling with `Answer:` makes the model's "I'm starting to answer" tokens happen *before* the position we read logits at, so position `-1` is where it actually commits to a label.

### Label-token resolution

Every supported backbone must encode our labels as single tokens. `bernoulli/labels.py` holds the resolution logic + the `assert_single_token_labels()` guard that fires at scorer construction. If you point Bernoulli at a tokenizer that breaks "A" into two tokens, you'll get a `ValueError` with a clear message naming the offending label.

### Scorer

The actual forward pass. `bernoulli/scorer.py` defines a `Scorer` Protocol:

```python
class Scorer(Protocol):
    model_id: str
    revision: str | None
    def score(self, prompt: str, allowed_token_ids: list[int], *, images=None) -> NDArray[np.float32]: ...
    def score_batch(self, prompts: list[str], allowed_token_ids_list: list[list[int]], ...) -> list[NDArray[np.float32]]: ...
```

Two implementations ship:
- **`HFScorer`** — straight transformers, one `model(**inputs)` per `score()`, `torch.no_grad()`, logit slicing + numpy conversion. The dev path.
- **`VLLMScorer`** — vLLM engine with continuous batching + prefix caching + paged attention. The prod path. Signatures are identical; batching is where vLLM wins.

The `Scorer` protocol is intentionally dumb. It doesn't know about debiasing, calibration, or question types. It just takes a prompt + a list of allowed token IDs and returns the logits at the last position, sliced to the IDs. All higher-level logic composes on top.

### Debiasing

Models have **position bias** — all else equal, they slightly prefer whichever label appears first. Mitigation: run the same request with options in different orders, map back to canonical positions, average. `bernoulli/debias.py`. Default is `reverse` (two forward passes: canonical order + reversed order). `cyclic` runs K passes for K options, up to K=6.

Important footgun: **debias is auto-no-op for `BinaryQuestion`**. The Yes/No labels are *semantic*, not positional — reversing them doesn't swap positions, it inverts the meaning. On PAWS we had a 0.86-accuracy classifier flip to 0.37-accuracy before we caught this. See `product/learnings/paws-below-random.md` for the full post-mortem.

### Calibration

Temperature scaling. One scalar `T` per question type (choice/binary/rating), fit on NLL over a held-out set with SciPy's bounded `minimize_scalar`. `bernoulli/calibrate.py`. At serve-time we re-softmax `log(probs) / T` — mathematically identical to fitting on the pre-softmax logits since any constant offset cancels under softmax.

Calibration JSON lives at `BERNOULLI_CALIBRATION_PATH`; unset = no calibration. Request-level override via `request.options.calibrated`.

### Decision assembly

Dispatcher: `bernoulli/decide.py` (`_decide_one`). Takes the per-option probability array, maps back to the original option strings, builds the type-specific `Decision` (ChoiceDecision has `.distribution`, BinaryDecision has `.probability` + `.answer`, RatingDecision has `.distribution` + `.expected`).

### The stack at a glance

Our ~1,500 lines sit between an HTTP/CLI entry and PyTorch. transformers (or vLLM) is the one layer we lean on from outside — it packages the model and tokenizer so we don't carry architecture code ourselves.

```
┌───────────────────────────────────────────────────────────────────┐
│ Entry points                                                      │
│   bernoulli/server.py       FastAPI · /v1/decide, /v1/generate    │
│   bernoulli/cli.py          `bernoulli decide` on stdin           │
├───────────────────────────────────────────────────────────────────┤
│ Orchestration                                                     │
│   bernoulli/decide.py       request → response threading          │
├───────────────────────────────────────────────────────────────────┤
│ Primitives (all pure numpy / Python)                              │
│   bernoulli/prompt.py       chat template + label rendering       │
│   bernoulli/labels.py       single-token label resolution         │
│   bernoulli/debias.py       permutation averaging                 │
│   bernoulli/calibrate.py    temperature scaling                   │
│   bernoulli/chunked.py      >26-option choice path                │
├───────────────────────────────────────────────────────────────────┤
│ Scorer protocol  ← swappable seam, backbone-agnostic              │
│   bernoulli/scorer.py       Scorer protocol + load_scorer factory │
├─────────────────────────────────┬─────────────────────────────────┤
│ HFScorer (dev / test)           │ VLLMScorer (prod / throughput)  │
│   bernoulli/scorer.py           │   bernoulli/vllm_scorer.py      │
│                                 │                                 │
│   • AutoTokenizer               │   • reimplements forward pass   │
│   • AutoConfig                  │     with continuous batching +  │
│   • AutoModelForImageTextToText │     paged attention +           │
│     (VL) or AutoModelForCausalLM│     prefix caching              │
│     (text) — see _load_backbone │   • loads weights directly;     │
│   • apply_chat_template         │     doesn't go through          │
│   • one model(**inputs) per     │     transformers.AutoModel      │
│     scorer.score() call         │                                 │
├─────────────────────────────────┴─────────────────────────────────┤
│ transformers (used only in the HFScorer branch above)             │
│   model packaging + tokenizers + chat templates — four calls      │
│   total; see §5 for the full enumeration                          │
├───────────────────────────────────────────────────────────────────┤
│ PyTorch — the actual math                                         │
│   torch.nn.Module forward pass (where the decision happens)       │
│   torch.no_grad() · bfloat16 · device_map='cuda' · tensor slicing │
├───────────────────────────────────────────────────────────────────┤
│ CUDA · cuDNN · cuBLAS · (optional) Triton kernels on the vLLM path│
├───────────────────────────────────────────────────────────────────┤
│ NVIDIA GPU                                                        │
│   dev    · A10G 24 GB on AWS g5.xlarge   (bernoulli AWS)          │
│   prod   · L40S 48 GB target for 32B-AWQ (parked on capacity)     │
│   demo   · A10G 24 GB on HF Spaces       (shyamsfo/bernoulli-demo)│
└───────────────────────────────────────────────────────────────────┘
```

**How to read it**: a request enters at the top and flows down. The call stack through our code (`server.py → decide.py → debias.py → scorer.py`) stops at the Scorer protocol — the first thing *below* it is either transformers+torch (HFScorer) or vllm+torch (VLLMScorer). The actual decision — the probability for every option — is produced inside PyTorch, one layer above CUDA. Everything else is structure around that one forward pass.

**Rule of thumb for what's where**:

- *Shaping the request*: top four layers (entry → orchestration → primitives → scorer protocol). Pure Python + numpy.
- *Executing the forward pass*: transformers + PyTorch (HF path) *or* vLLM + PyTorch (prod path). Minutes-to-write-ourselves work that we delegate.
- *Doing the math*: PyTorch, dispatching to CUDA kernels. Not something we'd write.

---

## 4 — End-to-end walkthrough: one `/v1/decide` request

Let's follow a concrete choice-question request through the stack. The ticket:

```json
POST /v1/decide
{
  "state": {"text": "Where is my package? I ordered it two weeks ago..."},
  "questions": [
    {"type": "choice", "id": "intent", "prompt": "What does the customer want?",
     "options": ["tracking", "refund", "cancel"]}
  ],
  "options": {"debias": "reverse", "calibrated": false}
}
```

### 4.1 — HTTP → `decide()`

`bernoulli/server.py` is a thin FastAPI wrapper. On startup, `lifespan()` loads the scorer (`HFScorer` or `VLLMScorer` based on `BERNOULLI_SCORER`) and optional calibration JSON, stashes both on `app.state`. The `/v1/decide` endpoint is three lines:

```python
@app.post("/v1/decide", response_model=DecideResponse)
def decide_endpoint(request: DecideRequest) -> DecideResponse:
    return decide(request, app.state.scorer, calibration=app.state.calibration)
```

Pydantic parses the request body into a `DecideRequest` with a discriminated-union `Question` type (ChoiceQuestion / BinaryQuestion / RatingQuestion). The discriminator is the `type` field.

### 4.2 — `decide()` → `_decide_one()`

`bernoulli/decide.py:decide()` loops over `request.questions`, calling `_decide_one()` per question. For our example (one choice question with 3 options, below the 26-option chunking threshold), it goes to the standard path:

```python
probs = debias(scorer, state_text=state_text, question=question, mode="reverse")
```

### 4.3 — `debias()` builds prompts for both permutations

For `mode="reverse"` on a 3-option question, `_permutations_for()` returns:

```python
[[0, 1, 2], [2, 1, 0]]
```

For each permutation, `build_prompt()` renders the full prompt. The canonical permutation [0, 1, 2] produces something like (after `tokenizer.apply_chat_template`):

```
<|im_start|>system
You are a decision function. Answer with a single letter.<|im_end|>
<|im_start|>user
Where is my package? I ordered it two weeks ago...

What does the customer want?
A) tracking
B) refund
C) cancel<|im_end|>
<|im_start|>assistant
Answer:
```

(Exact tokens depend on the backbone's chat template; this is Qwen2.5's rough shape.)

The reversed permutation [2, 1, 0] produces the same thing but with `A) cancel / B) refund / C) tracking` — same canonical options, mapped to label positions in reverse.

### 4.4 — Label-token resolution

`_allowed_ids()` resolves which token IDs to read logits at:

```python
labels = labels_for(question)                       # ["A", "B", "C"]
mapping = letter_token_ids(tokenizer)               # {"A": 362, "B": 425, ...}  (Qwen2.5 example)
return [mapping["A"], mapping["B"], mapping["C"]]   # [362, 425, 507]
```

This is a one-time lookup — token IDs don't change between calls, we could cache them on the scorer, but encoding one letter per call is cheap enough that we don't bother.

### 4.5 — Scorer forward passes

`debias` calls `scorer.score_batch(prompts, [allowed] * 2)` with both prompts. On `HFScorer` this is a loop over `score()`:

```python
# bernoulli/scorer.py:HFScorer.score
inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
with torch.no_grad():
    outputs = self._model(**inputs)
last_logits = outputs.logits[0, -1, :]       # (vocab_size,)
selected = last_logits[allowed_token_ids].float().cpu().numpy()
return np.asarray(selected, dtype=np.float32)
```

Two forward passes = ~180 ms on A10G 24GB with Qwen2.5-VL-7B bf16. On `VLLMScorer` the two prompts go in one engine call with continuous batching + prefix caching, and the state tokens are shared across both (prefix cache hit on the second).

### 4.6 — Softmax + permutation unwinding

`debias()` softmaxes each permutation's logits into probabilities and then maps them back to canonical positions:

```python
# For permutation [0,1,2]: probs[0] is P(tracking), probs[1] is P(refund), probs[2] is P(cancel)
# For permutation [2,1,0]: probs[0] is P(cancel), probs[1] is P(refund), probs[2] is P(tracking)
# After mapping both back to canonical, we average.
```

Output: a `(3,)` float32 array indexed by canonical option order.

### 4.7 — Calibration (skipped here)

Request had `calibrated: false`, so this step is a no-op. If it were true and a `Calibration` were loaded, we'd apply `T` scaling via `apply_temperature()` — divide pseudo-logits (log of probs) by `T` and re-softmax. One `T` per question type.

### 4.8 — Decision assembly

`_decide_one()` builds the type-specific decision:

```python
best_opt = max(probs_dict, key=lambda k: probs_dict[k])
return ChoiceDecision(answer=best_opt, confidence=probs_dict[best_opt], distribution=probs_dict)
```

### 4.9 — Response

`decide()` wraps all decisions into a `DecideResponse`:

```json
{
  "decisions": {
    "intent": {
      "type": "choice",
      "answer": "tracking",
      "confidence": 0.942,
      "distribution": {"tracking": 0.942, "refund": 0.047, "cancel": 0.011}
    }
  },
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "calibration_version": null,
  "latency_ms": 183
}
```

Serialized by Pydantic. Sent back over HTTP.

**Total: 2 forward passes, 1 softmax, 1 permutation unwind, 1 Decision assembly. The whole request path is <200 lines of code.**

---

## 5 — How we're using Qwen2.5-VL-7B specifically

The dev backbone is `Qwen/Qwen2.5-VL-7B-Instruct` pinned at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. Pulled with `huggingface_hub` + `hf_transfer` (fast parallel downloader) to `/opt/dlami/nvme/hf-cache` on the dev box.

What we use from it:

1. **The tokenizer** — `AutoTokenizer.from_pretrained(model_id, revision=rev)`. Qwen's tokenizer is a BPE variant. Space-prefixed letters A-Z are single tokens; space-prefixed digits are two tokens; "Yes" and "No" are single tokens.

2. **The model class** — dispatched automatically via `_load_backbone()` in `scorer.py`. For Qwen2.5-VL this resolves to `AutoModelForImageTextToText` (which selects `Qwen2_5_VLForConditionalGeneration` under the hood). Swap backbones via env vars; see §6.

3. **The chat template** — baked into the tokenizer. `apply_chat_template(messages, add_generation_prompt=True)` emits the ChatML format with the correct `<|im_start|>`, `<|im_end|>` turn boundaries.

4. **Loading in bf16** — `torch_dtype=torch.bfloat16`, `device_map="cuda"`. On A10G 24GB this fits the 7B weights (~15 GB) with room for the KV cache. For vLLM you also need `max_model_len=8192` on 24GB cards; HFScorer doesn't allocate the KV cache up front, so it doesn't care.

5. **`eval()` mode** — dropout off, deterministic forward pass. We never gradient on the model.

6. **`torch.no_grad()` around the forward pass** — essential; without it torch would build the autograd graph for a 7B model's forward and OOM instantly.

### What we don't use

- **No text generation** on the decide path. HFScorer *also* has a `.generate()` method (greedy, `max_new_tokens=10`) but it's for the generative-baseline comparison in `/v1/generate`, not for decisions. VLLMScorer has no `.generate()` at all.
- **No tool use, no function calling, no structured output modes.** The model's text-generation surface is irrelevant to us — we read logits before any token is emitted.
- **No image path yet.** Qwen2.5-VL is a VLM but we only use its text path for M2-M5. `AutoModelForImageTextToText` happily loads it as a text-only module since we never pass image inputs to `forward()`. The `AutoProcessor` + `torchvision` deps that actual image handling needs are deferred to M6; `HFScorer.score()` raises `NotImplementedError` on any request with images attached.

---

## 6 — Working with a different model

This section is the one you'll return to when you want to swap backbones.

### The dispatch flowchart

```
Want to swap to a different Qwen2.5-VL variant (3B / 32B-AWQ)?
  → BERNOULLI_MODEL_ID + BERNOULLI_MODEL_REVISION env vars. No code change.

Want to swap to a different VL family (Qwen3-VL, Llama-3.2-Vision, Gemma-3-Vision, Pixtral, LLaVA)?
  → Same env-var-only change. _load_backbone() dispatches to AutoModelForImageTextToText.

Want to swap to a text-only decoder-only LM (Llama 3, Gemma 2, Mistral, Qwen2.5-text)?
  → Same env-var-only change. _load_backbone() dispatches to AutoModelForCausalLM.

In all four cases, the only thing to verify is:
  (a) The tokenizer encodes letters A-Z and Yes/No as single tokens — checked
      at scorer construction by assert_single_token_labels(), fails loud if not.
  (b) The model is registered under one of the two Auto classes above —
      nearly every modern HF-hosted LM is.

Want to use vLLM instead of transformers?
  → BERNOULLI_SCORER=vllm env var. No code change. Note: VLLMScorer requires a model vLLM supports.
```

### How dispatch works

`bernoulli/scorer.py:_load_backbone()` reads `AutoConfig(model_id)` and routes:

```python
config = AutoConfig.from_pretrained(model_id, revision=revision)
model_cls = AutoModelForImageTextToText if _is_vl_config(config) else AutoModelForCausalLM
return model_cls.from_pretrained(model_id, revision=revision, torch_dtype=..., device_map=...)
```

`_is_vl_config(config)` is a one-liner: "does the config have a non-None `vision_config` sub-config?" That signal holds for every VLM in the HF ecosystem (Qwen2.5-VL, Qwen3-VL, Llama-3.2-Vision, Gemma-3-Vision, Pixtral, InternVL, LLaVA, ...) without needing a hard-coded list of model_type strings — if HF adds a new VL family tomorrow, our dispatch picks it up for free.

Both `AutoModelForImageTextToText` and `AutoModelForCausalLM` give us a model whose `forward()` returns a `.logits` tensor shaped `(batch, seq, vocab_size)` — the only interface Bernoulli actually calls.

**Historical note**: an earlier version of this file (and the project) hard-coded `Qwen2_5_VLForConditionalGeneration` because there was only one backbone in play and the generalization wasn't worth building. The config-driven dispatch landed 2026-10-08.

### Tokenizer gotchas to check on any new backbone

`assert_single_token_labels(tokenizer)` is called at scorer construction and will loudly refuse to load if any label isn't a single token. Specifically it checks:

- All 26 space-prefixed letters: ` A`, ` B`, ..., ` Z`
- Space-prefixed `Yes`, `No`

If any of those encode to 2+ tokens, scorer construction fails with a clear error message. **If you hit this on a new backbone, the fix is either (a) switch to a different label alphabet that *is* single-token on that tokenizer, or (b) use a different backbone.**

Known cases:
- Qwen2.5-VL and family: ✅ all 26 letters + Yes/No are single-token
- Llama 3 family: ✅ (verified on Llama-3.1-8B)
- Gemma 2 family: ✅ (verified on Gemma-2-9B)
- Mistral family: ⚠️ some mistral tokenizers split certain capital letters — verify first

### Chat template handling

Any HF tokenizer with a `chat_template` set works out of the box with `tokenizer.apply_chat_template()`. Our prompt builder (`bernoulli/prompt.py`) calls that directly — no backbone-specific prompt code anywhere. If a tokenizer doesn't ship a chat template, `apply_chat_template` raises; the fix is to set one explicitly before loading into HFScorer.

### Dtype considerations

`BERNOULLI_DTYPE` defaults to `bfloat16`. For A10G/L4/L40S 24GB+ cards, bf16 is the sweet spot. For older GPUs without bf16 (T4, V100), use `float16`. For CPU-only testing (not production — this is slow), use `float32`.

### Max model length

Only matters for `VLLMScorer`. `BERNOULLI_MAX_MODEL_LEN=8192` on 24GB cards; vLLM pre-allocates KV cache based on this at engine init. Default is 32768, which is too big on 24GB cards once the model weights are loaded. HFScorer doesn't pre-allocate, so it ignores this setting.

---

## 7 — How Bernoulli differs from LiteLLM / LangChain / Instructor

If you're coming from the LLM-tooling ecosystem, the mental shift is: **Bernoulli replaces the LLM call primitive, not the orchestration around it.**

### vs LiteLLM

LiteLLM is a proxy layer in front of chat-completion APIs. You call `litellm.completion(messages=..., response_format=...)`, it dispatches to the underlying provider, you get back text (optionally JSON-structured) and token logprobs (optionally, for providers that expose them).

- **LiteLLM's logprobs are over the generated tokens**, not over an arbitrary restricted label set. If the model's structured output is `{"category": "sports"}`, you get logprobs for the tokens of `sports` — but if you wanted logprobs for `{"category": "world"}` you'd have to actually ask it to say that.
- **Bernoulli reads logits over any label set you specify**, in one forward pass. Logprobs for A/B/C are all available simultaneously, not sequentially.
- **No HTTP tariff.** LiteLLM is API-gateway thinking; Bernoulli is model-serving thinking.
- **Bernoulli is specifically scoped to typed decisions.** LiteLLM supports chat, tool use, streaming, embeddings — a huge surface. Bernoulli does three question types (choice / binary / rating) and that's it. Everything LiteLLM does falls back to generation-and-parse under the hood.

**When to use each**: LiteLLM for anything that needs chat or tool use. Bernoulli for anything that's structurally "pick one of N options with a probability."

### vs LangChain

LangChain is an orchestration framework: prompts, chains, tools, retrievers, memory, callbacks. The "LLM" is one node in a graph.

- **Bernoulli doesn't do orchestration.** It gives you a well-calibrated decision primitive. You'd put a Bernoulli call *inside* a LangChain graph as one of the nodes — probably in place of a "classifier" or "router" node that currently uses a prompt + schema + LLM call.
- **LangChain's output parsers** (`PydanticOutputParser`, `StructuredOutputParser`) are the thing Bernoulli replaces. They work by generating text and parsing it. Bernoulli skips the generation and reads native probabilities.

**When to use each**: LangChain for multi-step workflows with retrieval, tool calls, memory. Swap specific nodes in it for Bernoulli calls when the node is "classify / rate / branch."

### vs Instructor / BAML / Outlines / xgrammar

These are the "constrained-generation" category. They force the LLM to emit a specific schema (JSON, Pydantic, regex, grammar) by masking logits during generation.

- **Shared intuition**: both constrained-generation and Bernoulli use the same observation (logits over a restricted token set carry calibrated signal). The implementation difference is dramatic.
- **Constrained generation still generates text**, token-by-token, applying masks at each step. A 10-token JSON object = 10 forward passes.
- **Bernoulli reads one position, one forward pass**, over the ultimate answer tokens directly. No sequential generation.
- **Constrained generation outputs point-mass JSON by default**. Even if the mask includes probabilities per token, assembling a well-calibrated per-option probability from a multi-token structured output is nontrivial. (You'd have to multiply token logprobs along the generated path for each possible path — doable, but you're reimplementing Bernoulli through a very roundabout mechanism.)
- **Bernoulli has a hard constraint on label alphabet size** (26 single-token letters). Constrained generation can handle arbitrary vocabularies at the cost of more generation steps.

**When to use each**: Instructor/BAML for arbitrary structured output (nested JSON, long text fields with constraints). Bernoulli for low-cardinality typed decisions where calibration matters.

### vs HuggingFace Transformers directly

You could write Bernoulli's hot path yourself in ~30 lines of transformers code. We did. The library packages:

- **The debiasing scheme** (reverse / cyclic permutation averaging + the BinaryQuestion exception)
- **The temperature-scaling calibrator**
- **The scorer protocol with vLLM swap-in**
- **The chunked-choice path for >26 options** (not covered in this doc; see `bernoulli/chunked.py`)
- **Benchmarks + reproducibility harness** across 10 datasets + JevBench
- **An HTTP + CLI surface** matching an API spec, with Pydantic validation and OpenAPI docs

If your use case is "I want one decision from one prompt and I'm fine without debiasing or calibration," you can indeed just write the 30 lines. The point of Bernoulli is that calibrated, debiased, benchmarked decisions are a different-shaped product.

---

## 8 — Non-obvious things that bit us during development

Running list of things that aren't obvious from the architecture, but have shown up as bugs or performance issues. Documented here so you don't rediscover them:

1. **Digits aren't single tokens on Qwen.** Vision doc §4 originally called for digit labels for ratings; the first test revealed `tokenizer.encode(" 1")` returns `[220, 16]` (two tokens: space + digit). We use letters A-I for rating scales instead. Any new tokenizer family needs this re-verified.

2. **Reverse debias flips BinaryQuestion.** PAWS below-random bug (0.37 accuracy where it should have been 0.83). Short-circuited debias to no-op for binary. Full post-mortem in `product/learnings/paws-below-random.md`.

3. **`eval()` and `torch.no_grad()` are both required.** `eval()` alone still builds autograd for the forward; `no_grad()` alone leaves dropout on in some model configs. Both.

4. **vLLM's `max_model_len` must drop to 8192 on 24GB cards.** The default 32768 pre-allocates ~17 GB of KV cache, which plus 15 GB of 7B weights OOMs the 24 GB card. HFScorer has no equivalent issue since it allocates KV cache on-demand per request.

5. **vLLM EngineCore can zombie.** If vLLM init crashes partway (bad config), the engine subprocess can survive the Python exit holding ~18 GB on the GPU. `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r sudo kill -9`.

6. **Transformers 5.x imports torchaudio at import time.** If CUDA versions mismatch across torch/vision/audio, you get a cryptic error from inside torchaudio when importing `transformers`. Bump all three together when bumping the CUDA pin.

7. **"Last position" is not always `-1`.** After `apply_chat_template(add_generation_prompt=True)` the tokenizer adds a few trailing tokens for the assistant turn (ChatML has `<|im_start|>assistant\n`); we then add `Answer:` as a prefill. The last position after all of this is where logits are read. If a tokenizer doesn't add any trailing tokens, our logit index is still `-1` and that happens to work. If one did add weird trailing structure, we'd need to be smarter about picking the index. So far no tokenizer we've tested has this problem.

8. **JevBench score-task legend numbering must match the rendered option digits.** The adapter mapped 0-indexed level codes against our 1-indexed internal rating scale, which made the model pick the letter one below correct. Caught by per-family analysis of the first JevBench run; one-line fix. See the commit history on `benchmarks/jevbench/adapter.py` for the full story. This isn't a Bernoulli-core bug — it's an adapter mismatch — but it's a cautionary tale about any wrapper that translates between external label schemes and internal ones.

---

## 9 — If you want to extend this

Starter prompts for common extensions:

- **"I want to swap in Llama 3.1 8B."** `BERNOULLI_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct` + `BERNOULLI_MODEL_REVISION=<sha>`. That's it — `_load_backbone()` dispatches to `AutoModelForCausalLM` automatically. Scorer construction will fail if any label isn't single-token; verify by running the test suite (`just test-fast` covers this).

- **"I want to add a fourth question type (e.g., ranked list)."** Add a new `RankedQuestion` to `bernoulli/types.py` (will auto-register in the discriminated union). Add a prompt builder entry in `bernoulli/prompt.py` (`labels_for` + `option_strings_for`). Add decision assembly to `_decide_one` in `bernoulli/decide.py`. Add a result type in `bernoulli/types.py`. ~50 lines total if you follow the pattern of the existing three.

- **"I want to fine-tune the backbone on my labels."** This is M7 (parked). The idea is a small LoRA adapter on the backbone — same scorer path, same forward pass, better zero-shot accuracy. See `bernoulli/scorer.py` for where `PEFTModel.from_pretrained` would wrap the base model.

- **"I want a different debias scheme."** Add a new `DebiasMode` in `bernoulli/debias.py` + a case in `_permutations_for`. The averaging logic in the main `debias()` loop is permutation-agnostic.

- **"I want to replace the HTTP server with gRPC."** The scorer, debiaser, calibrator, and decision assembly are all pure Python with no HTTP coupling. `decide(request, scorer, calibration=...)` is a plain function call. Write a gRPC server that wraps it.

---

## 10 — Where everything lives

```
bernoulli/
├── types.py        # DecideRequest/Response, three question types, three decision types
├── config.py       # BERNOULLI_* env var parsing (pydantic-settings)
├── labels.py       # Label-token resolution + single-token assertion
├── prompt.py       # Chat template + prompt rendering
├── scorer.py       # Scorer protocol + HFScorer + factory
├── vllm_scorer.py  # VLLMScorer (lazy-imported)
├── debias.py       # Permutation averaging across label positions
├── chunked.py      # >26-option choice via chunked scoring + tournament
├── calibrate.py    # Temperature-scaling calibrator
├── decide.py       # The orchestrator: request → response
├── generative.py   # Baseline-only generative decode (for /v1/generate)
├── server.py       # FastAPI app, /v1/decide + /v1/generate + /healthz + /v1/models
└── cli.py          # `bernoulli decide` CLI
```

Total: ~1,500 lines of Python. The forward pass itself is in `scorer.py` and is ~20 lines; everything else is bookkeeping.

Tests live at `tests/`. Benchmarks (which are the real integration tests of the technique) live at `benchmarks/`.
