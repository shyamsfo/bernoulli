# Public demo options — "let friends and nerds try Bernoulli"

**Status:** research, not a decision. Feeds the first task of **M11 — public verification endpoint** ("decide the shape") when that milestone is pulled.

**Scope split from M11 proper.** M11 is the fully-hardened, Benchmark-Heaven-grade endpoint that earns a *ranked* leaderboard row — depends on M5 hardening and probably M4e production backbone. **This doc is about the scrappier version:** a link you can paste in a text message so a friend/nerd can type a question and see a native-probs response. The two overlap in infra and in some decisions (auth model, abuse controls) but not in SLO or audience.

## What "friends and nerds" actually need

The demo audience is small, informal, forgiving, and specifically wants to see **the technique**, not a production SaaS. A good demo for them:

1. **Shows the native-probs output** — a distribution, not a point mass. The whole thesis is on screen in one example.
2. **Supports all three question types** (choice / binary / rating). "Yes/No with a probability" is where it clicks; being able to flip to "A/B/C/D/E with five probabilities" is where the generality clicks.
3. **Is clickable from a phone** — no git clone, no venv, no HuggingFace login. The bar is: paste the link in a text message, friend taps it on their phone, something interesting happens in ≤60 s.
4. **Costs ~$0/mo when nobody's using it.** Nerds use demos in bursts; renting a GPU 24/7 for a demo audience of 5 people is absurd.
5. **Is okay with a cold-start** on first use. 30-60 s loading a 7B VLM from a cached checkpoint is acceptable for a demo. Friends and nerds will wait if a progress bar explains what's happening.

Non-goals:
- Production SLO (that's M11-ranked)
- Independent third-party verification (that's M11-ranked)
- Pay-per-use revenue
- Beating other hosted LLM demos on raw UX polish

## Options

### Option A — HuggingFace Space + Gradio, sleep-when-idle

**Shape.** A `gradio.Interface` or `gradio.Blocks` app running on HF's A10G Small tier (same GPU family as our dev box). Users land on `https://huggingface.co/spaces/<user>/bernoulli-demo`, pick a question type, type a prompt, see probabilities as a bar chart. HF Spaces has a native sleep-when-idle mode: the Space suspends after N minutes of no traffic; next visitor triggers a cold start.

**Cost.** $0/mo while asleep. HF A10G Small is $1.05/hr when active; a 60-second cold-start plus 30-minute active session for a single user is ~$0.55 per visit. For a demo audience of 5–20 people/week, that's $5–20/mo tops.

**Effort to stand up.** 4–8 hours end-to-end:
- Package the model + scorer into a Space-compatible repo (requirements.txt, app.py, pinned transformers/torch versions matching our Dockerfile).
- Port `HFScorer` + `decide()` logic into a Gradio handler. The internals don't change; only the request path (Gradio event handler, not FastAPI route).
- Build the UI: three tabs for the three question types, a text area for state, a dynamic form for options (choice) or scale (rating), a bar chart for the response distribution. Gradio's `gr.BarPlot` or just numeric labels.
- First-boot: upload or pre-cache the model to the Space's `/data` volume so cold-start isn't downloading 15 GB.

**Risks / downsides.**
- **Cold-start wait is 30–60 s** on first visitor. Need a loading state and a one-line explanation ("warming up the GPU — Bernoulli's technique needs a 7B model loaded; this takes ~45 s on first visit, then is fast until nobody uses it for 30 min").
- **HF Spaces hosts the backbone**, not us — so if HF has an outage, the demo is down. Fine for friends-and-nerds; not fine for a ranked JevBench row.
- **Abuse surface:** anyone can hit the Space. HF has some built-in rate limiting but it's not auth-grade. For a demo, a hard per-session request cap in `app.py` is probably enough.
- **Discoverability cuts both ways:** HF Spaces have discovery (`spaces.huggingface.co/<user>`), which is good for inbound but also means strangers will find and use it. Burn rate scales with traffic.

**Verdict for the "friends and nerds" bar.** This is probably the right call. It gives the exact "link in text message → clickable demo" experience, with ~$0 idle cost, 4–8 h of setup, and a UI that shows the native-probs story cleanly.

---

### Option B — HuggingFace Inference Endpoint

**Shape.** A dedicated managed HTTP endpoint on HF's infra, serving the model via a custom handler. No UI; it's an API.

**Cost.** $0.60/hr minimum (CPU) up to $4+/hr (large GPU) — **does not support sleep-when-idle for dedicated endpoints**. Scale-to-zero exists for serverless endpoints but has its own tradeoffs (cold-start seconds, no custom handler). Expect ~$430/mo for a 24/7 A10G endpoint.

**Effort to stand up.** 2–4 hours — HF Inference Endpoints accept a custom `handler.py` that gets a request dict and returns a response dict. The scorer path maps cleanly. But we also need to write the client-side demo page or a Postman link so nerds can actually hit it.

**Risks / downsides.**
- **No idle discount.** You're renting the GPU 24/7.
- **API-only** means no visual demo — nerds have to curl or load a separate page.
- **If you want a visual demo on top**, you're building Option A anyway and pointing it at this endpoint instead of HF's Space-GPU — doubling the moving parts.

**Verdict.** Only makes sense if you specifically want an *API* demo (for other developers to integrate against), not a *visual* demo. Expensive for a friends-and-nerds audience. **Skip** unless the audience shifts.

---

### Option C — Replicate (pay-per-request)

**Shape.** Package the model as a Replicate "model" via their `cog` tool (Docker container with a `predict.py`). Users hit a REST API; Replicate manages cold-start, autoscaling, and billing. Model stays warm as long as there's traffic; scales to zero when idle (free when idle, cold-start ~15–30 s on wake).

**Cost.** $0/mo idle. Pay-per-second of GPU time; A40 is ~$0.000725/s = ~$0.044/decision assuming a 60 s session with a few requests. For 20 nerds × 10 requests/mo, expect $5–20/mo.

**Effort to stand up.** 2–4 hours — `cog.yaml` + `predict.py` + a push to Replicate's registry. API-only though; same visual-demo gap as Option B unless we also build a page.

**Risks / downsides.**
- **No native UI.** Replicate has a "run" tab on their model page that renders basic form fields from the `predict.py` signature — enough for nerds but not visually compelling.
- **GPU lineup is different** from AWS — A40 instead of A10G/L4. Not a correctness issue (same inference) but means we'd have to re-verify numbers on the Replicate-specific instance class if they're ever reported publicly.
- **Lock-in to Replicate's cog format** for distribution; our Docker image already works on vLLM-standard infra and this is one more thing to maintain.

**Verdict.** Decent middle ground if we want pay-per-use. **Skip for the demo bar** because Replicate's default UI is uglier than Gradio's and the audience is small enough that we don't need their autoscaling.

---

### Option D — Cloudflare Tunnel over our existing AWS instance

**Shape.** Install `cloudflared` on the bernoulli g5.xlarge, point a `demo.bernoulli.live` DNS record at it, keep the existing FastAPI `/v1/decide` server as the backend. Add rate limiting and a simple HTML page served from the same box.

**Cost.** $30/mo on-demand ($0 for the Cloudflare Tunnel itself — free tier) + whatever we're already paying for the g5 if we leave it running. Spot pricing (~$0.30/hr = $220/mo if 24/7) is cheaper but risks preemption, which is user-visible as "the demo is down."

**Effort to stand up.** 2 hours — cloudflared install, DNS record, basic HTML form + rate limiter middleware in FastAPI. Shortest path by far.

**Risks / downsides.**
- **We own uptime.** Spot preemption is user-visible. On-demand is $30/mo more than spot. Not a terrible trade for a demo.
- **We own abuse response.** Rate limiting is on us; if someone finds the URL and hammers it, the GPU is held hostage until we add their IP to a block list.
- **Shares the GPU with dev work.** Running `just serve` for development co-mingles with public traffic. For a demo this is probably fine (we're not doing training on the same box); it would be a problem for the ranked M11 version.
- **No scale-to-zero** — same billing concern as Option B, modulo spot vs on-demand.

**Verdict.** Fastest to ship but worst on cost ratio. If we have a reason to keep the AWS instance running anyway (e.g. an active development arc with daily use), the marginal cost of exposing it is tiny. If AWS is otherwise idle most days, this is the most expensive option on the list.

---

### Option E — Colab notebook (no live demo)

**Shape.** A `bernoulli_demo.ipynb` committed to the repo with a "Open in Colab" badge. Users click the badge, Colab provisions a free T4 GPU, model loads in-notebook, user runs cells to try their own prompts.

**Cost.** $0/mo on both sides. User pays with their Colab free-tier GPU allowance.

**Effort to stand up.** 1 hour — a small notebook that installs deps, loads the backbone, imports our `HFScorer` + `decide`, and provides three example cells (choice / binary / rating).

**Risks / downsides.**
- **Setup bar.** User needs a Google account, has to click through Colab's runtime assignment, has to wait for pip install + model download (~5 min total). Only nerds who actually want to run the code will do this.
- **No link-in-text-message experience.** This is self-selecting the audience to the subset of nerds who already know Colab.
- **No visual story.** It's a notebook. For devs, that's a feature; for showing a friend the probability bar chart, it's not.

**Verdict.** Zero-cost backup option. Worth adding **regardless** of whether we do a live demo — makes the "I want to run this myself" path trivial and costs an hour of setup. Doesn't satisfy the clickable-link bar on its own.

---

## Comparison table

| Option | Effort | $/month idle | $/month typical | UX bar | Verdict |
|---|---|---:|---:|---|---|
| A — HF Space + Gradio, sleep | 4-8 h | $0 | $5-20 | Link-click, visual, cold-start wait | **Pick this** for the friends-and-nerds bar |
| B — HF Inference Endpoint | 2-4 h | ~$430 | ~$430 | API-only | Skip |
| C — Replicate | 2-4 h | $0 | $5-20 | API-only (bland default UI) | Skip for demo; revisit if we want pay-per-use for M11 |
| D — Cloudflare Tunnel over AWS | 2 h | ~$30 on-demand | same | Link-click, we own abuse | Only if AWS is already running for other reasons |
| E — Colab notebook | 1 h | $0 | $0 | Setup bar, nerds-only | **Add as a secondary** regardless |

## Recommended path

**A + E together.** HF Space with Gradio for the clickable demo friends can tap on their phone, Colab notebook for the nerds who want to see the code. Total effort: 5–9 hours; total ongoing cost: ~$5–20/mo depending on traffic.

Skip B, C, D for this bar — they're all tradeoffs that make sense for different audiences (B/C for API consumers, D for when AWS is already up).

## Where this feeds back into M11

Option A, if we build it, is **not** M11 — M11 requires an endpoint Benchmark Heaven's harness can hit without a human in the loop on our side, and the Space URL plus a stable auth model gets us close but still has HF's "wake me up" cold-start and HF-managed abuse controls. For M11-ranked we want:
- The same (or stronger) backbone as the dev box, in a stable always-on configuration
- Auth model that lets Benchmark Heaven call us (API key or shared secret)
- Abuse controls we fully own (per-key budget, per-IP rate limit)
- Uptime SLO we can commit to (29.9% is not enough; the harness retries but a sustained outage during a benchmark re-run disqualifies the row)

So Option A is a **spike** that validates the shape (does Gradio + our scorer compose nicely? Does HF's sleep model behave as advertised? Does cold-start wreck the demo?). If it validates, M11-proper can build on the learnings; if it doesn't, we've paid 4–8 hours to learn that and can pivot to D (Cloudflare Tunnel over a hardened production backend).

## Open questions

- **Which backbone to demo on — 7B dev-tier or 32B-AWQ production?** 7B is what we've got running today and matches the published benchmark numbers. 32B is the production target and would make the demo more impressive, but the AWQ build on HF Spaces' A10G might not fit cleanly. **Default: start with 7B; revisit after M4e lands.**
- **Does the Space need auth at all?** For friends-and-nerds, a hard per-session request cap plus HF's built-in rate limiting is probably enough. For M11-ranked we'd add real auth. **Default: no auth for the demo, document the request cap on the page.**
- **Do we expose `/v1/generate` on the demo?** Our baselines use it internally, but exposing it publicly means we're hosting a general LLM text endpoint, which is a different liability profile (jailbreaks, harmful content). **Default: no. Bernoulli-only.**
- **Does Gradio's event model handle streaming distributions well?** Our decide path is one-shot; no streaming needed. Non-issue. **Default: static response, no streaming.**
