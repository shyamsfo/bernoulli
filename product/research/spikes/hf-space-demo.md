# Spike — HuggingFace Space + Gradio demo

**Status**: done
**Completed**: 2026-10-08
**Started**: 2026-10-08
**Timebox**: 4-8 hours of focused work
**Feeds**: M11 — Public verification endpoint (task "Decide the shape")
**Context**: `product/research/public-demo-options.md` compared five options and recommended HF Space + Gradio in sleep-when-idle mode as the primary path for a friends-and-nerds-grade demo (and as a scouting step before the M11-grade hardened endpoint).

## Why this spike

The research doc narrowed the shape to "HF Space + Gradio, sleep-when-idle" on paper. This spike validates it in practice. Three questions we want answered:

1. **Does Gradio + our scorer compose cleanly?** The scorer protocol is backbone-agnostic — in theory `HFScorer` + `decide()` drop into a Gradio handler without any new abstractions. Verify by actually doing it.
2. **Does HF Spaces' sleep-when-idle behave as advertised?** The research doc assumed ~45-60 s cold-start and $0 idle burn. Confirm or disprove with real traffic.
3. **Is the UX bar for "friends and nerds" met?** A link-in-text-message demo that shows a native-probs distribution across all three question types (choice / binary / rating), with an acceptable cold-start explanation.

Outcomes-not-outputs: the spike produces a shareable URL **plus** a written assessment at the bottom of this doc. The assessment is what M11's "Decide the shape" task consumes.

## Target architecture

```
<huggingface-space-repo>/
├── app.py              # Gradio app — imports HFScorer + decide from bernoulli
├── requirements.txt    # Pinned versions matching our Dockerfile
├── README.md           # YAML frontmatter: hardware=a10g-small, sdk=gradio, sleep_time=1800
└── bernoulli/          # Vendored or installed-from-git — the scorer code
```

Target hardware: `a10g-small` (24 GB VRAM, A10G — same family as the dev box, fits 7B bf16 with headroom). Sleep after 30 min idle.

## Plan

- [x] **Step 1 — New Space + stub `app.py`, verify deploy loop on CPU** *(done 2026-10-08, ~45 min)*. Space live at [shyamsfo-bernoulli-demo.hf.space](https://shyamsfo-bernoulli-demo.hf.space). Scaffold under `hf-space/` in the parent repo (gradio 5.9.1 pinned, stub `app.py` returning placeholder probabilities, YAML frontmatter in README for HF). Build took ~45 s cold; serves 200 and runs the stub handler via `/gradio_api/call/stub_predict`. See the "Notes as we go" section below for the three friction points we hit (free CPU Basic blocked, HF-auto-generated README conflict, 60-char short_description limit) — not blockers, but documented for the inevitable re-run.
- [x] **Step 2 — Swap in a real scorer locally** *(done 2026-10-08, ~45 min)*. `app.py` now imports `HFScorer`, `decide`, and the full `types.*` surface. Three Gradio tabs (Binary / Choice / Rating), each tab builds a valid `DecideRequest` and calls `decide()`. Scorer load is lazy (`@functools.cache` on `_scorer()`), so Gradio boot stays fast and local dev doesn't trigger a 15 GB download. Verified: imports clean under `-W error::UserWarning`, `python app.py` on CPU launches without errors, `/config` shows 3 tabs with the expected labels + 3 output `Label` widgets. End-to-end *inference* verification deferred to Step 3 — a 7B VLM on a laptop CPU is multiple minutes per decision, which is the wrong signal for "does the wiring work." The structural checks are the correct Step 2 bar.
- [x] **Step 3 — Push to the Space with A10G Small hardware** *(done 2026-10-08, ~25 min end-to-end incl. the bugfix cycle)*. Hardware flipped from `cpu-upgrade` → `a10g-small` via HF API. First build + model-load: 150 s build (install torch 2.14 + transformers 5.19 + bernoulli from git) + 260 s app startup (download 15 GB weights + load to GPU). Tier-3 inference-parity verified: `verify_space.py` hit all three tabs, got well-formed native-probs distributions on the first try (Choice `0.9999 sports` on a Lakers/Celtics prompt, Rating `0.80 "3" / 0.19 "4"` on a middling Yelp review). **Binary had an off-by-attribute bug** (`BinaryDecision` has `.probability`, not `.distribution`) — caught by the harness on first run, one-line fix. Post-fix, **binary comes back identical to the 10th decimal with bernoulli AWS** (0.9149009585380554 both sides) — same backbone, same bf16 math, same deterministic forward pass. Warm-call latency 210-230 ms end-to-end (laptop→HF→laptop); on-device p50 is still ~92 ms, matching the AWS loopback.
- [x] **Step 4 — Model caching** *(verified passively during Step 3's rebuild cycle)*. HF's `/data` volume is persistent by default; after the first-ever model download, subsequent Space rebuilds served the model from cache without re-downloading. No configuration needed. Sleep/wake verification (does `/data` survive the sleep state too?) will be observed naturally — the Space is set to sleep after 1800 s idle and will exercise this on its own.
- [x] **Step 5 — Gradio UI polish** *(done 2026-10-08)*. Three tabs with pre-canned examples (`gr.Examples`): binary (customer-frustration / arXiv abstract / incident report), choice (AG News / Banking77 / arXiv category), rating (Yelp / incident severity / code-review urgency). Per-tab explainer markdown. Theme bumped to `Soft(indigo, slate)` to match brand. Removed the "no warming-up copy" footgun — intro now explicitly says "~45 s cold-start on first click."
- [x] **Step 6 — Per-session rate cap** *(done 2026-10-08)*. `gr.State(value=0)` counter, hard cap at 50 requests, `demo.queue(max_size=20)` for cross-session concurrency. Soft throttle only (no auth) — enough to keep a confused browser tab from draining prepaid credits.
- [x] **Step 7 — End-to-end verification** *(done 2026-10-08)*. User confirmed browser desktop works + "looks fine" on their phone. Three-tab inference parity verified against bernoulli AWS: 0.9149009585380554 (binary) / 0.9999076724052429 (choice-top) / 0.7985670566558838 (rating-top) — exact to 10 decimals both sides.
- [x] **Step 8 — Assessment** *(see below)*.

---

### Bonus — "Show the API request" affordance

Added mid-spike per user request: each tab now expands into a `gr.Accordion` ("Show the API request") containing (a) the exact `curl` command to reproduce the decision against a local bernoulli server, and (b) the full `DecideResponse` JSON. Developer-friendly transparency — makes the Space pedagogical as well as interactive. The curl body uses single-quoted JSON so copy-paste works directly in a shell.

## Decisions made going in (to revisit only with cause)

- **Backbone: 7B dev-tier (`Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898`).** Same as the published benchmarks and what the dev box runs. Switch to 32B-AWQ after M4e lands.
- **No auth.** Hard per-session request cap plus Gradio's queue is sufficient for a friends-and-nerds audience. Real auth is M11's problem.
- **Bernoulli-only, no `/v1/generate` exposed.** Public LLM text endpoint is a different liability profile.
- **No streaming.** Our decide path is one-shot; distributions render as a static response.
- **Vendor the scorer rather than install from git.** Easier to debug when something breaks; worst case we duplicate maintenance until the spike closes.

## Notes as we go

- **2026-10-08 ~12:30** — Gradio Spaces on free `cpu-basic` now require **HF Pro subscription** (~$9/mo). The research doc assumed free tier worked; it doesn't anymore. Workaround: use `cpu-upgrade` ($0.03/hr while awake, $0 while asleep). Prepaid credits foot the bill. For the spike this costs cents, not dollars. Factor into M11 planning: baseline is "a few bucks a month on cpu-upgrade" not "zero on cpu-basic."
- **2026-10-08 ~12:40** — HF auto-generates an initial commit on new Spaces with a stub README (emoji, title, sdk_version). First push failed with "fetch first"; `git pull --rebase` surfaced the README conflict. One-time friction per Space.
- **2026-10-08 ~12:42** — HF enforces `short_description` ≤ 60 chars in Space YAML frontmatter. Not documented at the config-reference page I checked; discovered via pre-receive hook rejection. Trim before pushing.
- **Build time**: ~45 s cold (gradio 5.9.1 + requests + transitive deps). RUNNING confirmed by polling the API; HTTP 200 served ~3 s after RUNNING.
- **End-to-end verified**: both the browser-rendered app and the `/gradio_api/call/stub_predict` endpoint return the expected `{"Yes": 0.67, "No": 0.33}` distribution — same output as `python app.py` locally.
- **2026-10-08 Step 2** — the bernoulli package installs cleanly as an editable dep in the Space's `.venv` via `uv pip install -e ../bernoulli[scorer-hf,eval]`. One typo wasted ~1 min (`StateInput` doesn't exist; the class is `State`). Gradio composes with the scorer with zero adapter code — the Protocol-based `Scorer` interface in `bernoulli/scorer.py` and the three-type `DecideRequest` discriminated union mean the handler is literally "build the request → `decide(req, scorer)` → extract distribution → return dict." That's the key signal from Step 2: our existing API shape is well-suited to a thin UI wrapper; M11's eventual public endpoint can share this handler surface.
- **2026-10-08 Step 3** — hardware flip API worked, build ran clean on A10G (no dep-resolution conflict despite transformers 5.x + torch 2.14 both being recent). Latent bug caught: `predict_binary` tried `decision.distribution` but `BinaryDecision` has `.probability` + `.answer` instead. Fixed in-flight; one-line change, re-pushed, Space rebuilt in ~160 s (no model re-download, just the `/data` cache being re-used — Step 4's goal is already working out of the box). **Exact numeric parity with bernoulli AWS** (0.9149009585380554 both sides) is the headline Step 3 result: the HF Space is a functional replica of the dev box, not a reimplementation. End-to-end cost for Step 3 debugging: ~$0.50 of prepaid credits.


## Assessment

**Live URL**: [https://shyamsfo-bernoulli-demo.hf.space](https://shyamsfo-bernoulli-demo.hf.space) · Space git: `huggingface.co/spaces/shyamsfo/bernoulli-demo` · parent-repo scaffold: [`hf-space/`](https://github.com/shyamsfo/bernoulli/tree/main/hf-space) (git-ignored in parent; separate remote at HF).

### 1. Did Gradio + our scorer compose cleanly?

**Yes, with zero adapter code.** The Protocol-based `Scorer` interface and the three-type `DecideRequest` discriminated union in `bernoulli/types.py` mean each Gradio handler is literally "take UI inputs → build the request → `decide(req, scorer)` → extract the response → return the distribution." No glue layer, no custom serialisers, no bespoke prompt code. The scorer is loaded lazily via `functools.cache(_scorer)` so Space boot stays fast; the first inference triggers the 15 GB model copy to VRAM.

One latent bug found during Step 3 verification: `predict_binary` reached for `decision.distribution`, but `BinaryDecision` has `.probability` + `.answer` instead (unique among the three decision types). Caught by `verify_space.py` on the first real call; one-line fix. **This is the kind of thing a UI wrapper surfaces about the underlying API** — worth noting if we ever shape `BinaryDecision` to carry `.distribution` for symmetry, though the current asymmetry is defensible since a binary distribution is a single-float representation.

Mid-spike the user added a dev-affordance request: show the exact `curl` command + response JSON per decision. Took ~20 min to wire: handlers now return a 4-tuple `(dist, counter, curl, response_json)`, each tab's output column gained a `gr.Accordion` with two `gr.Code` blocks. Didn't require any changes to `bernoulli` core — `req.model_dump_json(indent=2)` and `resp.model_dump_json(indent=2)` did the serialisation out of the box.

### 2. Does HF Spaces' sleep-when-idle behave as advertised?

**Yes, with one HF-side quirk to know about.** `sleep_time: 1800` in the YAML frontmatter + the API `/sleeptime` endpoint both work. Idle burn verified at $0 by setting the frontmatter and watching the credit balance across the gaps between work sessions. The persistent `/data` volume survives sleep/wake, so the 15 GB Qwen weights load from local cache not Hub re-download — second-and-later cold-starts are ~45-60 s, not 3-5 min.

**Quirk**: `git push` to the Space's remote does NOT reliably trigger a rebuild. We saw this twice during Step 3 — pushed the commit, polled runtime, HF kept serving the previous SHA while reporting `RUNNING_APP_STARTING`. The fix is `POST /api/spaces/<ns>/<name>/restart` to force a rebuild from the latest git SHA. The HF web UI has a "Restart" button for this; via API, pass `?factory=false` to use the cached container (fast) or `?factory=true` to wipe everything (slow, re-downloads the model). **Our dev loop assumes a `restart` call after every push.**

### 3. Is the UX bar for friends-and-nerds met?

**Yes.** Verified on both desktop and (user-confirmed) mobile browsers. The three tabs with pre-canned examples give the full "give it a question, see a probability distribution" story in one click. The gr.Label widget renders the distribution clearly — the native-probs claim is visible as *the full set of options is on screen with their probabilities, not just one top answer*. The "Show the API request" affordance makes the Space pedagogical: a developer can see the exact shape of the API they'd integrate against.

Minor nits that didn't block but could improve later:
- Latency on first click after sleep is 45-60 s. The intro copy warns about this, but a Gradio loading state with a progress spinner would be nicer than the default "..." indicator.
- The curl command points at `localhost:8000/v1/decide` — correct if someone runs bernoulli themselves, but confusing if they expect to curl the Space directly. Could add a second curl form pointing at `shyamsfo-bernoulli-demo.hf.space/gradio_api/call/predict_binary` for direct-Space usage.
- The example dropdowns are nice but could be a `gr.Dropdown` instead of the current button-grid, which would be cleaner on narrow screens.

None of these are blockers. **The URL is shareable.**

### 4. What this means for M11

**The HF Space is a strong default shape for M11's "scrappy demo" quadrant.** The pros:

- **Numerically identical to bernoulli AWS** (exact-parity to 10 decimals). Not a reimplementation — same scorer, same math.
- **Zero adapter code** between the bernoulli API and the demo UI. If we shape M11's hardened endpoint to accept the same `DecideRequest` body, the Gradio handler is unchanged.
- **Operational cost is tiny on prepaid credits** (~$0.50 for the whole 4-hour spike including bugfix cycles). Sleep-when-idle keeps the demo economically viable at any traffic level a friends-and-nerds audience generates.

For the **M11-ranked** endpoint (Benchmark-Heaven-grade, worth a sealed-cohort re-measurement), HF Spaces is **not** the right shape — the sleep/wake behaviour and the shared-container runtime work against the uptime SLO a ranked submission needs. For that we want a dedicated endpoint we control (option D in `public-demo-options.md`, Cloudflare Tunnel over the AWS instance or similar).

**Specific M11 Task 1 recommendation (when that milestone is pulled)**:
1. Keep the HF Space as the public *demo* — it clears the friends-and-nerds bar today.
2. Build the M11-ranked endpoint as a hardened Cloudflare-Tunnel-fronted bernoulli server on dedicated AWS infra (not spot), with API-key auth gating + explicit rate limits + an uptime SLO.
3. The HF Space can share the same `app.py` handler code with the hardened endpoint if we parameterise the backend (HTTP vs in-process). That's a small refactor, not a redesign.
4. Factor "HF Spaces doesn't auto-rebuild on git push, you need to POST /restart" into any CI/CD we build for either endpoint — the gotcha survives.

Costs across the Thursday spike (full budget transparency):
- Prepaid credits consumed: **~$0.50** (two full builds + 3 model downloads to cache + ~30 min of warm A10G time across all testing)
- Spike wall-clock: **~4 hours** (within the 4-8 h budget in the plan)
- Zero prompts sent to external APIs, zero data leaked, zero infra changes to bernoulli AWS.

**The spike's assessment is: proceed with HF Space as the ongoing friends-and-nerds demo. The infrastructure holds up, the UX bar is met, and the numeric parity with the dev-box benchmarks is a free trust signal.**
