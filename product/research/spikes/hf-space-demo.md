# Spike — HuggingFace Space + Gradio demo

**Status**: in progress
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
- [ ] **Step 2 — Swap in a real scorer locally** *(target: 1-2 h)*. Rewrite `app.py` to load `HFScorer` (CPU is fine for structural verification — ignore latency), import `decide` from `bernoulli`, wire the three question types into Gradio inputs. Verify output distributions render correctly in `gr.Label` or `gr.BarPlot`. Run locally (`python app.py`), not on HF yet.
- [ ] **Step 3 — Push to the Space with A10G Small hardware** *(target: 1 h, mostly wall-clock)*. Change hardware tier in the Space settings, push `app.py` + `requirements.txt`. Watch the build. First time will download the 15 GB Qwen weights; expect 5-10 min for the initial build.
- [ ] **Step 4 — Deal with model caching** *(target: 1 h, could blow up)*. HF Spaces have a persistent `/data` volume; configure `HF_HOME` to point there so cold-start restarts don't re-download weights. Verify by stopping + starting the Space and timing the warm-up on the second start.
- [ ] **Step 5 — Gradio UI polish** *(target: 1-2 h)*. Three tabs for choice/binary/rating with example prompts pre-filled. Response panel renders the full distribution as a bar chart. Warming-up loading state with a "~45 s on first visit" note. Markdown explainer at the top with the Bernoulli pitch + a link to `benchmarks.html`.
- [ ] **Step 6 — Per-session rate cap + abuse control** *(target: 30 min)*. Hard cap on requests per session (`gr.State` counter). Rate limit by Gradio's queue (`.queue(max_size=10)`). Document the caps in the UI.
- [ ] **Step 7 — End-to-end verification** *(target: 30 min)*. Share the URL with yourself on your phone, run through all three question types, confirm cold-start behavior from a cold Space. Note any surprises.
- [ ] **Step 8 — Write the assessment at the bottom of this doc** *(target: 30 min)*. Did Gradio compose cleanly? Does sleep-when-idle work? Is the UX bar met for the audience? What breaks if we scale this up for M11?

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

## Assessment (to be filled at the end)

<!-- Three sections to write at the end:

1. **Did Gradio + our scorer compose cleanly?** Yes/no, what resisted.
2. **Does HF Spaces' sleep-when-idle behave as advertised?** Cold-start measurement, idle-burn observation.
3. **Is the UX bar for friends-and-nerds met?** Honest answer; if yes, link to the Space; if no, what fails.

4. **What this means for M11.** Recommendation to M11 Task 1: build on this shape, or fall back to Option D (Cloudflare Tunnel), or something else.
-->
