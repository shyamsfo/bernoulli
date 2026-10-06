# web/ — bernoulli.live landing page

Self-contained guide for working on the public landing page at [https://www.bernoulli.live/](https://www.bernoulli.live/). Written so this folder is useful in isolation — load just `web/` into Claude Desktop (or any tool) and you have everything needed to iterate and deploy.

## What this page is

The landing page for **Bernoulli**, an open-source decision model. Bernoulli takes unstructured state (text) plus typed questions (choice / binary / rating) and returns calibrated probabilities you can branch on — one forward pass per question, no generated tokens to parse. Think "classifier simplicity, LLM depth."

The source project lives at [github.com/shyamsfo/bernoulli](https://github.com/shyamsfo/bernoulli). Useful references when editing this page:

- **[README.md](https://github.com/shyamsfo/bernoulli/blob/main/README.md)** — current project state, milestones, benchmark numbers.
- **[USECASES.md](https://github.com/shyamsfo/bernoulli/blob/main/USECASES.md)** — the "Why Bernoulli instead of a prompt+parse pipeline?" pitch + 4 realistic use cases. The source of truth for the Why and Use cases sections on the landing page.
- **[HTTP.md](https://github.com/shyamsfo/bernoulli/blob/main/HTTP.md) / [CLI.md](https://github.com/shyamsfo/bernoulli/blob/main/CLI.md)** — the two interface guides. The three question types and their request/response shapes.
- **[evals/reports/](https://github.com/shyamsfo/bernoulli/tree/main/evals/reports)** — real benchmark numbers (SST-2 raw/calibrated/generative, AG News, BoolQ, Banking77, load test). Keep page metrics in sync with these when they change.

## Files

```
web/
├── index.html    # single-page landing — inlined CSS + JS, no build step
├── favicon.svg
├── justfile      # deploy recipes (local to this folder)
├── CLAUDE.md     # this file
└── README.md     # short "how to edit + deploy" (parent-project-oriented)
```

No build pipeline. Edit, deploy, done.

## Page structure

Single HTML file, section-per-section. In reading order:

| Section id   | Purpose                                                  |
|--------------|----------------------------------------------------------|
| hero         | Headline + lede + CTAs (GitHub / Book a demo / usage guide) + live demo card (right side). |
| `#why`       | Why Bernoulli vs a traditional LLM — side-by-side JSON code lanes + 4-wins strip. |
| `#use-cases` | Three use-case cards (Support triage / LLM guardrails / Content moderation) with links to USECASES.md. |
| `#features`  | 6-tile feature grid.                                     |
| `#how`       | 5-step visual pipeline — Build prompt → Forward pass → Keep labels → Debias → Calibrate. |
| `#benchmarks`| SST-2 table (Bernoulli vs generative baseline) + 3 highlight callouts. |
| `#trust`     | Confidence story — waffle chart + interactive coverage slider. |
| `.final`     | CTAs (GitHub / Book a demo / Contact us) + git clone one-liner. |
| `#demo-dialog` | "Book a demo" popup (native `<dialog>`), opened by any `[data-open-demo]` button. See *Book a demo form* below. |
| `footer`     | Tagline + `contact@deepstore.ai` mailto + Jev attribution. |

Nav links reference these ids. Keep ids stable if restructuring.

Top nav: Why · Use cases · How it works · Benchmarks, then an accent-filled **Book a demo** button (`.btn-accent`, opens the dialog), the theme toggle, and GitHub. `#trust` deliberately has no nav link. On narrow screens the text links hide (≤760px), GitHub shrinks to its icon (≤480px), and the theme toggle hides (≤340px) so the bar never overflows.

## Design system

CSS variables live at the top of the `<style>` block. Light + dark themes auto-switch via `prefers-color-scheme`; explicit override via `data-theme="light|dark"` on `<html>`.

**Theme toggle (two-state):** first visit follows the system setting (no `data-theme` set, CSS media query decides). The nav button flips light ↔ dark, sets `data-theme`, and saves the choice in `localStorage['theme']`. Its icon shows the theme a click switches *to* (moon on light, sun on dark). A tiny inline script in `<head>` applies the saved choice before first paint so there's no flash. While no choice is saved, the icon tracks live system changes. There is deliberately no "auto" state in the UI.

**Type:**
- `Instrument Serif` — hero H1, section H2s, big stat numerals.
- `Inter` 400/500/600 — body + UI chrome.
- `JetBrains Mono` 400/500 — code, tags, eyebrows, stats units.

**Palette tokens:**
```
--bg        page background
--surface   cards, panels
--surface-2 nested panels, tag backgrounds
--ink       primary text
--ink-2     secondary text
--ink-3     tertiary / meta text
--line      hairlines, borders
--accent    brand blue (links, highlights, active states)
--accent-soft background wash for accent chips
--good      green (correct, succeeded)
--warn      amber (wrong, caution)
--code-bg   dark background for code lanes
```

**Patterns that recur** (and should be reused rather than reinvented):
- `.eyebrow` — 12px uppercase JetBrains Mono, used above every `h2`.
- `.sec-head` — the heading block at the top of each section (`eyebrow + h2 + p`).
- `.callout` — card with Instrument Serif numeral (`.big`) + body paragraph. Used in Benchmarks.
- `.tag` — small accent-soft pill, used for question-type labels.
- `.code` with `.k` / `.s` / `.c` spans — syntax-highlighted dark panels. Used in API + Why sections.

**Grid breakpoints:**
- `900px` — collapses most 3-col grids to 2.
- `600px` / `560px` — collapses 2-col to 1.

**Keeping metrics honest:**
- The hero facts (`~145 ms p50`, `0.025 ECE`, `0 outbound calls`) and the Benchmarks table come from real eval runs. When upstream numbers change, update these.
- The ~70 ms "per added question" number in the Why 4-wins strip comes from the M4f load test ([`evals/reports/loadtest.md`](https://github.com/shyamsfo/bernoulli/blob/main/evals/reports/loadtest.md)). Current steady-state on A10G 24GB with reverse debias.

## Book a demo form

The "Book a demo" popup writes each request as a row in a Google Sheet, via a **Google Apps Script web app** bound to that sheet. No third-party form service.

**Wiring (in `index.html`):**
- Any element with `data-open-demo` opens `#demo-dialog`. Close via `data-close-demo`, Esc, or backdrop click.
- Fields: Name, Company, Email (required, validated client-side), Notes (optional).
- `submitDemoRequest()` in the inline script is the only integration point. It POSTs `application/x-www-form-urlencoded` (a CORS "simple request", so no preflight) to `DEMO_ENDPOINT` with keys `Name`, `Email`, `Notes`, `Company`, plus the honeypot `website`.
- Success requires the script to reply `{"ok":true}`. Anything else (HTTP error, `{"ok":false}`, network failure) shows the inline error with the `contact@deepstore.ai` fallback and keeps the user's input.
- `.hp` / `#d-website` is a visually hidden honeypot. The script silently drops rows where it's filled.

**The sheet:**
- Row 1 headers: `Name | Email | Notes | Company`. Optional `Timestamp` column anywhere, which the script fills with the submission time.
- The script maps columns **by header name**, so columns can be reordered freely. A new column only gets data if the page sends a key with exactly that name.

**The script** (Extensions → Apps Script in the sheet), `doPost(e)`:
- Reads `e.parameter`, drops honeypot hits, appends one row in header order with a `LockService` lock.
- Sanitizes values: caps at 5000 chars and prefixes `'` to anything starting with `= + - @` (blocks formula injection).
- Returns JSON via `ContentService`.
- Deployed as **Web app — Execute as: Me — Who has access: Anyone**.

**Changing the script — important:** use **Deploy → Manage deployments → ✏️ → Version: New version**. That keeps the same `/exec` URL. "New deployment" mints a new URL; if that happens, update `DEMO_ENDPOINT` in `index.html` and redeploy the page.

**Testing:**
- Script alone: `curl -sL -d "Name=Test&Email=t@example.com&Notes=hi&Company=Acme" "$DEMO_ENDPOINT"` should return `{"ok":true}` and add a row. Delete the test row afterwards.
- Page: after `just deploy`, submit once on the live site and confirm the row plus the thank-you screen.
- For automated/local tests, intercept `script.google.com` (e.g. Playwright `page.route`) so test runs don't write rows.

**History:** a SheetMonkey endpoint was tried first and dropped (rows weren't landing in the sheet).

## Recipes

```bash
just preview        # open index.html in the default browser (file://)
just serve-local    # python http.server on :8000 — closer to prod behavior
just deploy         # rsync to ssd2:/var/www/bernoulli.live/html/
just check          # curl the live site, assert HTTP 200, echo <title>
just diff-live      # unified diff: what's live vs local index.html
```

**`deploy`** uses `--rsync-path="sudo rsync"` because `/var/www/bernoulli.live/html/` is root-owned. Needs passwordless `sudo` for your SSH user on `ssd2`. Repo-only files (`README.md`, `CLAUDE.md`, `justfile`) are excluded from the live site.

**`check`** fails with exit 1 if the status isn't 200 — safe to chain in CI or in a post-deploy script.

**`diff-live`** compares your working copy of `index.html` against what's served from the live URL. Expect a trailing-newline difference (nginx strips the final newline from static files); any other diff means either an unshipped local edit or someone edited the server directly.

No staging environment — the site is small enough that eyeballing a diff + a local preview covers it. If deploys get risky later, add a staging host step here.
