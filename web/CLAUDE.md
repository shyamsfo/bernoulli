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
| hero         | Headline + lede + CTAs + live demo card (right side).    |
| `#why`       | Why Bernoulli vs a traditional LLM — side-by-side JSON code lanes + 4-wins strip. |
| `#use-cases` | Three use-case cards (Support triage / LLM guardrails / Content moderation) with links to USECASES.md. |
| `#features`  | 6-tile feature grid.                                     |
| `#how`       | 5-step visual pipeline — Build prompt → Forward pass → Keep labels → Debias → Calibrate. |
| `#benchmarks`| SST-2 table (Bernoulli vs generative baseline) + 3 highlight callouts. |
| `#trust`     | Confidence story — waffle chart + interactive coverage slider. |
| `.final`     | CTA + git clone one-liner.                               |

Nav links reference these ids. Keep ids stable if restructuring.

## Design system

CSS variables live at the top of the `<style>` block. Light + dark themes auto-switch via `prefers-color-scheme`; explicit override via `data-theme="light|dark"` on `<html>`.

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

## Deploy

```bash
just deploy
```

The recipe rsyncs the current directory to `ssd2:/var/www/bernoulli.live/html/`, using `--rsync-path="sudo rsync"` because the destination is root-owned. Needs passwordless `sudo` for your SSH user on `ssd2`. Repo-only files (`README.md`, `CLAUDE.md`, `justfile`) are excluded from the live site.

For a local preview before deploying:

```bash
just preview       # opens index.html in the default browser (macOS)
# or just open the file manually
```

No staging environment — the site is small enough that eyeballing a diff + previewing locally covers it. If deploys get risky, add a staging host step here.
