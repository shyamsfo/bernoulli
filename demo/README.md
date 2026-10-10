# demo/ — build brief for the Jev-via-OpenRouter companion app

This directory is **not an app** — it's a build brief for a separate project.

**For a human reader**: see [`BUILD-PROMPT.md`](BUILD-PROMPT.md). It describes a web app that mirrors the Bernoulli Gradio demo's UX but calls Jev through OpenRouter instead of our scorer.

**For another Claude Code instance**: `cd` into this directory, open a fresh Claude Code session, and point it at `BUILD-PROMPT.md`. The brief is self-contained — it carries the 9 pre-canned examples, UI copy, OpenRouter API contract, tech-stack guidance, and success criteria. No need to read the Bernoulli codebase.

## Why this is a prompt doc rather than a code copy

The reference Gradio app ([shyamsfo-bernoulli-demo.hf.space](https://shyamsfo-bernoulli-demo.hf.space)) is ~300 lines of Python, 90% of which is Gradio framework scaffolding that wouldn't port to a vanilla web app. The genuinely transferable content — the examples, the semantic contract per question type, the "Show the API request" affordance pattern, the dev-mode caveats — all fits in one Markdown file. A focused brief delivers more value than a code dump the next Claude would have to filter.

## Deliverable will live elsewhere

The Jev demo is a separate project, not part of the bernoulli repo. When the companion app is built, update this README with its public URL + source repo link for cross-reference.
