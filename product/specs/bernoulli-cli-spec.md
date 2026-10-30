# bernoulli-cli — Spec

A thin CLI client for `POST /v1/decide` to enable quick testing from the terminal without the Chrome extension, Postman, or hand-rolled `curl` + `jq` pipelines.

## Why this lives in the bernoulli repo

This is a **reference client**. Colocating with the Pydantic request/response models means schema drift breaks the CLI build in the same PR that introduces it — wire-format regressions get caught locally rather than at the next Chrome-plugin build. The Chrome extensions (`github.com/shyamsfo/bernoulli-chrome`) are production clients but can't fill the reference-client role: they're Vite + MV3 + TypeScript, and their wire-format assumptions only surface after a full extension rebuild + manual Chrome reload.

Suggested location inside this repo: a `cli/` subpackage exposed as a `bernoulli` (or `bernoulli-cli`) console script via `pyproject.toml` entry point. Reuse the existing Pydantic request/response models directly — do not duplicate the schema.

## Usage

```
bernoulli decide \
    [-f <state-file> | -t "inline text"] \
    -l <lens-pack.json> \
    [--url <api-url>] \
    [--mock] \
    [--raw] \
    [--timeout <seconds>]
```

Examples:

```bash
# Smoke test against the shared mock
BERNOULLI_API_KEY=xxx bernoulli decide --mock -t "hello world" -l general.json

# Real file, real endpoint
export BERNOULLI_API_URL=https://api.bernoulli.live
export BERNOULLI_API_KEY=xxx
bernoulli decide -f ./tos-page.txt -l ./tos-lens-pack.json

# Dump raw JSON for piping into jq
bernoulli decide -f state.txt -l lenses.json --raw | jq '.answers[0]'
```

## Flags

| Flag | Required | Description |
|------|----------|-------------|
| `-f, --state-file PATH` | one of `-f` / `-t` | Read state text from a UTF-8 file. |
| `-t, --text STRING` | one of `-f` / `-t` | Inline state text. Convenience for one-liners. |
| `-l, --lens-pack PATH` | yes | JSON file: either `{"questions": [...]}` or a bare `[...]` array of question objects. See Lens pack format below. |
| `--url URL` | — | API base URL. Overrides `BERNOULLI_API_URL`. |
| `--mock` | — | Shortcut for `--url https://bmock.nuwire.dev`. |
| `--raw` | — | Print the raw JSON response instead of pretty-rendering. |
| `--timeout SEC` | — | Default: 60. |
| `-v, --verbose` | — | DEBUG logging to stderr; prints the outgoing request body. |

## Environment variables

| Var | Description |
|-----|-------------|
| `BERNOULLI_API_URL` | Base URL for the API. Required unless `--url` or `--mock` is passed. |
| `BERNOULLI_API_KEY` | Credential. Auth header shape to match whatever the current API expects. |

## Input formats

### State file (`-f`)

Plain UTF-8 text. Shipped as the `text` field of the `state` object. The CLI does **not** prepend `URL:` / `Title:` headers — that's a Chrome-extraction-specific convention and does not belong in a generic client. If a caller wants headers, they can pre-format the file.

Image inputs are out of scope for v1. The API's `state.images` field exists; a future `-i IMAGE_PATH` flag can revisit this when there's a concrete use case.

### Lens pack (`-l`)

JSON, two accepted shapes:

```json
{"questions": [{"id": "...", "prompt": "...", "response_type": "...", ...}]}
```

or bare:

```json
[{"id": "...", "prompt": "...", "response_type": "...", ...}]
```

Both get normalized to `questions: [...]` in the request.

**UI-only fields must be stripped before serializing.** The Chrome plugins carry fields like `label`, `group`, `description` on their lens definitions. The Pydantic `Question` model is `extra="forbid"`, so these would 422 the request. The CLI should validate each item against the server's own `Question` model and either strip unknown fields with a warning or hard-fail — pick a default; `--strict` can toggle the other.

## Output

### Default (pretty)

For each answer in the response, print:
- Lens `id` (dim)
- Decision value (bold)
- Confidence (color-coded: green ≥ 0.8, yellow 0.5–0.8, red < 0.5)
- Reasoning (indented, wrapped)

Use `rich` or similar. Suppress color if stdout is not a TTY.

### `--raw`

Full response JSON to stdout (pretty-printed when stdout is a TTY, compact otherwise). Nothing else on stdout. Logs go to stderr.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | API returned 2xx and the response parsed. |
| 1 | CLI error — bad flags, file not found, invalid JSON in lens pack, missing env var. |
| 2 | API error — non-2xx response. Error body is printed to stderr. |
| 3 | Network / timeout. |

## Implementation notes

- Python; reuse the server's Pydantic request/response models directly.
- CLI framework: `typer` or `click` — pick what the repo already uses.
- No config file support in v1. Env vars + flags only.
- No batch mode, no file-watch mode, no interactive REPL. Revisit if quick-testing needs them.
- Logging via `logging` to stderr.

## Non-goals

- Not a general-purpose API client library. Python callers who want programmatic access should import the models directly.
- Not a replacement for the Chrome extensions' extraction pipeline (Readability, tab packing). The CLI assumes callers hand it clean state text.
- Not a lens-pack authoring or validation tool. It consumes packs; it does not create them or validate semantics beyond "does it parse and pass `Question`-model validation."

## Cross-references for context

- API wire format: `API.md` in this repo. Request shape is `{state, questions, options}` where `state` and each `questions[]` item are `extra="forbid"`.
- Reference consumers that build the same request today: `github.com/shyamsfo/bernoulli-chrome` — see `bernoulli-general/src/` and `bernoulli-tos/src/`, specifically how they strip UI-only fields from questions before serializing. The CLI must do the equivalent, server-side.
- Mock endpoint: `https://bmock.nuwire.dev` (shared with the Chrome plugins and `bernoulli-sfdc`).
