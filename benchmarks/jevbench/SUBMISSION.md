# JevBench — leaderboard submission procedure

Submission is a **manual web-form upload** at
[benchmarkheaven.com/submit](https://benchmarkheaven.com/submit) — Benchmark Heaven
does not expose a submission API. This doc is the checklist to run through once a
new result-run is captured in `results/<YYYY-MM-DD>.md`.

Latest run captured here: [`results/2026-10-08.md`](results/2026-10-08.md).

## Before you open the form

1. **Verify there is a committed result run.** The submission references a
   specific commit SHA — the numbers must be reproducible by anyone cloning the
   repo at that SHA and running the commands in `results/<date>.md § Reproduce`.
   `git log --oneline benchmarks/jevbench/results/` finds it.
2. **Check CI is green on that SHA.** If any post-run commit broke `just ci`,
   submit from a known-green SHA, not `main`.
3. **Pick the public URL you want listed.** Benchmark Heaven displays this on the
   leaderboard row — typically your GitHub repo (`https://github.com/shyamsfo/bernoulli`).
   If the repo URL ever changes, that's the field to update.
4. **(Optional) Decide fast lane vs free queue.** The free FIFO queue is
   evaluated in order received; paid 48h fast lane costs money. Default is free.

## Form fields

Fill in the form at [benchmarkheaven.com/submit](https://benchmarkheaven.com/submit):

| Field | Value |
|---|---|
| Model name | `Bernoulli (Qwen2.5-VL-7B)` |
| GitHub / public URL | `https://github.com/shyamsfo/bernoulli` |
| Email | your real email (the queue status email goes here) |
| Benchmark selection | `JevBench` |
| Access type | self-hosted / local — results come from a reproducible run, not a bookable endpoint |

### Notes field (paste verbatim, swap the `<SHA>` + `<date>` placeholders)

```
Dev-tier backbone Qwen2.5-VL-7B-Instruct @ cc594898
on AWS g5.xlarge spot (A10G 24GB), scored via native logprobs at the
answer position (probs_source="native"), not JSON-schema-constrained text
generation. Production step-up to Qwen2.5-VL-32B-AWQ is parked on AWS
capacity (tracked in product/milestones.md M4e).

Results reproducible from:
  https://github.com/shyamsfo/bernoulli/blob/<SHA>/benchmarks/jevbench/results/<date>.md

Harness: fstandhartinger/jevbench @ bb05a335.
Adapter: benchmarks/jevbench/adapter.py (BernoulliLocalAdapter).

Public cohort (231 tasks, 0 failures):
  original  acc 0.833   ECE 0.116   p50 92 ms
  easy      acc 1.000   ECE 0.003   p50 92 ms
  hard      acc 0.432   ECE 0.264   p50 338 ms
```

For a run other than 2026-10-08, pull the three per-split summary numbers from
`results/<date>.md § Headline — per split` and paste them under the final line.

### Artifacts to attach (if the form accepts files)

- `results/<date>.md` — the writeup
- `results/<split>-<date>.summary.json` for `split ∈ {original, easy, hard}` — the
  full jevbench-v1 summarizer output
- `results/<split>-<date>.jsonl` + `.ledger.json` — raw per-task responses + reserve/settle ledger

If the form only accepts one file, upload the markdown and link the JSON summaries
by raw GitHub URL (`https://github.com/shyamsfo/bernoulli/blob/<SHA>/...`).

## After you submit

1. **Record the submission ID** (if the form issues one) in the next commit
   message so future-you can trace it: `M10 step 5: submission queued, id=<ID>`.
2. **Wait on the free FIFO queue.** No ETA is published; expect days to weeks.
3. **When the entry posts**, run the follow-ups tracked in `product/milestones.md` M10:
   - Link the leaderboard URL from `README.md` ("as ranked on JevBench — rank #N")
   - Add the row to `web/benchmarks.html` + link to the Benchmark Heaven page
   - Append a benchmark-state snapshot documenting rank + gap-to-#1
4. **If the Capability Score comes back sub-30**, don't celebrate — the adapter
   translation is likely wrong. Debug per the ordinal entry in `parking-lot.md`
   before announcing anything.

## Re-submitting after a backbone change

When the production step-up to Qwen2.5-VL-32B-AWQ lands (M4e), the submission is
a **new row**, not an edit to the existing one. Benchmark Heaven keeps old rows
pinned to their reported configuration — the comparison between dev-tier and
production is on the leaderboard itself.

Procedure:
1. Re-run all three splits with `BERNOULLI_MODEL_ID` + `BERNOULLI_MODEL_REVISION`
   pointing at the production backbone — commands in `results/2026-10-08.md § Reproduce`.
2. Write `results/<new-date>.md` from the template of `results/2026-10-08.md`.
3. Submit a new row with `Model name: Bernoulli (Qwen2.5-VL-32B-AWQ)` and the
   updated numbers in the Notes field.
4. Both rows stay on the leaderboard — delta between them is the step-up value.
