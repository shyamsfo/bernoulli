# Project Management Principles — Lite Mode

This is a **lite-mode** project. It uses a stripped-down version of the `ds-work-*` system: just milestones, backlog, parking-lot, recurring, and session reports. Use it when the work has a clear execution path and doesn't need product-level framing or per-milestone design records.

For the full command reference, run **`/ds-work-user-guide`**.

To promote this project to full mode (adds vision.md, roadmap.md, now.md, per-milestone PRDs/PLANs, design reviews), run **`/ds-work-graduate`**.

---

## The Document Stack (lite)

```
product/
├── ds-work-mode.txt               ← contains "lite"
├── milestones.md               ← all milestones in one file. Sections per milestone with goal + task checklist. **This file IS the plan.**
├── backlog.md                  ← low-commitment mind-dump (no priority, no tags)
├── parking-lot.md              ← unscheduled work items (bugs, chores, ideas captured for later)
├── recurring.md                ← recurring-maintenance list (tasks that decay and need periodic touching)
├── project-management-principles.md   ← this file
├── research/                   ← pre-decision options analyses (ds-work-research)
│   └── spikes/                 ← time-boxed feasibility investigations (ds-work-spike)
├── operations/                 ← cluster runbooks, infra how-tos
├── learnings/                  ← post-hoc reports, incident retrospectives
├── reports/                    ← session history (YYYY-MM-DD.md, written by update/halt)
└── how-to/                     ← project-specific skill extensions
    ├── ds-work-continue.md
    ├── ds-work-halt.md
    ├── ds-work-status.md
    └── ds-work-update.md
```

**Skipped vs full mode:** `vision.md`, `roadmap.md`, `now.md`, `design/` (PRDs + PLANs), `reviews/` (challenge reports), `market-research/`, `one-pager.md`, `elevator-pitch.md`, and the `how-to/ds-work-{plan,challenge}.md` extensions.

### `milestones.md` — *the plan*

All milestones live here, one section each. The active milestone is the first one marked `🔄 in progress`. `/ds-work-continue` and `/ds-work-status` find the next `[~]` or `[ ]` task within it.

Task checkboxes:
- `[ ]` not done · `[x]` done · `[~]` in progress · `[-]` skipped / won't do

Milestone status markers:
- `⏳ pending` · `🔄 in progress` · `✅ done` · `🚫 blocked`

There is no separate "sub-state" (no PRD/PLAN files), so `/ds-work-continue` and `/ds-work-status` simply surface the next unchecked task and call it done. If a milestone outgrows a flat checklist, run `/ds-work-graduate`.

### `backlog.md` — mind-dump (unchanged from full mode)

Capture half-formed thoughts. No priority. Append-only. Managed by `/ds-work-backlog`. Sweep periodically to promote items to the parking-lot or drop them silently.

### `parking-lot.md` — unscheduled work (unchanged from full mode)

Items that are worth tackling eventually but aren't in the active milestone. Tagged. Managed by `/ds-work-parking-lot`. In lite mode, `promote` adds the item to the active milestone's task list in `milestones.md` (not to a PLAN file, since none exists).

### `recurring.md` — recurring-maintenance list (unchanged from full mode)

Tasks that decay and need periodic touching (refresh a snapshot, sweep a directory, re-validate a pin). Cadence marker + pointer + last-run field per item. Managed by `/ds-work-recurring`. `/ds-work-status` and `/ds-work-continue` surface overdue counts; `/ds-work-halt` asks about done items each session.

### `reports/` — session history (unchanged from full mode)

One file per day, written by `/ds-work-update` (mid-session) and `/ds-work-halt` (session end).

---

## Commands Available in Lite Mode

| Command | Notes |
|---|---|
| `/ds-work-continue` | Reads `milestones.md`, finds active milestone + next unchecked task |
| `/ds-work-status` | Same as continue, but tighter output |
| `/ds-work-update` | Mid-session snapshot to today's report — same as full mode but skips `now.md` |
| `/ds-work-halt` | Tick `milestones.md`, write report, commit, push, run project-specific teardown. Does not touch a PLAN file because there isn't one. |
| `/ds-work-backlog` | Unchanged from full mode |
| `/ds-work-parking-lot` | Unchanged from full mode |
| `/ds-work-recurring` | Unchanged from full mode |
| `/ds-work-spike` | Unchanged from full mode |
| `/ds-work-research` | Unchanged from full mode |
| `/ds-work-graduate` | Promote this project to full mode |

Commands that target full-mode-only artifacts (`/ds-work-vision`, `/ds-work-roadmap`, `/ds-work-plan`, `/ds-work-one-pager`, `/ds-work-elevator-pitch`, `/ds-work-challenge`) detect lite mode and offer to run `/ds-work-graduate` first. They do not create vision/roadmap/PRDs in a lite project.

---

## Typical Session Flow

```
Start session
  → /ds-work-continue           (reads milestones.md, surfaces next task)

Do the work
  → edit files, run commands
  → /ds-work-update             (optional mid-session checkpoint)

End session
  → /ds-work-halt               (ticks milestones.md, writes report, commits, pushes)
```

---

## When to Graduate

Run `/ds-work-graduate` when:
- The project picks up a real product story (problem framing, customer, north star) worth writing down
- A milestone has enough internal design decisions to need its own spec (PRD) and execution plan (PLAN) — flat checklists in `milestones.md` stop being enough
- You want adversarial reviews of planning artifacts (`/ds-work-challenge`)
- You need external-facing summaries (`/ds-work-one-pager`, `/ds-work-elevator-pitch`)

Graduation is one-way and additive — existing milestones and their checkboxes are preserved as-is; new full-mode artifacts (vision.md, roadmap.md, now.md, design/) are added alongside.

---

*This is the lite-variant principles doc. The canonical copy is distributed by `/ds-work-scaffold` when invoked with `--lite`. To update for all future lite projects, edit `~/.claude/commands/ds-work-scaffold.md`.*
