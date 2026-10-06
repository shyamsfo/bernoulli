"""Build the post-cutoff arXiv classification snapshot.

Not run in CI; invoke manually on the dev box when generating or refreshing
`data/dataset.jsonl`:

    uv add arxiv
    uv run python -m benchmarks.academic.arxiv_post_cutoff.make_dataset \
        --target 200 \
        --out benchmarks/academic/arxiv_post_cutoff/data/dataset.jsonl

The script queries the arXiv API for recent papers in each of the four
`CATEGORIES`, filters to publication dates strictly after
`BACKBONE_CUTOFF`, balances per-category counts, splits into
`train`/`test` deterministically, and writes a JSONL with one example
per line.

**Fields written per row**:

- `id` — the arXiv identifier (e.g. `2501.12345`).
- `title` — raw paper title.
- `abstract` — raw paper abstract.
- `primary_category` — one of the four `CATEGORIES`.
- `publication_date` — ISO date string, strictly after `BACKBONE_CUTOFF`.
- `split` — `"train"` or `"test"`.

**Why this script exists:**
The post-cutoff claim is load-bearing — the whole benchmark is pointless
if any example could have been in the backbone's pretraining data. This
script makes the cutoff rule mechanical (not a judgment call) and keeps
the generation reproducible so the claim survives backbone swaps:

- Update `BACKBONE_CUTOFF` in `loader.py` when swapping to a backbone
  with a later cutoff.
- Re-run this script; commit the regenerated `data/dataset.jsonl`.
- Update the model-cutoff claim in `benchmarks/README.md` and each run's
  `results/latest.md`.

Status: skeleton only. `arxiv` dep + the real fetch loop land in M8 task 10b.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from benchmarks.academic.arxiv_post_cutoff.loader import BACKBONE_CUTOFF, CATEGORIES


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        type=int,
        default=200,
        help="Total number of papers to collect across all categories (balanced).",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.5,
        help="Fraction of rows to assign to the 'test' split; the rest go to 'train'.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the JSONL snapshot.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260101,
        help="Deterministic split seed; also controls per-category ordering tiebreaks.",
    )
    return p.parse_args()


def main() -> None:  # pragma: no cover — invoked manually on the dev box, not in CI
    args = _parse_args()

    cutoff = datetime.fromisoformat(BACKBONE_CUTOFF).date()
    print(f"Target: {args.target} papers (balanced across {len(CATEGORIES)} categories)")
    print(f"Cutoff: strictly after {cutoff.isoformat()}")
    print(f"Output: {args.out}")

    # TODO (10b): implement the arxiv-API fetch loop here.
    #
    #   import arxiv
    #   client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    #   per_cat = args.target // len(CATEGORIES)
    #   rows: list[dict] = []
    #   for cat in CATEGORIES:
    #       search = arxiv.Search(
    #           query=f"cat:{cat}",
    #           max_results=per_cat * 3,
    #           sort_by=arxiv.SortCriterion.SubmittedDate,
    #           sort_order=arxiv.SortOrder.Descending,
    #       )
    #       for result in client.results(search):
    #           pub = result.published.date()
    #           if pub <= cutoff:
    #               continue
    #           rows.append({...})
    #           if sum(1 for r in rows if r['primary_category'] == cat) >= per_cat:
    #               break
    #
    # Then deterministically split train/test (seeded shuffle), and write JSONL.
    #
    raise SystemExit(
        "make_dataset: skeleton only. The arxiv-API fetch lands in M8 task 10b.\n"
        "Install the dep with `uv add arxiv` and implement the TODO loop above."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
