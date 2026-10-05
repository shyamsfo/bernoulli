"""`bernoulli decide` CLI — thin wrapper around decide().

Usage:
    bernoulli decide --state state.txt --question question.json
    bernoulli decide --request request.json

The --request form takes a full DecideRequest JSON; the --state/--question
pair is a convenience for the common single-question case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bernoulli.config import load_settings
from bernoulli.decide import decide
from bernoulli.types import DecideRequest, State


def _build_request(args: argparse.Namespace) -> DecideRequest:
    if args.request:
        payload: dict[str, Any] = json.loads(Path(args.request).read_text())
        return DecideRequest.model_validate(payload)

    if not (args.state and args.question):
        raise SystemExit("provide --request OR both --state and --question")

    state_text = Path(args.state).read_text()
    question_payload = json.loads(Path(args.question).read_text())
    questions = question_payload if isinstance(question_payload, list) else [question_payload]
    return DecideRequest.model_validate({"state": State(text=state_text), "questions": questions})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bernoulli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_decide = sub.add_parser("decide", help="score one or more questions against a state")
    p_decide.add_argument("--request", help="path to a full DecideRequest JSON")
    p_decide.add_argument("--state", help="path to a text state file")
    p_decide.add_argument("--question", help="path to a question JSON (object or array)")
    p_decide.add_argument("--model", help="override backbone (defaults to BERNOULLI_MODEL_ID)")

    args = parser.parse_args(argv)

    if args.cmd == "decide":
        from bernoulli.scorer import load_scorer

        settings = load_settings()
        if args.model:
            # Simple override: swap just the model_id. Revision stays pinned so
            # someone passing a different checkpoint has to also set
            # BERNOULLI_MODEL_REVISION explicitly.
            settings = settings.model_copy(update={"model_id": args.model})
        scorer = load_scorer(settings)
        request = _build_request(args)
        response = decide(request, scorer)
        print(response.model_dump_json(indent=2))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
