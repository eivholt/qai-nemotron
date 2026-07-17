#!/usr/bin/env python3
"""Recompute hospital results with strict excess-call scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_arena.hospital_logistics_runtime import hospital_scenario
from agent_arena.model_client import write_summary
from agent_arena.pydantic_hospital_logistics_arena import (
    classify_failure,
    diagnose_timeout_or_runtime,
    is_context_exhaustion,
    score_hospital_calls_strict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Hospital results.json to rescore.")
    parser.add_argument("--output", type=Path, help="Optional path for rescored JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.results.read_text(encoding="utf-8"))
    old_passes = sum(bool(row["score"]["passed"]) for row in rows)
    old_avg = sum(float(row["score"]["score"]) for row in rows) / len(rows)

    rescored = []
    for row in rows:
        server_requests = row.get("server_requests", [])
        exception = row.get("exception", "")
        context_exhaustion = (
            row.get("runtime_diagnosis", {}).get("kind") == "context_exhaustion"
            or bool(row.get("score", {}).get("context_exhaustion"))
            or is_context_exhaustion(exception)
            or any(
                is_context_exhaustion(item.get("raw_answer", "")) for item in server_requests
            )
        )
        infrastructure_error = (
            not context_exhaustion
            and (
                row.get("runtime_diagnosis", {}).get("kind") == "infrastructure_error"
                or bool(row.get("score", {}).get("infrastructure_error"))
            )
        )
        updated = dict(row)
        updated_score = score_hospital_calls_strict(
            hospital_scenario(row["case_id"]),
            row.get("tool_calls", []),
            row.get("attempted_tool_calls", []),
            infrastructure_error=infrastructure_error,
            context_exhaustion=context_exhaustion,
        )
        updated["score"] = updated_score
        updated["runtime_diagnosis"] = diagnose_timeout_or_runtime(
            exception,
            server_requests,
            row.get("attempted_tool_calls", []),
            row.get("tool_calls", []),
        )
        updated["failure_kind"] = classify_failure(updated_score, exception, server_requests)
        rescored.append(updated)

    passes = sum(bool(row["score"]["passed"]) for row in rescored)
    avg_score = sum(float(row["score"]["score"]) for row in rescored) / len(rescored)
    print(f"original: {old_passes}/{len(rows)} avg={old_avg:.3f}")
    print(f"strict:   {passes}/{len(rows)} avg={avg_score:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rescored, indent=2) + "\n", encoding="utf-8")
        write_summary(
            args.output.with_suffix(".md"),
            "Pydantic AI Hospital Logistics Arena (strict rescore)",
            rescored,
        )
        print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()