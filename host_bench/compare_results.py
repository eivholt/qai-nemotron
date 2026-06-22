#!/usr/bin/env python3
"""Aggregate host-side HF benchmark result folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    results = []
    for path in sorted(root.glob("*/results.json")):
        results.extend(json.loads(path.read_text()))

    lines = [
        "# Host HF Full Comparison",
        "",
        f"Root: `{root}`",
        "",
        "| model | mode | cases | pass | avg score | open think | avg final chars | avg decode tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    grouped: dict[tuple[str, str], list[dict]] = {}
    for item in results:
        grouped.setdefault((item["model"], item["mode"]), []).append(item)

    for (model, mode), items in sorted(grouped.items()):
        scores = [item["score"].get("score", 0) for item in items]
        decode = [
            item.get("profile", {}).get("token-generation-rate")
            for item in items
            if item.get("profile", {}).get("token-generation-rate") is not None
        ]
        final_chars = [len(item.get("answer", "")) for item in items]
        lines.append(
            f"| {model} | {mode} | {len(items)} | "
            f"{sum(1 for item in items if item['score'].get('passed'))} | "
            f"{avg(scores):.3f} | "
            f"{sum(1 for item in items if item.get('reasoning_open'))} | "
            f"{avg(final_chars):.0f} | {avg(decode):.2f} |"
        )

    lines += [
        "",
        "## By Category",
        "",
        "| model | mode | category | cases | pass | avg score |",
        "|---|---|---|---:|---:|---:|",
    ]
    by_category: dict[tuple[str, str, str], list[dict]] = {}
    for item in results:
        by_category.setdefault(
            (item["model"], item["mode"], item["category"]), []
        ).append(item)
    for (model, mode, category), items in sorted(by_category.items()):
        scores = [item["score"].get("score", 0) for item in items]
        lines.append(
            f"| {model} | {mode} | {category} | {len(items)} | "
            f"{sum(1 for item in items if item['score'].get('passed'))} | "
            f"{avg(scores):.3f} |"
        )

    lines += [
        "",
        "## Case Detail",
        "",
        "| case | category | hf nemotron off | hf nemotron on | hf stock |",
        "|---|---|---:|---:|---:|",
    ]
    by_case: dict[tuple[str, str], dict[tuple[str, str], dict]] = {}
    for item in results:
        by_case.setdefault((item["case_id"], item["category"]), {})[
            (item["model"], item["mode"])
        ] = item
    model_keys = [
        ("hf_nemotron_bf16_off512", "thinking_off"),
        ("hf_nemotron_bf16_on2048", "thinking_on"),
        ("hf_stock_llama31_bf16_512", "stock"),
    ]
    for (case_id, category), model_results in sorted(by_case.items()):
        cells = []
        for key in model_keys:
            item = model_results.get(key)
            if item is None:
                cells.append("")
            else:
                score = item["score"].get("score", 0)
                status = "yes" if item["score"].get("passed") else "no"
                cells.append(f"{score:.3f} {status}")
        lines.append(f"| {case_id} | {category} | {cells[0]} | {cells[1]} | {cells[2]} |")

    output = root / "comparison.md"
    output.write_text("\n".join(lines) + "\n")
    print(output.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
