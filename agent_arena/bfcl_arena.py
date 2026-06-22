#!/usr/bin/env python3
"""BFCL-inspired function-call probes for OpenAI-compatible model endpoints.

This runner intentionally stays close to the benchmark shape NVIDIA reports for
Nemotron Nano: provide function schemas, ask a single user request, then score the
AST-like tool call rather than a long agent workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_arena.pydantic_mcp_arena import apply_env_defaults


BFCL_SYSTEM = (
    "You are a function-calling assistant. Use the supplied tools when they are "
    "needed. If no supplied tool is relevant, answer normally and do not call a tool."
)


def tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


BFCL_CASES: list[dict[str, Any]] = [
    {
        "id": "bfcl_01_simple_weather",
        "category": "simple",
        "description": "One relevant function, one required string argument.",
        "user": "What is the weather in Oslo? Use the available tool.",
        "tools": [
            tool(
                "get_weather",
                "Get the current weather for a city.",
                {"city": {"type": "string", "description": "City name."}},
                ["city"],
            )
        ],
        "expected_tool_calls": [{"name": "get_weather", "arguments": {"city": "Oslo"}}],
    },
    {
        "id": "bfcl_02_multiple_select_weather",
        "category": "multiple",
        "description": "Several functions are present; only weather is relevant.",
        "user": "Use the right tool to check the weather in Bergen.",
        "tools": [
            tool(
                "book_flight",
                "Book a flight between two airports.",
                {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                },
                ["origin", "destination"],
            ),
            tool(
                "get_weather",
                "Get the current weather for a city.",
                {"city": {"type": "string"}},
                ["city"],
            ),
            tool(
                "lookup_invoice",
                "Look up an invoice by invoice ID.",
                {"invoice_id": {"type": "string"}},
                ["invoice_id"],
            ),
        ],
        "expected_tool_calls": [{"name": "get_weather", "arguments": {"city": "Bergen"}}],
    },
    {
        "id": "bfcl_03_parallel_same_tool",
        "category": "parallel",
        "description": "Two independent invocations of the same function.",
        "user": "Check the weather for Oslo and Tokyo.",
        "tools": [
            tool(
                "get_weather",
                "Get the current weather for a city.",
                {"city": {"type": "string"}},
                ["city"],
            )
        ],
        "expected_tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Oslo"}},
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        ],
        "allow_orderless": True,
    },
    {
        "id": "bfcl_04_parallel_multiple_tools",
        "category": "parallel_multiple",
        "description": "Two independent calls to different functions.",
        "user": "Get the weather in Oslo and look up invoice INV-42.",
        "tools": [
            tool(
                "get_weather",
                "Get the current weather for a city.",
                {"city": {"type": "string"}},
                ["city"],
            ),
            tool(
                "lookup_invoice",
                "Look up an invoice by invoice ID.",
                {"invoice_id": {"type": "string"}},
                ["invoice_id"],
            ),
        ],
        "expected_tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Oslo"}},
            {"name": "lookup_invoice", "arguments": {"invoice_id": "INV-42"}},
        ],
        "allow_orderless": True,
    },
    {
        "id": "bfcl_05_relevance_no_tool",
        "category": "irrelevance",
        "description": "Available tools are irrelevant, so the model should abstain.",
        "user": "Say exactly: no-tool-needed",
        "tools": [
            tool(
                "get_weather",
                "Get the current weather for a city.",
                {"city": {"type": "string"}},
                ["city"],
            ),
            tool(
                "lookup_invoice",
                "Look up an invoice by invoice ID.",
                {"invoice_id": {"type": "string"}},
                ["invoice_id"],
            ),
        ],
        "expected_tool_calls": [],
        "expected_content_regex": [r"no-tool-needed"],
    },
    {
        "id": "bfcl_06_nested_object",
        "category": "nested",
        "description": "One function with nested object and enum-like fields.",
        "user": (
            "Create a maintenance ticket for device D-9 in room LAB-2. "
            "Priority is high, and the issue summary is: fan vibration."
        ),
        "tools": [
            tool(
                "create_ticket",
                "Create a maintenance ticket.",
                {
                    "device_id": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                    "details": {
                        "type": "object",
                        "properties": {
                            "room": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                        "required": ["room", "summary"],
                    },
                },
                ["device_id", "priority", "details"],
            )
        ],
        "expected_tool_calls": [
            {
                "name": "create_ticket",
                "arguments": {
                    "device_id": "D-9",
                    "priority": "high",
                    "details": {"room": "LAB-2", "summary": "fan vibration"},
                },
            }
        ],
    },
    {
        "id": "bfcl_07_relevance_some_tool",
        "category": "relevance",
        "description": "Exact arguments are underdetermined, but a relevant tool call should be made.",
        "user": "I need help with the invoice I just mentioned; use the invoice tool if it fits.",
        "tools": [
            tool(
                "lookup_invoice",
                "Look up an invoice by invoice ID.",
                {"invoice_id": {"type": "string"}},
                ["invoice_id"],
            ),
            tool(
                "get_weather",
                "Get the current weather for a city.",
                {"city": {"type": "string"}},
                ["city"],
            ),
        ],
        "expected_any_tool_names": ["lookup_invoice"],
    },
]


def http_post_json(url: str, payload: dict[str, Any], api_key: str, timeout_s: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def extract_tool_calls(response: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    choices = response.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return [], "", ""
    choice = choices[0]
    message = choice.get("message", {})
    if not isinstance(message, dict):
        return [], "", str(choice.get("finish_reason", ""))
    calls: list[dict[str, Any]] = []
    for idx, call in enumerate(message.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function", {})
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        calls.append(
            {
                "id": str(call.get("id") or f"call_{idx + 1}"),
                "name": name,
                "arguments": parse_arguments(function.get("arguments", {})),
            }
        )
    return calls, str(message.get("content") or ""), str(choice.get("finish_reason", ""))


def scalar_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().lower() == expected.strip().lower()
    return actual == expected


def contains_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, expected_value in expected.items():
            if key not in actual or not contains_subset(actual[key], expected_value):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) < len(expected):
            return False
        return all(any(contains_subset(candidate, item) for candidate in actual) for item in expected)
    return scalar_equal(actual, expected)


def call_signature(call: dict[str, Any]) -> str:
    return json.dumps(
        {"name": call.get("name"), "arguments": call.get("arguments", {})},
        sort_keys=True,
        separators=(",", ":"),
    )


def score_case(case: dict[str, Any], tool_calls: list[dict[str, Any]], content: str) -> dict[str, Any]:
    checks: list[tuple[str, bool]] = []
    expected_calls = case.get("expected_tool_calls")
    if expected_calls is not None:
        checks.append(("tool_call_count", len(tool_calls) == len(expected_calls)))
        if case.get("allow_orderless"):
            expected_names = Counter(call["name"] for call in expected_calls)
            actual_names = Counter(call["name"] for call in tool_calls)
            checks.append(("tool_names", actual_names == expected_names))
            for expected in expected_calls:
                checks.append(
                    (
                        f"{expected['name']}_arguments",
                        any(
                            call.get("name") == expected["name"]
                            and contains_subset(call.get("arguments", {}), expected.get("arguments", {}))
                            for call in tool_calls
                        ),
                    )
                )
        else:
            for idx, expected in enumerate(expected_calls):
                actual = tool_calls[idx] if idx < len(tool_calls) else {}
                checks.append((f"call_{idx + 1}_name", actual.get("name") == expected["name"]))
                checks.append(
                    (
                        f"call_{idx + 1}_arguments",
                        contains_subset(actual.get("arguments", {}), expected.get("arguments", {})),
                    )
                )

    expected_any_tool_names = case.get("expected_any_tool_names")
    if expected_any_tool_names:
        checks.append(
            (
                "relevant_tool_called",
                any(call.get("name") in set(expected_any_tool_names) for call in tool_calls),
            )
        )

    for idx, pattern in enumerate(case.get("expected_content_regex", []), start=1):
        checks.append((f"content_regex_{idx}", bool(re.search(pattern, content, flags=re.I | re.S))))

    passed = [name for name, ok in checks if ok]
    missing = [name for name, ok in checks if not ok]
    score = len(passed) / len(checks) if checks else 1.0
    return {
        "passed": not missing,
        "score": round(score, 3),
        "missing": missing,
        "tool_signatures": [call_signature(call) for call in tool_calls],
    }


def write_bfcl_summary(path: Path, title: str, results: list[dict[str, Any]]) -> None:
    passed = sum(1 for result in results if result["score"]["passed"])
    avg = sum(result["score"]["score"] for result in results) / len(results) if results else 0.0
    lines = [
        f"# {title}",
        "",
        f"Cases: {len(results)}",
        f"Passed: {passed}",
        f"Average score: {avg:.3f}",
        "",
        "| category | cases | pass | avg score |",
        "|---|---:|---:|---:|",
    ]
    categories = sorted({result["category"] for result in results})
    for category in categories:
        items = [result for result in results if result["category"] == category]
        cat_passed = sum(1 for item in items if item["score"]["passed"])
        cat_avg = sum(item["score"]["score"] for item in items) / len(items)
        lines.append(f"| {category} | {len(items)} | {cat_passed} | {cat_avg:.3f} |")
    lines.extend(
        [
            "",
            "| case | category | score | pass | finish | failure | parsed calls |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for result in results:
        calls = "<br>".join(result["score"].get("tool_signatures", [])) or "none"
        missing = ", ".join(result["score"].get("missing", []))
        lines.append(
            f"| {result['case_id']} | {result['category']} | "
            f"{result['score']['score']:.3f} | "
            f"{'yes' if result['score']['passed'] else 'no'} | "
            f"{result.get('finish_reason', '')} | {missing} | {calls} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("agent_arena/.env"))
    parser.add_argument("--provider", choices=["openai-compatible", "azure", "azure-foundry-anthropic", "anthropic"])
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--azure-endpoint")
    parser.add_argument("--azure-api-version")
    parser.add_argument("--anthropic-foundry-base-url")
    parser.add_argument("--model-name")
    parser.add_argument("--model-label")
    parser.add_argument("--mode", default="stock")
    parser.add_argument("--case-ids")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--out-root", type=Path, default=Path("agent_arena_results"))
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def select_cases(case_ids: str | None) -> list[dict[str, Any]]:
    if not case_ids:
        return list(BFCL_CASES)
    wanted = {item.strip() for item in case_ids.split(",") if item.strip()}
    return [case for case in BFCL_CASES if case["id"] in wanted]


def run_case(args: argparse.Namespace, case: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": BFCL_SYSTEM},
        {"role": "user", "content": case["user"]},
    ]
    payload = {
        "model": args.model_name,
        "messages": messages,
        "tools": case["tools"],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    started = time.monotonic()
    response: dict[str, Any] = {}
    exception = ""
    try:
        response = http_post_json(
            args.base_url.rstrip("/") + "/chat/completions",
            payload,
            args.api_key or "agent-arena",
            args.timeout_s,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        exception = repr(exc)
    elapsed_s = time.monotonic() - started
    tool_calls, content, finish_reason = extract_tool_calls(response)
    score = score_case(case, tool_calls, content) if not exception else {
        "passed": False,
        "score": 0.0,
        "missing": ["request_failed"],
        "tool_signatures": [],
    }
    record = {
        "model": args.model_label,
        "mode": args.mode,
        "client": "bfcl_ast_probe",
        "case_id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "messages": messages,
        "tools": case["tools"],
        "response": response,
        "content": content,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls,
        "score": score,
        "exception": exception,
        "elapsed_s": round(elapsed_s, 3),
    }
    (result_dir / f"{args.model_label}__{args.mode}__{case['id']}.json").write_text(
        json.dumps(record, indent=2)
    )
    return record


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for case in BFCL_CASES:
            print(f"{case['id']}\t{case['category']}\t{case['description']}")
        return 0
    apply_env_defaults(args)
    if args.provider != "openai-compatible":
        raise SystemExit("bfcl_arena currently targets OpenAI-compatible /v1/chat/completions endpoints.")
    args.api_key = args.api_key or os.environ.get("AGENT_ARENA_API_KEY", "agent-arena")
    cases = select_cases(args.case_ids)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = (
        args.out_root.expanduser()
        / f"{timestamp}__bfcl_arena__{args.model_label}__{args.mode}__openai-compatible"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    results = [run_case(args, case, result_dir) for case in cases]
    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_bfcl_summary(result_dir / "summary.md", "BFCL-Inspired Function Calling Arena", results)
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
