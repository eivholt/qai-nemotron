#!/usr/bin/env python3
"""Run the tool arena through a real Pydantic AI agent client."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_arena.model_client import score_text, write_summary
from agent_arena.tool_arena import TOOL_CASES, ToolRuntime, safe_eval


def require_pydantic_ai() -> tuple[Any, Any, Any]:
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Pydantic AI dependencies. Install on the host with:\n"
            '  python -m pip install "pydantic-ai-slim[openai]"\n'
            f"Original import error: {exc}"
        ) from exc
    return Agent, OpenAIChatModel, OpenAIProvider


def root_from_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def http_json(url: str, method: str = "GET") -> dict[str, Any]:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def reset_server_log(base_url: str) -> None:
    http_json(root_from_base_url(base_url) + "/debug/reset", method="POST")


def get_server_log(base_url: str) -> list[dict[str, Any]]:
    payload = http_json(root_from_base_url(base_url) + "/debug/requests")
    requests = payload.get("requests", [])
    return requests if isinstance(requests, list) else []


def call_tool(tool_log: list[dict[str, Any]], name: str, fn: Callable[[], Any], args: dict[str, Any]) -> Any:
    started = time.monotonic()
    try:
        result = fn()
        record = {
            "name": name,
            "args": args,
            "ok": True,
            "elapsed_s": round(time.monotonic() - started, 4),
            "result": result,
        }
        tool_log.append(record)
        return result
    except Exception as exc:
        record = {
            "name": name,
            "args": args,
            "ok": False,
            "elapsed_s": round(time.monotonic() - started, 4),
            "error": str(exc),
        }
        tool_log.append(record)
        return {"ok": False, "error": str(exc)}


def build_tools(runtime: ToolRuntime, tool_log: list[dict[str, Any]]) -> list[Callable[..., Any]]:
    def lookup_json_record(path: str, array_path: str, key: str, value: str) -> Any:
        """Find one object in a JSON array by matching a key/value."""
        return call_tool(
            tool_log,
            "lookup_json_record",
            lambda: runtime.lookup_json_record(path, array_path, key, value),
            {"path": path, "array_path": array_path, "key": key, "value": value},
        )

    def csv_max_difference(path: str, id_col: str, minuend_col: str, subtrahend_col: str) -> Any:
        """Find the row with the largest numeric minuend_col - subtrahend_col."""
        return call_tool(
            tool_log,
            "csv_max_difference",
            lambda: runtime.csv_max_difference(path, id_col, minuend_col, subtrahend_col),
            {
                "path": path,
                "id_col": id_col,
                "minuend_col": minuend_col,
                "subtrahend_col": subtrahend_col,
            },
        )

    def list_files(path: str = ".") -> Any:
        """List fixture files below a relative directory."""
        return call_tool(tool_log, "list_files", lambda: runtime.list_files(path), {"path": path})

    def read_file(path: str, max_chars: int = 1200) -> Any:
        """Read a fixture file. Large outputs are truncated."""
        return call_tool(
            tool_log,
            "read_file",
            lambda: runtime.read_file(path, max_chars),
            {"path": path, "max_chars": max_chars},
        )

    def search_text(path: str, pattern: str, max_matches: int = 5) -> Any:
        """Search a fixture text file with a regular expression."""
        return call_tool(
            tool_log,
            "search_text",
            lambda: runtime.search_text(path, pattern, max_matches),
            {"path": path, "pattern": pattern, "max_matches": max_matches},
        )

    def json_query(path: str, query: str) -> Any:
        """Read a JSON file and return a dotted path."""
        return call_tool(
            tool_log,
            "json_query",
            lambda: runtime.json_query(path, query),
            {"path": path, "query": query},
        )

    def http_get(url: str) -> Any:
        """Return a fixture HTTP response for an allowed URL."""
        return call_tool(tool_log, "http_get", lambda: runtime.http_get(url), {"url": url})

    def calculator(expression: str) -> Any:
        """Evaluate a small arithmetic expression."""
        return call_tool(
            tool_log,
            "calculator",
            lambda: safe_eval(expression),
            {"expression": expression},
        )

    return [
        lookup_json_record,
        csv_max_difference,
        list_files,
        read_file,
        search_text,
        json_query,
        http_get,
        calculator,
    ]


def summarize_messages(result: Any) -> list[str]:
    try:
        return [repr(message) for message in result.all_messages()]
    except Exception:
        return []


def classify_failure(score: dict[str, Any], exception: str, server_requests: list[dict[str, Any]]) -> str:
    if score["passed"]:
        return "passed"
    if exception:
        return "agent_exception"
    if any(item.get("timed_out") for item in server_requests):
        return "model_timeout"
    if any(not item.get("parsed_ok") for item in server_requests):
        return "protocol_error"
    return "wrong_result"


def run_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    model: Any,
    Agent: Any,
    result_dir: Path,
) -> dict[str, Any]:
    reset_server_log(args.base_url)
    tool_log: list[dict[str, Any]] = []
    output = ""
    exception = ""
    messages: list[str] = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"pydantic__{case['id']}__") as temp:
        runtime = ToolRuntime(Path(temp), case, args.obs_chars)
        tools = build_tools(runtime, tool_log)
        agent = Agent(
            model,
            instructions=(
                "You are a practical Linux/HTTP/data agent. Use tools when needed. "
                "Do not guess values that should come from files, HTTP fixtures, or logs. "
                "Return only the final answer in the format requested by the user."
            ),
            tools=tools,
            retries=args.agent_retries,
        )
        try:
            result = agent.run_sync(case["goal"])
            output = str(result.output)
            messages = summarize_messages(result)
        except Exception as exc:
            exception = repr(exc)
    elapsed_s = time.monotonic() - started
    server_requests = get_server_log(args.base_url)
    score = score_text(output, case["required_regex"])
    record = {
        "model": args.model_label,
        "mode": args.mode,
        "client": "pydantic_ai",
        "case_id": case["id"],
        "difficulty": case["difficulty"],
        "output": output,
        "score": score,
        "failure_kind": classify_failure(score, exception, server_requests),
        "exception": exception,
        "elapsed_s": round(elapsed_s, 3),
        "tool_calls": tool_log,
        "server_requests": server_requests,
        "messages": messages,
        "notes": (
            f"model_requests={len(server_requests)},"
            f"tool_calls={len(tool_log)},"
            f"parse_failures={sum(1 for item in server_requests if not item.get('parsed_ok'))}"
        ),
    }
    (result_dir / f"{args.model_label}__{args.mode}__{case['id']}.json").write_text(
        json.dumps(record, indent=2)
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://evk:8001/v1")
    parser.add_argument("--api-key", default="agent-arena")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--mode", choices=["stock", "thinking_off", "thinking_on"], default="thinking_off")
    parser.add_argument("--case-ids")
    parser.add_argument("--agent-retries", type=int, default=0)
    parser.add_argument("--obs-chars", type=int, default=1200)
    parser.add_argument("--out-root", type=Path, default=Path.home() / "agent_arena_results")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for case in TOOL_CASES:
            print(f"{case['id']}\tdifficulty={case['difficulty']}")
        return 0
    Agent, OpenAIChatModel, OpenAIProvider = require_pydantic_ai()
    wanted = {item.strip() for item in args.case_ids.split(",")} if args.case_ids else None
    cases = [case for case in TOOL_CASES if wanted is None or case["id"] in wanted]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.out_root.expanduser() / f"{timestamp}__pydantic_arena__{args.model_label}__{args.mode}"
    result_dir.mkdir(parents=True, exist_ok=True)
    model = OpenAIChatModel(
        args.model_name,
        provider=OpenAIProvider(base_url=args.base_url, api_key=args.api_key),
    )
    results = [run_case(args, case, model, Agent, result_dir) for case in cases]
    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(result_dir / "summary.md", "Pydantic AI Agent Arena", results)
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
