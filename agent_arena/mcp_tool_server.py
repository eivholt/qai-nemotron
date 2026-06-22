#!/usr/bin/env python3
"""Real MCP server implementations for the agent arena tools."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from agent_arena.tool_arena import ToolRuntime, safe_eval


def load_fastmcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing MCP dependency. Install with:\n"
            '  python -m pip install "pydantic-ai-slim[mcp]"\n'
            f"Original import error: {exc}"
        ) from exc
    return FastMCP


def parse_allowed(text: str) -> set[str] | None:
    if not text or text == "all":
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def append_log(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def logged_call(
    log_path: Path | None,
    name: str,
    args: dict[str, Any],
    fn: Callable[[], Any],
) -> Any:
    started = time.monotonic()
    try:
        result = fn()
        append_log(
            log_path,
            {
                "name": name,
                "args": args,
                "ok": True,
                "elapsed_s": round(time.monotonic() - started, 5),
                "result": result,
            },
        )
        return result
    except Exception as exc:
        error = str(exc)
        append_log(
            log_path,
            {
                "name": name,
                "args": args,
                "ok": False,
                "elapsed_s": round(time.monotonic() - started, 5),
                "error": error,
            },
        )
        return {"ok": False, "error": error}


def register_tools(mcp: Any, runtime: ToolRuntime, log_path: Path | None, allowed: set[str] | None) -> None:
    def include(name: str) -> bool:
        return allowed is None or name in allowed

    if include("lookup_json_record"):

        @mcp.tool()
        def lookup_json_record(path: str, array_path: str, key: str, value: str) -> Any:
            """Find one object in a JSON array by matching a key/value."""
            return logged_call(
                log_path,
                "lookup_json_record",
                {"path": path, "array_path": array_path, "key": key, "value": value},
                lambda: runtime.lookup_json_record(path, array_path, key, value),
            )

    if include("csv_max_difference"):

        @mcp.tool()
        def csv_max_difference(path: str, id_col: str, minuend_col: str, subtrahend_col: str) -> Any:
            """Find the row with the largest numeric minuend_col - subtrahend_col."""
            return logged_call(
                log_path,
                "csv_max_difference",
                {
                    "path": path,
                    "id_col": id_col,
                    "minuend_col": minuend_col,
                    "subtrahend_col": subtrahend_col,
                },
                lambda: runtime.csv_max_difference(path, id_col, minuend_col, subtrahend_col),
            )

    if include("list_files"):

        @mcp.tool()
        def list_files(path: str = ".") -> Any:
            """List fixture files below a relative directory."""
            return logged_call(log_path, "list_files", {"path": path}, lambda: runtime.list_files(path))

    if include("read_file"):

        @mcp.tool()
        def read_file(path: str, max_chars: int = 1200) -> Any:
            """Read a fixture file. Large outputs are truncated."""
            return logged_call(
                log_path,
                "read_file",
                {"path": path, "max_chars": max_chars},
                lambda: runtime.read_file(path, max_chars),
            )

    if include("search_text"):

        @mcp.tool()
        def search_text(path: str, pattern: str, max_matches: int = 5) -> Any:
            """Search a fixture text file with a regular expression."""
            return logged_call(
                log_path,
                "search_text",
                {"path": path, "pattern": pattern, "max_matches": max_matches},
                lambda: runtime.search_text(path, pattern, max_matches),
            )

    if include("json_query"):

        @mcp.tool()
        def json_query(path: str, query: str) -> Any:
            """Read a JSON file and return a dotted path."""
            return logged_call(
                log_path,
                "json_query",
                {"path": path, "query": query},
                lambda: runtime.json_query(path, query),
            )

    if include("http_get"):

        @mcp.tool()
        def http_get(url: str) -> Any:
            """Return a fixture HTTP response for an allowed URL."""
            return logged_call(log_path, "http_get", {"url": url}, lambda: runtime.http_get(url))

    if include("calculator"):

        @mcp.tool()
        def calculator(expression: str) -> Any:
            """Evaluate a small arithmetic expression."""
            return logged_call(
                log_path,
                "calculator",
                {"expression": expression},
                lambda: safe_eval(expression),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--obs-chars", type=int, default=1200)
    parser.add_argument("--allowed-tools", default="all")
    parser.add_argument("--tool-log", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    FastMCP = load_fastmcp()
    case = json.loads(args.case_file.read_text())
    runtime = ToolRuntime(args.root.resolve(), case, args.obs_chars)
    allowed = parse_allowed(args.allowed_tools)
    server = FastMCP(
        "agent-arena-tools",
        instructions=(
            "Use these fixture tools to inspect files, fixture HTTP responses, logs, "
            "CSV data, JSON data, and arithmetic. Do not guess values that can be retrieved."
        ),
    )
    register_tools(server, runtime, args.tool_log, allowed)
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
