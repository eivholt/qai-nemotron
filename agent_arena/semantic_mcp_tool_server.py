#!/usr/bin/env python3
"""Real MCP server for semantic edge-agent arena tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_arena.mcp_tool_server import load_fastmcp, parse_allowed
from agent_arena.semantic_runtime import SemanticRuntime, logged_semantic_call


def register_semantic_tools(
    mcp: Any,
    runtime: SemanticRuntime,
    log_path: Path | None,
    allowed: set[str] | None,
) -> None:
    def include(name: str) -> bool:
        return allowed is None or name in allowed

    if include("lookup_ticket_owner"):

        @mcp.tool()
        def lookup_ticket_owner(ticket_id: str) -> dict[str, Any]:
            """Return the owner for a ticket id from the fixture ticket data."""
            return logged_semantic_call(
                None,
                log_path,
                "lookup_ticket_owner",
                {"ticket_id": ticket_id},
                lambda: runtime.lookup_ticket_owner(ticket_id),
            )

    if include("find_largest_restock"):

        @mcp.tool()
        def find_largest_restock() -> dict[str, Any]:
            """Return the SKU with the largest target - on_hand restock shortfall."""
            return logged_semantic_call(
                None,
                log_path,
                "find_largest_restock",
                {},
                runtime.find_largest_restock,
            )

    if include("get_device_status"):

        @mcp.tool()
        def get_device_status(device_id: str) -> dict[str, Any]:
            """Return fixture HTTP health fields for a device id such as D-9."""
            return logged_semantic_call(
                None,
                log_path,
                "get_device_status",
                {"device_id": device_id},
                lambda: runtime.get_device_status(device_id),
            )

    if include("find_log_error_code"):

        @mcp.tool()
        def find_log_error_code(request_id: str) -> dict[str, Any]:
            """Return the error code for a request id found in the fixture logs."""
            return logged_semantic_call(
                None,
                log_path,
                "find_log_error_code",
                {"request_id": request_id},
                lambda: runtime.find_log_error_code(request_id),
            )

    if include("calculate"):

        @mcp.tool()
        def calculate(expression: str) -> dict[str, Any]:
            """Evaluate a small arithmetic expression and return the numeric result."""
            return logged_semantic_call(
                None,
                log_path,
                "calculate",
                {"expression": expression},
                lambda: runtime.calculate(expression),
            )

    if include("get_config_value"):

        @mcp.tool()
        def get_config_value(key: str) -> dict[str, Any]:
            """Read a key=value field from the simple fixture text file."""
            return logged_semantic_call(
                None,
                log_path,
                "get_config_value",
                {"key": key},
                lambda: runtime.get_config_value(key),
            )

    if include("get_profile_field"):

        @mcp.tool()
        def get_profile_field(field: str) -> dict[str, Any]:
            """Read one top-level field from profile.json."""
            return logged_semantic_call(
                None,
                log_path,
                "get_profile_field",
                {"field": field},
                lambda: runtime.get_profile_field(field),
            )

    if include("get_ping_status"):

        @mcp.tool()
        def get_ping_status() -> dict[str, Any]:
            """Fetch the fixture ping endpoint and return its status field."""
            return logged_semantic_call(
                None,
                log_path,
                "get_ping_status",
                {},
                runtime.get_ping_status,
            )

    if include("list_first_fixture_file"):

        @mcp.tool()
        def list_first_fixture_file() -> dict[str, Any]:
            """Return the first fixture filename in sorted order."""
            return logged_semantic_call(
                None,
                log_path,
                "list_first_fixture_file",
                {},
                runtime.list_first_fixture_file,
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
    runtime = SemanticRuntime(args.root.resolve(), case, args.obs_chars)
    allowed = parse_allowed(args.allowed_tools)
    server = FastMCP(
        "semantic-agent-arena-tools",
        instructions=(
            "Use these semantic fixture tools to retrieve reliable facts for "
            "the user's Linux/HTTP/data task. Prefer the task-shaped tool over "
            "guessing or reconstructing raw file operations."
        ),
    )
    register_semantic_tools(server, runtime, args.tool_log, allowed)
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
