#!/usr/bin/env python3
"""Real MCP server for OR-agent-style arena tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_arena.mcp_tool_server import load_fastmcp, parse_allowed
from agent_arena.or_agent_runtime import OR_AGENT_INSTRUCTION_STYLES, ORRuntime, or_mcp_instructions


def register_or_tools(mcp: Any, runtime: ORRuntime, allowed: set[str] | None) -> None:
    def include(name: str) -> bool:
        return allowed is None or name in allowed

    if include("get_case"):

        @mcp.tool()
        def get_case(case_id: str) -> dict[str, Any]:
            """Fetch the surgical case and required equipment list from the EMR.

            Call this FIRST to learn what instruments/supplies the procedure requires.
            """
            return runtime.log_call(
                "get_case",
                {"case_id": case_id},
                lambda: runtime.get_case(case_id),
            )

    if include("check_supplies"):

        @mcp.tool()
        def check_supplies() -> dict[str, Any]:
            """Compare detected instruments against the surgical case requirements.

            Call this AFTER get_case. all_present=false means there are supply
            deficits: set yellow and request_resupply for each deficit. Supply
            deficits are NOT sterile zone issues and must NOT create human_review
            tasks.
            """
            return runtime.log_call("check_supplies", {}, runtime.check_supplies)

    if include("inspect_scene"):

        @mcp.tool()
        def inspect_scene(image_path: str) -> dict[str, Any]:
            """Inspect the OR scene image for sterile zone violations.

            Call this whenever at least one instrument was detected. The image_path
            must come from the event. Use verdict=true to set red and create human_review.
            """
            return runtime.log_call(
                "inspect_scene",
                {"image_path": image_path},
                lambda: runtime.inspect_scene(image_path),
            )

    if include("create_task"):

        @mcp.tool()
        def create_task(
            case_id: str,
            task_type: str,
            priority: str,
            summary: str,
            reason: str,
        ) -> dict[str, Any]:
            """Create a workflow task for sterile zone issues only.

            Use ONLY when inspect_scene verdict is true. For supply deficits, use
            request_resupply instead.
            """
            return runtime.log_call(
                "create_task",
                {
                    "case_id": case_id,
                    "task_type": task_type,
                    "priority": priority,
                    "summary": summary,
                    "reason": reason,
                },
                lambda: runtime.create_task(case_id, task_type, priority, summary, reason),
            )

    if include("request_resupply"):

        @mcp.tool()
        def request_resupply(item_name: str, room_id: str, urgency: str) -> dict[str, Any]:
            """Request sterile processing delivery for a missing item."""
            return runtime.log_call(
                "request_resupply",
                {"item_name": item_name, "room_id": room_id, "urgency": urgency},
                lambda: runtime.request_resupply(item_name, room_id, urgency),
            )

    if include("set_stacklight"):

        @mcp.tool()
        def set_stacklight(room_id: str, color: str, reason: str) -> dict[str, Any]:
            """Set the OR prep status stacklight.

            Green = logistics-ready, Yellow = supply deficit, Red = sterile contamination.
            """
            return runtime.log_call(
                "set_stacklight",
                {"room_id": room_id, "color": color, "reason": reason},
                lambda: runtime.set_stacklight(room_id, color, reason),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--allowed-tools", default="all")
    parser.add_argument("--tool-log", type=Path)
    parser.add_argument("--tool-hints", action="store_true")
    parser.add_argument("--tool-guardrails", action="store_true")
    parser.add_argument("--instruction-style", choices=sorted(OR_AGENT_INSTRUCTION_STYLES), default="legacy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    FastMCP = load_fastmcp()
    arena_case = json.loads(args.case_file.read_text())
    runtime = ORRuntime(
        arena_case=arena_case,
        include_hints=args.tool_hints,
        enforce_guardrails=args.tool_guardrails,
        log_path=args.tool_log,
    )
    server = FastMCP(
        "or-agent-arena-tools",
        instructions=or_mcp_instructions(args.instruction_style),
    )
    register_or_tools(server, runtime, parse_allowed(args.allowed_tools))
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
