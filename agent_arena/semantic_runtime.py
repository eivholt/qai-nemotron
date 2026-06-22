#!/usr/bin/env python3
"""Semantic tools for edge-realistic agent arena runs.

These tools deliberately expose task-shaped operations instead of low-level
file/HTTP/search primitives. This mirrors the OR-agent style: deterministic
client code gathers reliable facts, while the model chooses and sequences a
small number of meaningful operations.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

from agent_arena.mcp_tool_server import append_log
from agent_arena.tool_arena import ToolRuntime, safe_eval


SEMANTIC_TOOLSETS: dict[str, list[str]] = {
    "tool_00_direct_final_protocol": [],
    "tool_01_single_json_lookup": ["lookup_ticket_owner"],
    "tool_02_multi_step_inventory": ["find_largest_restock"],
    "tool_03_fixture_http": ["get_device_status"],
    "tool_04_context_breaking_log": ["find_log_error_code"],
    "tool_05_calculator_then_final": ["calculate"],
    "tool_06_small_file_read": ["get_config_value"],
    "tool_07_easy_calculator_add": ["calculate"],
    "tool_08_easy_read_color": ["get_config_value"],
    "tool_09_easy_json_query_name": ["get_profile_field"],
    "tool_10_easy_http_ping": ["get_ping_status"],
    "tool_11_easy_short_search": ["find_log_error_code"],
    "tool_12_easy_list_files": ["list_first_fixture_file"],
}


ALL_SEMANTIC_TOOLS = [
    "lookup_ticket_owner",
    "find_largest_restock",
    "get_device_status",
    "find_log_error_code",
    "calculate",
    "get_config_value",
    "get_profile_field",
    "get_ping_status",
    "list_first_fixture_file",
]


SEMANTIC_AGENT_INSTRUCTIONS = """\
You are an edge Linux/HTTP/data agent for a constrained device.

WORKFLOW:
1. Read the user's task.
2. If a semantic tool matches the task, call exactly that tool first.
3. Use the returned fields to answer in the exact format requested by the user.
4. If no tool is needed, answer directly.

RULES:
- Do not guess values that a tool can retrieve.
- Do not answer with placeholders like <value> or <name>.
- Keep the final response to one short line.
- The final response must match the user's requested key=value format.
"""


def semantic_tools_for_case(case: dict[str, Any], pruning: str) -> list[str]:
    if pruning == "none":
        return list(ALL_SEMANTIC_TOOLS)
    return SEMANTIC_TOOLSETS.get(case["id"], [])


def semantic_hint_for_case(case: dict[str, Any]) -> str:
    tools = ", ".join(SEMANTIC_TOOLSETS.get(case["id"], [])) or "none"
    return (
        f"Case id: {case['id']}\n"
        f"Relevant semantic tools: {tools}\n"
        "Use the user's requested output format exactly, but do not guess hidden fixture values.\n"
    )


def logged_semantic_call(
    log: list[dict[str, Any]] | None,
    log_path: Path | None,
    name: str,
    args: dict[str, Any],
    fn: Callable[[], Any],
) -> Any:
    started = time.monotonic()
    try:
        result = fn()
        record = {
            "name": name,
            "args": args,
            "ok": True,
            "elapsed_s": round(time.monotonic() - started, 5),
            "result": result,
        }
    except Exception as exc:
        record = {
            "name": name,
            "args": args,
            "ok": False,
            "elapsed_s": round(time.monotonic() - started, 5),
            "error": str(exc),
        }
        result = {"ok": False, "error": str(exc)}
    if log is not None:
        log.append(record)
    append_log(log_path, record)
    return result


class SemanticRuntime:
    def __init__(self, root: Path, case: dict[str, Any], obs_chars: int) -> None:
        self.low_level = ToolRuntime(root, case, obs_chars)
        self.case = case

    def lookup_ticket_owner(self, ticket_id: str) -> dict[str, Any]:
        """Return the owner for a ticket id from the fixture ticket data."""
        record = self.low_level.lookup_json_record("tickets.json", "tickets", "id", ticket_id)
        if not record:
            return {"ok": False, "ticket_id": ticket_id, "error": "ticket not found"}
        return {"ok": True, "ticket_id": ticket_id, "owner": record.get("owner")}

    def find_largest_restock(self) -> dict[str, Any]:
        """Return the SKU with the largest target - on_hand restock shortfall."""
        result = self.low_level.csv_max_difference(
            "inventory.csv",
            "sku",
            "target",
            "on_hand",
        )
        return {"ok": True, "sku": result["id"], "restock": result["difference"]}

    def get_device_status(self, device_id: str) -> dict[str, Any]:
        """Return fixture HTTP health fields for a device id such as D-9."""
        response = self.low_level.http_get(f"https://arena.local/api/device/{device_id}")
        data = response.get("json", {})
        return {
            "ok": response.get("status") == 200,
            "device_id": device_id,
            "health": data.get("health"),
            "temp_c": data.get("temp_c"),
        }

    def find_log_error_code(self, request_id: str) -> dict[str, Any]:
        """Return the error code for a request id found in the fixture logs."""
        path = "tiny.log" if self.case["id"] == "tool_11_easy_short_search" else "app.log"
        matches = self.low_level.search_text(path, request_id, 3)
        for line in matches:
            code = re.search(r"code=([A-Za-z0-9_-]+)", line)
            if code:
                return {"ok": True, "request_id": request_id, "code": code.group(1), "line": line}
        return {"ok": False, "request_id": request_id, "matches": matches, "error": "code not found"}

    def calculate(self, expression: str) -> dict[str, Any]:
        """Evaluate a small arithmetic expression and return the numeric result."""
        result = safe_eval(expression)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return {"ok": True, "expression": expression, "result": result}

    def get_config_value(self, key: str) -> dict[str, Any]:
        """Read a key=value field from the simple fixture text file."""
        path = "color.txt" if self.case["id"] == "tool_08_easy_read_color" else "config.txt"
        text = self.low_level.read_file(path, 400)["text"]
        for line in text.splitlines():
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return {"ok": True, "key": key, "value": value.strip()}
        return {"ok": False, "key": key, "error": "key not found"}

    def get_profile_field(self, field: str) -> dict[str, Any]:
        """Read one top-level field from profile.json."""
        value = self.low_level.json_query("profile.json", field)
        return {"ok": True, "field": field, "value": value}

    def get_ping_status(self) -> dict[str, Any]:
        """Fetch the fixture ping endpoint and return its status field."""
        response = self.low_level.http_get("https://arena.local/api/ping")
        data = response.get("json", {})
        return {"ok": response.get("status") == 200, "status": data.get("status")}

    def list_first_fixture_file(self) -> dict[str, Any]:
        """Return the first fixture filename in sorted order."""
        files = self.low_level.list_files(".")
        return {"ok": bool(files), "file": files[0] if files else "", "files": files}
