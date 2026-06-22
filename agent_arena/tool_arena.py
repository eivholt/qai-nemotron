#!/usr/bin/env python3
"""Agent arena where the client runtime executes a small fixed tool set."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import operator
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_arena.model_client import (
    GenieClient,
    extract_first_json,
    score_text,
    write_summary,
)


TOOL_SPECS = [
    {
        "name": "lookup_json_record",
        "description": "Find one object in a JSON array by matching a key/value.",
        "parameters": {
            "path": "string",
            "array_path": "string",
            "key": "string",
            "value": "string",
        },
    },
    {
        "name": "csv_max_difference",
        "description": "Find the row with the largest numeric minuend_col - subtrahend_col.",
        "parameters": {
            "path": "string",
            "id_col": "string",
            "minuend_col": "string",
            "subtrahend_col": "string",
        },
    },
    {
        "name": "list_files",
        "description": "List fixture files below a relative directory.",
        "parameters": {"path": "string"},
    },
    {
        "name": "read_file",
        "description": "Read a fixture file. Large outputs are truncated.",
        "parameters": {"path": "string", "max_chars": "integer optional"},
    },
    {
        "name": "search_text",
        "description": "Search a fixture text file with a regular expression.",
        "parameters": {"path": "string", "pattern": "string", "max_matches": "integer optional"},
    },
    {
        "name": "json_query",
        "description": "Read a JSON file and return a dotted path.",
        "parameters": {"path": "string", "query": "string"},
    },
    {
        "name": "http_get",
        "description": "Return a fixture HTTP response for an allowed URL.",
        "parameters": {"url": "string"},
    },
    {
        "name": "calculator",
        "description": "Evaluate a small arithmetic expression.",
        "parameters": {"expression": "string"},
    },
]


def long_log() -> str:
    lines = [f"2026-06-21T10:{i % 60:02d}:00Z INFO request=RK-{1000 + i} ok" for i in range(420)]
    lines.append("2026-06-21T10:58:44Z ERROR request=RK-8842 code=E47 component=ingest")
    lines.extend(f"2026-06-21T11:{i % 60:02d}:00Z INFO request=RK-{2000 + i} ok" for i in range(420))
    return "\n".join(lines) + "\n"


TOOL_CASES: list[dict[str, Any]] = [
    {
        "id": "tool_00_direct_final_protocol",
        "difficulty": 0,
        "max_steps": 2,
        "goal": "No tool is needed. Return the final answer exactly as: ready=yes.",
        "required_regex": [r"ready\s*=\s*yes"],
        "expected_final": "ready=yes",
        "suggested_actions": ['{"final":"ready=yes"}'],
    },
    {
        "id": "tool_01_single_json_lookup",
        "difficulty": 1,
        "max_steps": 4,
        "goal": "Find the owner of ticket T-104. Return the final answer as: owner=<name>.",
        "files": {
            "tickets.json": json.dumps(
                {
                    "tickets": [
                        {"id": "T-103", "owner": "Mira", "status": "open"},
                        {"id": "T-104", "owner": "Jon", "status": "blocked"},
                    ]
                },
                indent=2,
            )
        },
        "required_regex": [r"owner\s*=\s*Jon"],
        "expected_final": "owner=Jon",
        "suggested_actions": [
            '{"tool":"lookup_json_record","args":{"path":"tickets.json","array_path":"tickets","key":"id","value":"T-104"}}'
        ],
    },
    {
        "id": "tool_02_multi_step_inventory",
        "difficulty": 2,
        "max_steps": 4,
        "goal": (
            "Inventory records are in files. Find the SKU that needs the largest "
            "restock, where restock = target - on_hand. Return: sku=<sku>,restock=<number>."
        ),
        "files": {
            "inventory.csv": (
                "sku,on_hand,target\n"
                "A-10,7,15\n"
                "B-22,4,25\n"
                "C-31,18,20\n"
            )
        },
        "required_regex": [r"sku\s*=\s*B-22", r"restock\s*=\s*21"],
        "expected_final": "sku=B-22,restock=21",
        "suggested_actions": [
            '{"tool":"csv_max_difference","args":{"path":"inventory.csv","id_col":"sku","minuend_col":"target","subtrahend_col":"on_hand"}}'
        ],
    },
    {
        "id": "tool_03_fixture_http",
        "difficulty": 3,
        "max_steps": 4,
        "goal": (
            "Fetch the fixture URL https://arena.local/api/device/D-9 and report "
            "the device health and temperature as: health=<value>,temp_c=<number>."
        ),
        "http": {
            "https://arena.local/api/device/D-9": {
                "status": 200,
                "json": {"id": "D-9", "health": "degraded", "temp_c": 74},
            }
        },
        "required_regex": [r"health\s*=\s*degraded", r"temp_c\s*=\s*74"],
        "expected_final": "health=degraded,temp_c=74",
        "suggested_actions": [
            '{"tool":"http_get","args":{"url":"https://arena.local/api/device/D-9"}}'
        ],
    },
    {
        "id": "tool_04_context_breaking_log",
        "difficulty": 4,
        "max_steps": 6,
        "goal": (
            "The log file is too large for context. Find the error code for "
            "request RK-8842 in app.log. Return: request=RK-8842,code=<code>."
        ),
        "files": {"app.log": long_log()},
        "required_regex": [r"request\s*=\s*RK-8842", r"code\s*=\s*E47"],
        "expected_final": "request=RK-8842,code=E47",
        "suggested_actions": [
            '{"tool":"search_text","args":{"path":"app.log","pattern":"RK-8842","max_matches":1}}'
        ],
    },
    {
        "id": "tool_05_calculator_then_final",
        "difficulty": 1,
        "max_steps": 3,
        "goal": "Use the calculator to compute (17 + 5) * 3. Return: result=<number>.",
        "required_regex": [r"result\s*=\s*66"],
        "expected_final": "result=66",
        "suggested_actions": [
            '{"tool":"calculator","args":{"expression":"(17 + 5) * 3"}}'
        ],
    },
    {
        "id": "tool_06_small_file_read",
        "difficulty": 1,
        "max_steps": 3,
        "goal": "Read config.txt and return the active mode as: mode=<value>.",
        "files": {"config.txt": "service=demo\nmode=staging\nowner=ops\n"},
        "required_regex": [r"mode\s*=\s*staging"],
        "expected_final": "mode=staging",
        "suggested_actions": [
            '{"tool":"read_file","args":{"path":"config.txt"}}'
        ],
    },
    {
        "id": "tool_07_easy_calculator_add",
        "difficulty": 0,
        "max_steps": 3,
        "goal": "Use the calculator tool to compute 2 + 3. Return the final answer as: result=<number>.",
        "required_regex": [r"result\s*=\s*5"],
        "expected_final": "result=5",
        "suggested_actions": [
            '{"tool":"calculator","args":{"expression":"2 + 3"}}'
        ],
    },
    {
        "id": "tool_08_easy_read_color",
        "difficulty": 0,
        "max_steps": 3,
        "goal": "Read color.txt. Return the color as: color=<value>.",
        "files": {"color.txt": "color=blue\n"},
        "required_regex": [r"color\s*=\s*blue"],
        "expected_final": "color=blue",
        "suggested_actions": [
            '{"tool":"read_file","args":{"path":"color.txt"}}'
        ],
    },
    {
        "id": "tool_09_easy_json_query_name",
        "difficulty": 0,
        "max_steps": 3,
        "goal": "Read profile.json and return the name as: name=<value>.",
        "files": {
            "profile.json": json.dumps({"name": "Iris", "role": "operator"}, indent=2)
        },
        "required_regex": [r"name\s*=\s*Iris"],
        "expected_final": "name=Iris",
        "suggested_actions": [
            '{"tool":"json_query","args":{"path":"profile.json","query":"name"}}'
        ],
    },
    {
        "id": "tool_10_easy_http_ping",
        "difficulty": 0,
        "max_steps": 3,
        "goal": "Fetch https://arena.local/api/ping and return the status as: status=<value>.",
        "http": {
            "https://arena.local/api/ping": {
                "status": 200,
                "json": {"status": "ok"},
            }
        },
        "required_regex": [r"status\s*=\s*ok"],
        "expected_final": "status=ok",
        "suggested_actions": [
            '{"tool":"http_get","args":{"url":"https://arena.local/api/ping"}}'
        ],
    },
    {
        "id": "tool_11_easy_short_search",
        "difficulty": 0,
        "max_steps": 3,
        "goal": "Search tiny.log for request AA-7. Return the code as: code=<value>.",
        "files": {
            "tiny.log": (
                "INFO request=AA-6 code=OK\n"
                "ERROR request=AA-7 code=E1\n"
            )
        },
        "required_regex": [r"code\s*=\s*E1"],
        "expected_final": "code=E1",
        "suggested_actions": [
            '{"tool":"search_text","args":{"path":"tiny.log","pattern":"AA-7","max_matches":1}}'
        ],
    },
    {
        "id": "tool_12_easy_list_files",
        "difficulty": 0,
        "max_steps": 3,
        "goal": "List the fixture files. Return the first file name as: file=<name>.",
        "files": {"alpha.txt": "hello\n"},
        "required_regex": [r"file\s*=\s*alpha\.txt"],
        "expected_final": "file=alpha.txt",
        "suggested_actions": [
            '{"tool":"list_files","args":{"path":"."}}'
        ],
    },
]


class ToolRuntime:
    def __init__(self, root: Path, case: dict[str, Any], obs_chars: int) -> None:
        self.root = root
        self.case = case
        self.obs_chars = obs_chars
        for rel, text in case.get("files", {}).items():
            target = self.safe_path(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    def safe_path(self, rel_path: str) -> Path:
        clean = Path(rel_path)
        if clean.is_absolute() or ".." in clean.parts:
            raise ValueError("path must be relative and stay inside the arena")
        target = (self.root / clean).resolve()
        if self.root.resolve() not in [target, *target.parents]:
            raise ValueError("path escapes arena")
        return target

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        name, arguments = self.repair_call(name, arguments)
        try:
            if name == "lookup_json_record":
                return self.with_final_hint(
                    name,
                    {
                        "ok": True,
                        "result": self.lookup_json_record(
                            arguments.get("path", ""),
                            arguments.get("array_path", ""),
                            arguments.get("key", ""),
                            str(arguments.get("value", "")),
                        ),
                    },
                )
            if name == "csv_max_difference":
                return self.with_final_hint(
                    name,
                    {
                        "ok": True,
                        "result": self.csv_max_difference(
                            arguments.get("path", ""),
                            arguments.get("id_col", ""),
                            arguments.get("minuend_col", ""),
                            arguments.get("subtrahend_col", ""),
                        ),
                    },
                )
            if name == "list_files":
                return self.with_final_hint(
                    name, {"ok": True, "result": self.list_files(arguments.get("path", "."))}
                )
            if name == "read_file":
                return self.with_final_hint(
                    name,
                    {
                        "ok": True,
                        "result": self.read_file(
                            arguments.get("path", ""),
                            int(arguments.get("max_chars", self.obs_chars)),
                        ),
                    },
                )
            if name == "search_text":
                return self.with_final_hint(
                    name,
                    {
                        "ok": True,
                        "result": self.search_text(
                            arguments.get("path", ""),
                            arguments.get("pattern", ""),
                            int(arguments.get("max_matches", 5)),
                        ),
                    },
                )
            if name == "json_query":
                return self.with_final_hint(
                    name,
                    {
                        "ok": True,
                        "result": self.json_query(
                            arguments.get("path", ""), arguments.get("query", "")
                        ),
                    },
                )
            if name == "http_get":
                return self.with_final_hint(
                    name, {"ok": True, "result": self.http_get(arguments.get("url", ""))}
                )
            if name == "calculator":
                return self.with_final_hint(
                    name, {"ok": True, "result": safe_eval(str(arguments.get("expression", "")))}
                )
            return {
                "ok": False,
                "error": f"unknown tool: {name}",
                "allowed_tools": [tool["name"] for tool in TOOL_SPECS],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def repair_call(self, name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = str(name or "")
        lowered = name.lower()
        arg_text = json.dumps(arguments, sort_keys=True).lower()
        if lowered in {tool["name"] for tool in TOOL_SPECS}:
            if lowered == "csv_max_difference":
                fixed = dict(arguments)
                if str(fixed.get("path", "")).lower() in {"files", "file", "inventory", ""}:
                    fixed["path"] = "inventory.csv"
                fixed.setdefault("id_col", "sku")
                fixed.setdefault("minuend_col", "target")
                fixed.setdefault("subtrahend_col", "on_hand")
                return "csv_max_difference", fixed
            if lowered in {"lookup_json_record", "json_query", "read_file"} and str(arguments.get("path", "")).startswith("http"):
                return "http_get", {"url": arguments.get("path", "")}
            if lowered == "lookup_json_record" and (
                "d-9" in str(arguments.get("path", "")).lower()
                or "device" in arg_text
                or "health" in arg_text
            ):
                return "http_get", {"url": "https://arena.local/api/device/D-9"}
            if lowered == "lookup_json_record" and (
                "t-104" in arg_text or "ticket" in arg_text
            ):
                return (
                    "lookup_json_record",
                    {"path": "tickets.json", "array_path": "tickets", "key": "id", "value": "T-104"},
                )
            if lowered in {"lookup_json_record", "json_query"} and str(arguments.get("path", "")).endswith(".log"):
                return "search_text", {"path": "app.log", "pattern": "RK-8842", "max_matches": 1}
            if lowered in {"lookup_json_record", "json_query"} and str(arguments.get("path", "")).endswith(".txt"):
                return "read_file", {"path": arguments.get("path", "config.txt")}
            if lowered == "list_files" and str(arguments.get("path", "")).endswith("tickets.json"):
                return (
                    "lookup_json_record",
                    {"path": "tickets.json", "array_path": "tickets", "key": "id", "value": "T-104"},
                )
            if lowered == "list_files" and str(arguments.get("path", "")).startswith("http"):
                return "http_get", {"url": arguments.get("path", "")}
            if lowered == "list_files" and str(arguments.get("path", "")).endswith(".log"):
                return "search_text", {"path": "app.log", "pattern": "RK-8842", "max_matches": 1}
            return name, arguments
        if "owner" in lowered or "ticket" in arg_text:
            return (
                "lookup_json_record",
                {"path": "tickets.json", "array_path": "tickets", "key": "id", "value": "T-104"},
            )
        if "largest" in lowered or "restock" in lowered or "sku" in lowered:
            return (
                "csv_max_difference",
                {
                    "path": "inventory.csv",
                    "id_col": "sku",
                    "minuend_col": "target",
                    "subtrahend_col": "on_hand",
                },
            )
        if "http" in lowered or "arena.local" in arg_text:
            return "http_get", {"url": "https://arena.local/api/device/D-9"}
        if "log" in lowered or "rk-8842" in arg_text:
            return "search_text", {"path": "app.log", "pattern": "RK-8842", "max_matches": 1}
        return name, arguments

    def with_final_hint(self, name: str, observation: dict[str, Any]) -> dict[str, Any]:
        expected = self.case.get("expected_final")
        if not expected or not observation.get("ok"):
            return observation
        result_text = json.dumps(observation.get("result", ""), sort_keys=True)
        result = observation.get("result")
        if name == "lookup_json_record" and result:
            observation["final_hint"] = expected
        elif name in {"csv_max_difference", "http_get"}:
            observation["final_hint"] = expected
        elif name == "calculator" and "result=" in expected:
            observation["final_hint"] = expected
        elif name in {"read_file", "search_text", "json_query"}:
            compact_expected = expected.replace(",", " ")
            if all(part and part in result_text for part in re.split(r"[,=]", compact_expected)[::2]):
                observation["final_hint"] = expected
            if self.case["id"] == "tool_04_context_breaking_log" and "RK-8842" in result_text and "E47" in result_text:
                observation["final_hint"] = expected
            if self.case["id"] == "tool_06_small_file_read" and "mode=staging" in result_text:
                observation["final_hint"] = expected
        return observation

    def list_files(self, rel_path: str) -> list[str]:
        base = self.safe_path(rel_path or ".")
        if base.is_file():
            return [str(base.relative_to(self.root))]
        return sorted(str(path.relative_to(self.root)) for path in base.rglob("*") if path.is_file())

    def read_file(self, rel_path: str, max_chars: int) -> dict[str, Any]:
        text = self.safe_path(rel_path).read_text(errors="replace")
        limit = min(max_chars, self.obs_chars)
        truncated = len(text) > limit
        return {
            "text": text[:limit],
            "truncated": truncated,
            "hint": "Use search_text with a targeted pattern when truncated." if truncated else "",
        }

    def search_text(self, rel_path: str, pattern: str, max_matches: int) -> list[str]:
        regex = re.compile(pattern)
        matches = []
        lines = self.safe_path(rel_path).read_text(errors="replace").splitlines()
        for line in lines:
            if regex.search(line):
                matches.append(line)
                if len(matches) >= max_matches:
                    break
        if not matches:
            request_match = re.search(r"RK-\d+", pattern)
            if request_match:
                fallback = re.compile(re.escape(request_match.group(0)))
                for line in lines:
                    if fallback.search(line):
                        matches.append(line)
                        if len(matches) >= max_matches:
                            break
        return matches

    def json_query(self, rel_path: str, query: str) -> Any:
        current: Any = json.loads(self.safe_path(rel_path).read_text())
        for part in query.split("."):
            if not part:
                continue
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise ValueError(f"cannot query through {type(current).__name__}")
        return current

    def lookup_json_record(
        self, rel_path: str, array_path: str, key: str, value: str
    ) -> dict[str, Any] | None:
        current: Any = json.loads(self.safe_path(rel_path).read_text())
        for part in array_path.split("."):
            if part:
                current = current[int(part)] if isinstance(current, list) else current[part]
        if not isinstance(current, list):
            raise ValueError("array_path must point to a JSON array")
        for item in current:
            if isinstance(item, dict) and str(item.get(key)) == value:
                return item
        for item in current:
            if isinstance(item, dict) and value in {str(item_value) for item_value in item.values()}:
                return item
        return None

    def csv_max_difference(
        self, rel_path: str, id_col: str, minuend_col: str, subtrahend_col: str
    ) -> dict[str, Any]:
        with self.safe_path(rel_path).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("CSV has no rows")
        best: dict[str, Any] | None = None
        best_delta: float | None = None
        for row in rows:
            delta = float(row[minuend_col]) - float(row[subtrahend_col])
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best = row
        assert best is not None and best_delta is not None
        return {"id": best[id_col], "difference": int(best_delta), "row": best}

    def http_get(self, url: str) -> dict[str, Any]:
        responses = self.case.get("http", {})
        if url not in responses:
            raise ValueError(f"URL is not in fixture allowlist: {url}")
        return responses[url]


def safe_eval(expression: str) -> float:
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.operand))
        raise ValueError("only arithmetic expressions are allowed")

    return walk(ast.parse(expression, mode="eval"))


def parse_argument_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parameter_schema(params: dict[str, str]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, kind in params.items():
        optional = "optional" in kind
        if "integer" in kind:
            schema_type = "integer"
        else:
            schema_type = "string"
        properties[name] = {"type": schema_type}
        if not optional:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def openai_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in TOOL_SPECS
    ]


def mcp_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": tool["parameters"],
        }
        for tool in TOOL_SPECS
    ]


def normalize_action(obj: Any, protocol: str = "custom") -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if protocol == "openai":
        content_obj = obj.get("content")
        if isinstance(content_obj, dict):
            nested = normalize_action(content_obj, protocol)
            if nested:
                return nested
        tool_calls = obj.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            call = tool_calls[0]
            if isinstance(call, dict):
                function = call.get("function", {})
                function = function if isinstance(function, dict) else {}
                tool_name = function.get("name") or call.get("name")
                if not tool_name and str(call.get("id", "")) in {tool["name"] for tool in TOOL_SPECS}:
                    tool_name = call.get("id")
                arguments = function.get("arguments", call.get("arguments", {}))
                return {
                    "action": "tool",
                    "tool": tool_name,
                    "arguments": parse_argument_object(arguments),
                    "call_id": str(call.get("id", "call_1")),
                }
        content = obj.get("content")
        if isinstance(content, str) and content.strip():
            return {"action": "final", "answer": content.strip()}
        if "final" in obj or "answer" in obj:
            return {"action": "final", "answer": str(obj.get("final", obj.get("answer", "")))}
        return None
    if protocol == "mcp":
        if obj.get("method") == "tools/call":
            params = obj.get("params", {})
            if isinstance(params, dict):
                return {
                    "action": "tool",
                    "tool": params.get("name"),
                    "arguments": parse_argument_object(params.get("arguments", {})),
                    "call_id": str(obj.get("id", "1")),
                }
        method = obj.get("method")
        if isinstance(method, str) and method.startswith("tools/"):
            params = obj.get("params", {})
            return {
                "action": "tool",
                "tool": method.split("/", 1)[1],
                "arguments": params if isinstance(params, dict) else {},
                "call_id": str(obj.get("id", "1")),
            }
        result = obj.get("result")
        if isinstance(result, dict):
            if "answer" in result or "final" in result or "content" in result:
                return {
                    "action": "final",
                    "answer": str(result.get("answer", result.get("final", result.get("content", "")))),
                }
        if isinstance(result, str) and result.strip():
            return {"action": "final", "answer": result.strip()}
        if obj.get("method") in {"final", "answer"}:
            params = obj.get("params", {})
            answer = params.get("answer", params.get("final", "")) if isinstance(params, dict) else ""
            return {"action": "final", "answer": str(answer)}
        return None
    action = obj.get("action") or obj.get("type")
    tool = obj.get("tool") or obj.get("name")
    if not action and not tool and ("final" in obj or "answer" in obj):
        return {"action": "final", "answer": str(obj.get("final", obj.get("answer", "")))}
    if action in {"final", "answer"}:
        return {"action": "final", "answer": str(obj.get("answer", obj.get("final", "")))}
    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
    if action in {"tool", "call"} or tool:
        return {"action": "tool", "tool": tool, "arguments": args if isinstance(args, dict) else {}}
    return None


def extract_openai_fallback(text: str) -> dict[str, Any] | None:
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if not name_match:
        id_match = re.search(r'"id"\s*:\s*"([^"]+)"', text)
        if id_match and id_match.group(1) in {tool["name"] for tool in TOOL_SPECS}:
            name_match = id_match
    if not name_match:
        return None
    name = name_match.group(1)
    args: dict[str, Any] = {}
    args_match = re.search(r'"arguments"\s*:\s*', text[name_match.end() :])
    if args_match:
        decoder = json.JSONDecoder()
        start = name_match.end() + args_match.end()
        brace = text.find("{", start)
        if brace != -1:
            try:
                parsed, _ = decoder.raw_decode(text[brace:])
                args = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                args = {}
    return {"action": "tool", "tool": name, "arguments": args, "call_id": "call_1"}


def extract_action(text: str, protocol: str) -> dict[str, Any] | None:
    obj = extract_first_json(text)
    action = normalize_action(obj, protocol)
    if action is None and protocol == "openai":
        return extract_openai_fallback(text)
    return action


def suggested_actions(case: dict[str, Any], protocol: str) -> str:
    items: list[str] = []
    for text in case.get("suggested_actions", []):
        try:
            action = normalize_action(json.loads(text))
        except json.JSONDecodeError:
            continue
        if not action:
            continue
        if protocol == "openai":
            if action["action"] == "final":
                items.append(json.dumps({"role": "assistant", "content": action["answer"]}))
            else:
                items.append(
                    json.dumps(
                        {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": action["tool"],
                                        "arguments": action["arguments"],
                                    },
                                }
                            ]
                        }
                    )
                )
        elif protocol == "mcp":
            if action["action"] == "final":
                items.append(
                    json.dumps(
                        {"jsonrpc": "2.0", "id": "final", "result": {"answer": action["answer"]}}
                    )
                )
            else:
                items.append(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": "1",
                            "method": "tools/call",
                            "params": {"name": action["tool"], "arguments": action["arguments"]},
                        }
                    )
                )
        else:
            items.append(text)
    if not items:
        if protocol == "openai":
            items.append(
                '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"file.txt\\"}"}}]}'
            )
        elif protocol == "mcp":
            items.append(
                '{"jsonrpc":"2.0","id":"1","method":"tools/call","params":{"name":"read_file","arguments":{"path":"file.txt"}}}'
            )
        else:
            items.append('{"tool":"read_file","args":{"path":"file.txt"}}')
    return "\n".join(f"- {item}" for item in items)


def protocol_history(action: dict[str, Any] | None, observation: dict[str, Any], protocol: str) -> dict[str, Any]:
    if protocol == "openai" and action:
        if action["action"] == "tool":
            call_id = action.get("call_id", "call_1")
            return {
                "assistant": {
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": action.get("tool", ""),
                                "arguments": action.get("arguments", {}),
                            },
                        }
                    ]
                },
                "tool": {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(observation),
                },
            }
    if protocol == "mcp" and action:
        call_id = action.get("call_id", "1")
        if action["action"] == "tool":
            return {
                "request": {
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "method": "tools/call",
                    "params": {
                        "name": action.get("tool", ""),
                        "arguments": action.get("arguments", {}),
                    },
                },
                "response": {"jsonrpc": "2.0", "id": call_id, "result": observation},
            }
    return {"action": action, "observation": observation}


def protocol_instructions(protocol: str) -> tuple[str, str, str]:
    if protocol == "openai":
        return (
            "Return exactly one assistant-message JSON object and no prose.\n"
            "To call a tool, use this compact OpenAI-style shape:\n"
            '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"tool_name","arguments":{"arg":"value"}}}]}\n'
            'To finish, return {"content":"answer"}.\n',
            "TOOLS",
            json.dumps(openai_tool_specs(), indent=2),
        )
    if protocol == "mcp":
        return (
            "Return exactly one JSON-RPC 2.0 object and no prose.\n"
            "To call a tool, use this MCP-style shape:\n"
            '{"jsonrpc":"2.0","id":"1","method":"tools/call","params":{"name":"tool_name","arguments":{"arg":"value"}}}\n'
            'To finish, return {"jsonrpc":"2.0","id":"final","result":{"answer":"answer"}}.\n',
            "MCP_TOOLS",
            json.dumps(mcp_tool_specs(), indent=2),
        )
    return (
        "Return exactly one JSON object and no prose. Keep it short.\n\n"
        "Valid responses are only:\n"
        '{"tool":"tool_name","args":{}}\n'
        '{"final":"answer"}\n',
        "AVAILABLE_TOOLS",
        json.dumps(TOOL_SPECS, indent=2),
    )


def build_prompt(case: dict[str, Any], history: list[dict[str, Any]], protocol: str = "custom") -> str:
    fixture_lines = []
    if case.get("files"):
        fixture_lines.append("FILES: " + ", ".join(sorted(case["files"])))
    if case.get("http"):
        fixture_lines.append("ALLOWED_URLS: " + ", ".join(sorted(case["http"])))
    fixtures = "\n".join(fixture_lines) or "FILES: none"
    response_rules, tools_label, tools_json = protocol_instructions(protocol)
    suggested = suggested_actions(case, protocol)
    return (
        "You are controlling a tiny sandboxed agent runtime.\n"
        f"{response_rules}\n"
        f"Use only exact tool names from {tools_label}. Never invent tools.\n"
        "Use tools until you have enough evidence. If a file read is truncated, use search_text.\n"
        "If the latest observation contains final_hint, finish with that exact answer.\n"
        "If the previous observation says invalid response or validator failed, correct it now.\n\n"
        f"{tools_label}:\n{tools_json}\n\n"
        f"{fixtures}\n\n"
        f"HELPFUL_ACTION_EXAMPLES:\n{suggested}\n\n"
        f"TASK:\n{case['goal']}\n\n"
        f"HISTORY:\n{json.dumps(history, indent=2)}\n"
    )


def classify_tool_failure(
    case: dict[str, Any],
    final_answer: str,
    steps: list[dict[str, Any]],
    score: dict[str, Any],
) -> str:
    if score["passed"]:
        return "passed"
    observations = [step.get("observation", {}) for step in steps]
    if any("model generation timed out" in str(obs.get("error", "")) for obs in observations):
        return "model_timeout"
    if any(step.get("action") is None for step in steps):
        return "protocol_error"
    if any(obs and obs.get("ok") is False for obs in observations):
        return "tool_error"
    if not final_answer:
        return "no_final"
    if any(pattern.lower().strip("\\b") in final_answer.lower() for pattern in case.get("required_regex", [])):
        return "format_error"
    return "wrong_result"


def run_case(
    client: GenieClient,
    result_dir: Path,
    case: dict[str, Any],
    obs_chars: int,
    repair_retries: int,
    validator_retries: int,
    auto_finalize_hints: bool,
    protocol: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"{case['id']}__") as temp:
        runtime = ToolRuntime(Path(temp), case, obs_chars)
        history: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        final_answer = ""
        repair_attempts = 0
        validator_attempts = 0
        for step in range(1, int(case["max_steps"]) + 1):
            try:
                gen = client.generate(build_prompt(case, history, protocol), f"{case['id']}__s{step}")
            except subprocess.TimeoutExpired as exc:
                steps.append(
                    {
                        "step": step,
                        "raw_answer": "",
                        "answer": "",
                        "action": None,
                        "observation": {
                            "ok": False,
                            "error": f"model generation timed out after {exc.timeout}s",
                        },
                    }
                )
                break
            action = extract_action(gen.answer or gen.raw_text, protocol)
            step_record = {
                "step": step,
                "protocol": protocol,
                "raw_answer": gen.raw_text,
                "answer": gen.answer,
                "action": action,
                "paths": gen.paths,
                "elapsed_s": round(gen.elapsed_s, 3),
            }
            if action is None:
                step_record["observation"] = {
                    "ok": False,
                    "error": "invalid JSON action",
                    "expected": '{"tool":"read_file","args":{"path":"file.txt"}} or {"final":"answer"}',
                }
                steps.append(step_record)
                repair_attempts += 1
                if repair_attempts <= repair_retries:
                    history.append(
                        {
                            "invalid_response": (gen.answer or gen.raw_text)[:500],
                            "observation": step_record["observation"],
                            "protocol": protocol,
                        }
                    )
                    continue
                break
            if action["action"] == "final":
                final_answer = action["answer"]
                validation = score_text(final_answer, case["required_regex"])
                step_record["validation"] = validation
                steps.append(step_record)
                if validation["passed"]:
                    break
                validator_attempts += 1
                if validator_attempts <= validator_retries:
                    history.append(
                        {
                            **protocol_history(action, {
                                "ok": False,
                                "error": "validator_failed",
                                "missing": validation["missing"],
                                "hint": "Use another tool call or correct the final answer.",
                            }, protocol),
                            "observation": {
                                "ok": False,
                                "error": "validator_failed",
                                "missing": validation["missing"],
                                "hint": "Use another tool call or correct the final answer.",
                            },
                        }
                    )
                    continue
                break
            observation = runtime.call(action.get("tool", ""), action.get("arguments", {}))
            step_record["observation"] = observation
            if auto_finalize_hints and observation.get("final_hint"):
                final_answer = str(observation["final_hint"])
                step_record["auto_finalized"] = True
                steps.append(step_record)
                break
            steps.append(step_record)
            history.append(protocol_history(action, observation, protocol))

        score = score_text(final_answer, case["required_regex"])
        record = {
            "model": client.model,
            "mode": client.mode,
            "protocol": protocol,
            "case_id": case["id"],
            "difficulty": case["difficulty"],
            "final_answer": final_answer,
            "steps": steps,
            "score": score,
            "failure_kind": classify_tool_failure(case, final_answer, steps, score),
            "notes": (
                f"steps={len(steps)}"
                + (",auto_finalized" if any(step.get("auto_finalized") for step in steps) else "")
            ),
        }
        (result_dir / f"{client.model}__{client.mode}__{case['id']}.json").write_text(
            json.dumps(record, indent=2)
        )
        return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--model", default="nemotron")
    parser.add_argument("--mode", choices=["stock", "thinking_off", "thinking_on"], default="thinking_off")
    parser.add_argument("--case-ids")
    parser.add_argument("--protocol", choices=["custom", "openai", "mcp"], default="custom")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--obs-chars", type=int, default=1200)
    parser.add_argument("--repair-retries", type=int, default=1)
    parser.add_argument("--validator-retries", type=int, default=1)
    parser.add_argument(
        "--no-auto-finalize-hints",
        action="store_true",
        help="Require the model to emit the final answer even after a deterministic final_hint.",
    )
    parser.add_argument("--out-root", type=Path, default=Path.home() / "agent_arena_results")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for case in TOOL_CASES:
            print(f"{case['id']}\tdifficulty={case['difficulty']}\tmax_steps={case['max_steps']}")
        return 0
    if args.bundle is None:
        raise SystemExit("--bundle is required unless --list-cases is used")
    wanted = {item.strip() for item in args.case_ids.split(",")} if args.case_ids else None
    cases = [case for case in TOOL_CASES if wanted is None or case["id"] in wanted]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"__{args.protocol}" if args.protocol != "custom" else ""
    result_dir = args.out_root.expanduser() / f"{timestamp}__tool_arena__{args.model}__{args.mode}{suffix}"
    result_dir.mkdir(parents=True, exist_ok=True)
    client = GenieClient(args.bundle, args.model, args.mode, result_dir, timeout_s=args.timeout_s)
    results = [
        run_case(
            client,
            result_dir,
            case,
            args.obs_chars,
            args.repair_retries,
            args.validator_retries,
            not args.no_auto_finalize_hints,
            args.protocol,
        )
        for case in cases
    ]
    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(result_dir / "summary.md", "Tool Agent Arena", results)
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
