#!/usr/bin/env python3
"""Run an OR-agent-style semantic tool arena.

This lane intentionally avoids the raw low-level tool surface. It registers
task-shaped Pydantic function tools directly on an Agent, mirroring the OR
agent pattern. It can also run the same semantic tools through MCP for a
transport comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_arena.model_client import score_text, write_summary
from agent_arena.pydantic_arena import get_server_log, require_pydantic_ai, reset_server_log, root_from_base_url
from agent_arena.pydantic_mcp_arena import (
    apply_env_defaults,
    build_model,
    compact_values,
    read_jsonl,
    require_mcp,
    server_debug_log_enabled,
    summarize_messages,
)
from agent_arena.semantic_runtime import (
    SEMANTIC_AGENT_INSTRUCTIONS,
    SemanticRuntime,
    logged_semantic_call,
    semantic_hint_for_case,
    semantic_tools_for_case,
)
from agent_arena.tool_arena import TOOL_CASES

try:
    from pydantic_ai import RunContext
except ModuleNotFoundError:
    RunContext = Any  # type: ignore[misc,assignment]


@dataclass
class SemanticDeps:
    runtime: SemanticRuntime
    tool_log: list[dict[str, Any]] = field(default_factory=list)


def telemetry(
    score: dict[str, Any],
    output: str,
    exception: str,
    server_requests: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_protocol_valid = all(item.get("parsed_ok") for item in server_requests) if server_requests else not exception
    tool_call_attempted = (
        any(item.get("finish_reason") == "tool_calls" for item in server_requests)
        if server_requests
        else bool(tool_calls)
    )
    return {
        "final_pass": score["passed"],
        "raw_protocol_valid": raw_protocol_valid,
        "tool_call_attempted": tool_call_attempted,
        "tool_call_executed": bool(tool_calls),
        "client_repaired": False,
        "model_timeout": any(item.get("timed_out") for item in server_requests),
        "empty_answer": not output.strip(),
        "placeholder_answer": any(
            "<" in str(item.get("raw_answer", "")) or "placeholder" in str(item.get("raw_answer", "")).lower()
            for item in server_requests
        )
        or "<" in output
        or "placeholder" in output.lower(),
        "parse_failures": sum(1 for item in server_requests if not item.get("parsed_ok")),
        "model_requests": len(server_requests),
        "semantic_tool_calls": len(tool_calls),
        "agent_exception": bool(exception),
    }


def classify_failure(score: dict[str, Any], facts: dict[str, Any], exception: str) -> str:
    if score["passed"]:
        return "passed"
    if facts["model_timeout"]:
        return "model_timeout"
    if exception:
        return "agent_exception"
    if not facts["raw_protocol_valid"]:
        return "protocol_error"
    if facts["placeholder_answer"]:
        return "placeholder_answer"
    if facts["tool_call_attempted"] and not facts["tool_call_executed"]:
        return "tool_not_executed"
    if not facts["tool_call_attempted"]:
        return "no_tool_call"
    return "wrong_result"


def register_function_tools(agent: Any, allowed_tools: list[str]) -> None:
    allowed = set(allowed_tools)

    def include(name: str) -> bool:
        return name in allowed

    if include("lookup_ticket_owner"):

        @agent.tool
        def lookup_ticket_owner(ctx: RunContext[SemanticDeps], ticket_id: str) -> dict[str, Any]:
            """Return the owner for a ticket id from the fixture ticket data."""
            return logged_semantic_call(
                ctx.deps.tool_log,
                None,
                "lookup_ticket_owner",
                {"ticket_id": ticket_id},
                lambda: ctx.deps.runtime.lookup_ticket_owner(ticket_id),
            )

    if include("find_largest_restock"):

        @agent.tool
        def find_largest_restock(ctx: RunContext[SemanticDeps]) -> dict[str, Any]:
            """Return the SKU with the largest target - on_hand restock shortfall."""
            return logged_semantic_call(
                ctx.deps.tool_log,
                None,
                "find_largest_restock",
                {},
                ctx.deps.runtime.find_largest_restock,
            )

    if include("get_device_status"):

        @agent.tool
        def get_device_status(ctx: RunContext[SemanticDeps], device_id: str) -> dict[str, Any]:
            """Return fixture HTTP health fields for a device id such as D-9."""
            return logged_semantic_call(
                ctx.deps.tool_log,
                None,
                "get_device_status",
                {"device_id": device_id},
                lambda: ctx.deps.runtime.get_device_status(device_id),
            )

    if include("find_log_error_code"):

        @agent.tool
        def find_log_error_code(ctx: RunContext[SemanticDeps], request_id: str) -> dict[str, Any]:
            """Return the error code for a request id found in the fixture logs."""
            return logged_semantic_call(
                ctx.deps.tool_log,
                None,
                "find_log_error_code",
                {"request_id": request_id},
                lambda: ctx.deps.runtime.find_log_error_code(request_id),
            )

    if include("calculate"):

        @agent.tool
        def calculate(ctx: RunContext[SemanticDeps], expression: str) -> dict[str, Any]:
            """Evaluate a small arithmetic expression and return the numeric result."""
            return logged_semantic_call(
                ctx.deps.tool_log,
                None,
                "calculate",
                {"expression": expression},
                lambda: ctx.deps.runtime.calculate(expression),
            )

    if include("get_config_value"):

        @agent.tool
        def get_config_value(ctx: RunContext[SemanticDeps], key: str) -> dict[str, Any]:
            """Read a key=value field from the simple fixture text file."""
            return logged_semantic_call(
                ctx.deps.tool_log,
                None,
                "get_config_value",
                {"key": key},
                lambda: ctx.deps.runtime.get_config_value(key),
            )

    if include("get_profile_field"):

        @agent.tool
        def get_profile_field(ctx: RunContext[SemanticDeps], field: str) -> dict[str, Any]:
            """Read one top-level field from profile.json."""
            return logged_semantic_call(
                ctx.deps.tool_log,
                None,
                "get_profile_field",
                {"field": field},
                lambda: ctx.deps.runtime.get_profile_field(field),
            )

    if include("get_ping_status"):

        @agent.tool
        def get_ping_status(ctx: RunContext[SemanticDeps]) -> dict[str, Any]:
            """Fetch the fixture ping endpoint and return its status field."""
            return logged_semantic_call(
                ctx.deps.tool_log,
                None,
                "get_ping_status",
                {},
                ctx.deps.runtime.get_ping_status,
            )

    if include("list_first_fixture_file"):

        @agent.tool
        def list_first_fixture_file(ctx: RunContext[SemanticDeps]) -> dict[str, Any]:
            """Return the first fixture filename in sorted order."""
            return logged_semantic_call(
                ctx.deps.tool_log,
                None,
                "list_first_fixture_file",
                {},
                ctx.deps.runtime.list_first_fixture_file,
            )


def build_mcp_toolset(
    args: argparse.Namespace,
    MCPToolset: Any,
    StdioTransport: Any,
    case_path: Path,
    sandbox_root: Path,
    tool_log_path: Path,
    allowed_tools: list[str],
) -> Any:
    return MCPToolset(
        StdioTransport(
            sys.executable,
            args=[
                "-m",
                "agent_arena.semantic_mcp_tool_server",
                "--case-file",
                str(case_path),
                "--root",
                str(sandbox_root),
                "--obs-chars",
                str(args.obs_chars),
                "--allowed-tools",
                ",".join(allowed_tools),
                "--tool-log",
                str(tool_log_path),
            ],
            cwd=str(Path.cwd()),
        ),
        init_timeout=args.mcp_init_timeout,
        read_timeout=args.mcp_read_timeout,
        max_retries=args.mcp_tool_retries,
        include_instructions=args.include_mcp_instructions,
    )


async def run_case_async(
    args: argparse.Namespace,
    case: dict[str, Any],
    Agent: Any,
    OpenAIChatModel: Any,
    OpenAIProvider: Any,
    result_dir: Path,
    MCPToolset: Any | None = None,
    StdioTransport: Any | None = None,
) -> dict[str, Any]:
    if server_debug_log_enabled(args):
        reset_server_log(args.base_url)
    started = time.monotonic()
    output = ""
    exception = ""
    messages: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    allowed_tools = semantic_tools_for_case(case, args.tool_pruning)

    with tempfile.TemporaryDirectory(prefix=f"semantic_arena__{case['id']}__") as temp:
        temp_root = Path(temp)
        case_path = temp_root / "case.json"
        tool_log_path = temp_root / "semantic_tool_calls.jsonl"
        sandbox_root = temp_root / "sandbox"
        case_path.write_text(json.dumps(case))
        runtime = SemanticRuntime(sandbox_root.resolve(), case, args.obs_chars)
        deps = SemanticDeps(runtime=runtime)

        model = build_model(args, OpenAIChatModel, OpenAIProvider)
        toolsets = []
        if args.transport == "mcp" and allowed_tools:
            assert MCPToolset is not None and StdioTransport is not None
            toolsets.append(
                build_mcp_toolset(
                    args,
                    MCPToolset,
                    StdioTransport,
                    case_path,
                    sandbox_root,
                    tool_log_path,
                    allowed_tools,
                )
            )
        agent_kwargs: dict[str, Any] = {
            "instructions": SEMANTIC_AGENT_INSTRUCTIONS,
            "toolsets": toolsets,
            "retries": args.agent_retries,
        }
        if args.transport == "function":
            agent_kwargs["deps_type"] = SemanticDeps
        agent = Agent(model, **agent_kwargs)
        if args.transport == "function":
            register_function_tools(agent, allowed_tools)

        user_prompt = (
            f"{semantic_hint_for_case(case)}\n"
            f"User task:\n{case['goal']}\n"
        )
        try:
            if args.transport == "mcp":
                async with agent:
                    result = await agent.run(user_prompt)
            else:
                result = await agent.run(user_prompt, deps=deps)
            output = str(result.output)
            messages = summarize_messages(result)
        except Exception as exc:
            exception = repr(exc)

        if args.transport == "mcp":
            tool_calls = read_jsonl(tool_log_path)
        else:
            tool_calls = deps.tool_log

    elapsed_s = time.monotonic() - started
    server_requests = get_server_log(args.base_url) if server_debug_log_enabled(args) else []
    score = score_text(output, case["required_regex"])
    facts = telemetry(score, output, exception, server_requests, tool_calls)
    parser_modes = compact_values([str(item.get("parser", "")) for item in server_requests])
    parse_statuses = compact_values([str(item.get("parse_status", "")) for item in server_requests])
    executed_tools = compact_values([str(item.get("name", "")) for item in tool_calls])
    record = {
        "model": args.model_label,
        "mode": args.mode,
        "client": "pydantic_ai_semantic",
        "provider": args.provider,
        "transport": args.transport,
        "tool_transport": "function" if args.transport == "function" else "mcp_stdio",
        "tool_pruning": args.tool_pruning,
        "allowed_tools": allowed_tools,
        "case_id": case["id"],
        "difficulty": case["difficulty"],
        "output": output,
        "score": score,
        "telemetry": facts,
        "failure_kind": classify_failure(score, facts, exception),
        "exception": exception,
        "elapsed_s": round(elapsed_s, 3),
        "semantic_tool_calls": tool_calls,
        "server_requests": server_requests,
        "messages": messages,
        "notes": (
            f"model_requests={facts['model_requests']},"
            f"semantic_tool_calls={facts['semantic_tool_calls']},"
            f"parse_failures={facts['parse_failures']},"
            f"parser={parser_modes},"
            f"parse_status={parse_statuses},"
            f"executed_tools={executed_tools},"
            f"transport={args.transport},"
            f"tool_pruning={args.tool_pruning}"
        ),
    }
    (result_dir / f"{args.model_label}__{args.mode}__{case['id']}.json").write_text(
        json.dumps(record, indent=2)
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("agent_arena/.env"))
    parser.add_argument(
        "--provider",
        choices=["openai-compatible", "azure", "azure-foundry-anthropic", "anthropic"],
    )
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--azure-endpoint")
    parser.add_argument("--azure-api-version")
    parser.add_argument("--anthropic-foundry-base-url")
    parser.add_argument("--model-name")
    parser.add_argument("--model-label")
    parser.add_argument("--mode", choices=["stock", "thinking_off", "thinking_on"], default="stock")
    parser.add_argument("--transport", choices=["function", "mcp"], default="function")
    parser.add_argument("--case-ids")
    parser.add_argument("--agent-retries", type=int, default=1)
    parser.add_argument("--mcp-tool-retries", type=int, default=0)
    parser.add_argument("--mcp-init-timeout", type=float, default=10)
    parser.add_argument("--mcp-read-timeout", type=float, default=300)
    parser.add_argument("--include-mcp-instructions", action="store_true")
    parser.add_argument("--tool-pruning", choices=["case", "none"], default="case")
    parser.add_argument("--obs-chars", type=int, default=1200)
    parser.add_argument("--out-root", type=Path, default=Path.home() / "agent_arena_results")
    parser.add_argument("--no-server-debug-log", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


async def amain() -> int:
    args = parse_args()
    if args.list_cases:
        for case in TOOL_CASES:
            print(
                f"{case['id']}\tdifficulty={case['difficulty']}\t"
                f"semantic_tools={','.join(semantic_tools_for_case(case, 'case'))}"
            )
        return 0
    apply_env_defaults(args)
    Agent, OpenAIChatModel, OpenAIProvider = require_pydantic_ai()
    MCPToolset = None
    StdioTransport = None
    if args.transport == "mcp":
        MCPToolset, StdioTransport = require_mcp()
    wanted = {item.strip() for item in args.case_ids.split(",")} if args.case_ids else None
    cases = [case for case in TOOL_CASES if wanted is None or case["id"] in wanted]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = (
        args.out_root.expanduser()
        / f"{timestamp}__pydantic_semantic_arena__{args.model_label}__{args.mode}__{args.provider}__{args.transport}__{args.tool_pruning}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    results = [
        await run_case_async(
            args,
            case,
            Agent,
            OpenAIChatModel,
            OpenAIProvider,
            result_dir,
            MCPToolset,
            StdioTransport,
        )
        for case in cases
    ]
    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(result_dir / "summary.md", "Pydantic AI Semantic Agent Arena", results)
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())
    if server_debug_log_enabled(args):
        print(f"SERVER_DEBUG_ROOT={root_from_base_url(args.base_url)}/debug/requests")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
