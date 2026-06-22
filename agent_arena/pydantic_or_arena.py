#!/usr/bin/env python3
"""Run OR-agent-style workflow benchmarks through Pydantic AI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_arena.model_client import write_summary
from agent_arena.or_agent_runtime import (
    OR_AGENT_INSTRUCTION_STYLES,
    OR_ARENA_CASES,
    ORRuntime,
    or_agent_instructions,
    score_or_tool_calls,
)
from agent_arena.pydantic_arena import (
    get_server_log,
    require_pydantic_ai,
    reset_server_log,
    root_from_base_url,
)
from agent_arena.pydantic_mcp_arena import (
    apply_env_defaults,
    build_model,
    build_provider,
    compact_values,
    read_jsonl,
    require_mcp,
    server_debug_log_enabled,
    summarize_messages,
)

try:
    from pydantic_ai import ModelSettings, RunContext
except ModuleNotFoundError:
    ModelSettings = None  # type: ignore[assignment]
    RunContext = Any  # type: ignore[misc,assignment]


@dataclass
class ORDeps:
    runtime: ORRuntime


def build_or_model(args: argparse.Namespace, OpenAIChatModel: Any, OpenAIProvider: Any) -> Any:
    if args.provider in {"anthropic", "azure-foundry-anthropic"}:
        return build_model(args, OpenAIChatModel, OpenAIProvider)
    try:
        from pydantic_ai.profiles.openai import OpenAIModelProfile
    except ImportError:
        return OpenAIChatModel(
            args.model_name,
            provider=build_provider(args, OpenAIProvider),
        )
    return OpenAIChatModel(
        args.model_name,
        provider=build_provider(args, OpenAIProvider),
        profile=OpenAIModelProfile(
            openai_supports_strict_tool_definition=args.openai_strict_tools,
            openai_supports_tool_choice_required=args.openai_tool_choice_required,
        ),
    )


def mode_instruction_prefix(mode: str) -> str:
    if mode == "thinking_off":
        return (
            "detailed thinking off\n"
            "Use tools directly. Do not write chain-of-thought; keep the final answer concise.\n\n"
        )
    if mode == "thinking_on":
        return (
            "detailed thinking on\n"
            "Use a short reasoning budget, then act with tools and keep the final answer concise.\n\n"
        )
    return ""


def register_function_tools(agent: Any, *, strict: bool, sequential: bool) -> None:
    @agent.tool(strict=strict, sequential=sequential)
    def get_case(ctx: RunContext[ORDeps], case_id: str) -> dict[str, Any]:
        """Fetch the surgical case and required equipment list from the EMR.

        Call this FIRST to learn what instruments/supplies the procedure requires.

        Args:
            case_id: The surgical case identifier, for example CASE-BENCH-1.
        """
        return ctx.deps.runtime.log_call(
            "get_case",
            {"case_id": case_id},
            lambda: ctx.deps.runtime.get_case(case_id),
        )

    @agent.tool(strict=strict, sequential=sequential)
    def check_supplies(ctx: RunContext[ORDeps]) -> dict[str, Any]:
        """Compare detected instruments against the surgical case requirements.

        Call this AFTER get_case. all_present=false means there are supply
        deficits: set yellow and request_resupply for each deficit. Supply
        deficits are NOT sterile zone issues and must NOT create human_review
        tasks.
        """
        return ctx.deps.runtime.log_call("check_supplies", {}, ctx.deps.runtime.check_supplies)

    @agent.tool(strict=strict, sequential=sequential)
    def inspect_scene(ctx: RunContext[ORDeps], image_path: str) -> dict[str, Any]:
        """Inspect the OR scene image for sterile zone violations.

        Call this tool whenever at least one instrument was detected by the EI model
        (i.e. visible_items is non-empty). Skip it only if no objects were detected.

        IMPORTANT: check the verdict field:
        verdict=true -> sterile zone issue exists -> set red light and create human_review task.
        verdict=false -> sterile zone is clear -> do NOT create a human_review task.
        """
        return ctx.deps.runtime.log_call(
            "inspect_scene",
            {"image_path": image_path},
            lambda: ctx.deps.runtime.inspect_scene(image_path),
        )

    @agent.tool(strict=strict, sequential=sequential)
    def create_task(
        ctx: RunContext[ORDeps],
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
        return ctx.deps.runtime.log_call(
            "create_task",
            {
                "case_id": case_id,
                "task_type": task_type,
                "priority": priority,
                "summary": summary,
                "reason": reason,
            },
            lambda: ctx.deps.runtime.create_task(case_id, task_type, priority, summary, reason),
        )

    @agent.tool(strict=strict, sequential=sequential)
    def request_resupply(ctx: RunContext[ORDeps], item_name: str, room_id: str, urgency: str) -> dict[str, Any]:
        """Request sterile processing delivery for a missing item.

        Args:
            item_name: Name of the item to resupply.
            room_id: The OR room identifier.
            urgency: One of low, normal, high.
        """
        return ctx.deps.runtime.log_call(
            "request_resupply",
            {"item_name": item_name, "room_id": room_id, "urgency": urgency},
            lambda: ctx.deps.runtime.request_resupply(item_name, room_id, urgency),
        )

    @agent.tool(strict=strict, sequential=sequential)
    def set_stacklight(ctx: RunContext[ORDeps], room_id: str, color: str, reason: str) -> dict[str, Any]:
        """Set the OR prep status stacklight.

        Green = logistics-ready, Yellow = supply deficit, Red = sterile contamination.
        """
        return ctx.deps.runtime.log_call(
            "set_stacklight",
            {"room_id": room_id, "color": color, "reason": reason},
            lambda: ctx.deps.runtime.set_stacklight(room_id, color, reason),
        )


def build_mcp_toolset(
    args: argparse.Namespace,
    MCPToolset: Any,
    StdioTransport: Any,
    case_path: Path,
    tool_log_path: Path,
) -> Any:
    mcp_args = [
        "-m",
        "agent_arena.or_mcp_tool_server",
        "--case-file",
        str(case_path),
        "--allowed-tools",
        "all",
        "--tool-log",
        str(tool_log_path),
    ]
    if args.tool_hints:
        mcp_args.append("--tool-hints")
    if args.tool_guardrails:
        mcp_args.append("--tool-guardrails")
    mcp_args.extend(["--instruction-style", args.instruction_style])
    return MCPToolset(
        StdioTransport(
            sys.executable,
            args=mcp_args,
            cwd=str(Path.cwd()),
        ),
        init_timeout=args.mcp_init_timeout,
        read_timeout=args.mcp_read_timeout,
        max_retries=args.mcp_tool_retries,
        include_instructions=args.include_mcp_instructions,
    )


def extract_message_tool_calls(messages: list[str]) -> int:
    return sum(message.count("ToolCallPart(") + message.count("tool-call") for message in messages)


def telemetry(
    score: dict[str, Any],
    exception: str,
    server_requests: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    messages: list[str],
) -> dict[str, Any]:
    raw_protocol_valid = all(item.get("parsed_ok") for item in server_requests) if server_requests else not exception
    accepted_tool_calls = sum(1 for item in tool_calls if item.get("ok", True))
    rejected_tool_calls = sum(1 for item in tool_calls if not item.get("ok", True))
    attempted = (
        any(item.get("finish_reason") == "tool_calls" for item in server_requests)
        if server_requests
        else extract_message_tool_calls(messages) > 0 or bool(tool_calls)
    )
    return {
        "final_pass": score["passed"],
        "raw_protocol_valid": raw_protocol_valid,
        "tool_call_attempted": attempted,
        "tool_call_executed": accepted_tool_calls > 0,
        "model_timeout": any(item.get("timed_out") for item in server_requests),
        "parse_failures": sum(1 for item in server_requests if not item.get("parsed_ok")),
        "model_requests": len(server_requests),
        "or_tool_calls": len(tool_calls),
        "agent_exception": bool(exception),
        "accepted_tool_calls": accepted_tool_calls,
        "rejected_tool_calls": rejected_tool_calls,
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
    if not facts["tool_call_attempted"]:
        return "no_tool_call"
    if facts["tool_call_attempted"] and not facts["tool_call_executed"]:
        return "tool_not_executed"
    return "wrong_tool_behavior"


async def run_case_async(
    args: argparse.Namespace,
    arena_case: dict[str, Any],
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

    with tempfile.TemporaryDirectory(prefix=f"or_arena__{arena_case['id']}__") as temp:
        temp_root = Path(temp)
        case_path = temp_root / "case.json"
        tool_log_path = temp_root / "or_tool_calls.jsonl"
        case_path.write_text(json.dumps(arena_case))
        runtime = ORRuntime(
            arena_case=arena_case,
            include_hints=args.tool_hints,
            enforce_guardrails=args.tool_guardrails,
            log_path=None,
        )
        deps = ORDeps(runtime=runtime)

        model = build_or_model(args, OpenAIChatModel, OpenAIProvider)
        toolsets = []
        if args.transport == "mcp":
            assert MCPToolset is not None and StdioTransport is not None
            toolsets.append(build_mcp_toolset(args, MCPToolset, StdioTransport, case_path, tool_log_path))

        base_instructions = or_agent_instructions(args.instruction_style)
        agent_kwargs: dict[str, Any] = {
            "instructions": mode_instruction_prefix(args.mode) + base_instructions,
            "toolsets": toolsets,
            "retries": args.agent_retries,
        }
        if ModelSettings is not None:
            settings: dict[str, Any] = {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            if args.parallel_tool_calls is not None:
                settings["parallel_tool_calls"] = args.parallel_tool_calls
            if args.tool_choice:
                settings["tool_choice"] = args.tool_choice
            agent_kwargs["model_settings"] = ModelSettings(**settings)
        if args.transport == "function":
            agent_kwargs["deps_type"] = ORDeps
        agent = Agent(model, **agent_kwargs)
        if args.transport == "function":
            register_function_tools(
                agent,
                strict=args.strict_tools,
                sequential=args.sequential_tools,
            )

        resources = {
            key: value
            for key, value in runtime.resources.items()
            if key != "cloud_connected"
        }
        user_prompt = json.dumps(
            {
                "event": arena_case["event"],
                "resources": resources,
            },
            indent=2,
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
            tool_calls = runtime.tool_log

    elapsed_s = time.monotonic() - started
    server_requests = get_server_log(args.base_url) if server_debug_log_enabled(args) else []
    score = score_or_tool_calls(arena_case, tool_calls, accepted_only=args.score_accepted_only)
    facts = telemetry(score, exception, server_requests, tool_calls, messages)
    parser_modes = compact_values([str(item.get("parser", "")) for item in server_requests])
    parse_statuses = compact_values([str(item.get("parse_status", "")) for item in server_requests])
    executed_tools = compact_values([str(item.get("name", "")) for item in tool_calls])
    record = {
        "model": args.model_label,
        "mode": args.mode,
        "client": "pydantic_ai_or_style",
        "provider": args.provider,
        "transport": args.transport,
        "instruction_style": args.instruction_style,
        "include_mcp_instructions": args.include_mcp_instructions,
        "mcp_tool_retries": args.mcp_tool_retries,
        "scoring_mode": score["scoring"],
        "tool_hints": args.tool_hints,
        "tool_guardrails": args.tool_guardrails,
        "case_id": arena_case["id"],
        "level": arena_case["level"],
        "group": arena_case["group"],
        "description": arena_case["description"],
        "event": arena_case["event"],
        "output": output,
        "score": score,
        "telemetry": facts,
        "failure_kind": classify_failure(score, facts, exception),
        "exception": exception,
        "elapsed_s": round(elapsed_s, 3),
        "or_tool_calls": tool_calls,
        "server_requests": server_requests,
        "messages": messages,
        "notes": (
            f"level={arena_case['level']},"
            f"group={arena_case['group']},"
            f"model_requests={facts['model_requests']},"
            f"or_tool_calls={facts['or_tool_calls']},"
            f"parse_failures={facts['parse_failures']},"
            f"parser={parser_modes},"
            f"parse_status={parse_statuses},"
            f"executed_tools={executed_tools},"
            f"instructions={args.instruction_style},"
            f"include_mcp_instructions={args.include_mcp_instructions},"
            f"scoring={score['scoring']},"
            f"tool_hints={args.tool_hints},"
            f"tool_guardrails={args.tool_guardrails},"
            f"transport={args.transport}"
        ),
    }
    (result_dir / f"{args.model_label}__{args.mode}__{arena_case['id']}.json").write_text(
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
    parser.add_argument("--mode", default="stock")
    parser.add_argument("--transport", choices=["function", "mcp"], default="function")
    parser.add_argument("--case-ids")
    parser.add_argument("--groups", default="or_benchmark,or_scenario")
    parser.add_argument("--max-level", type=int)
    parser.add_argument("--agent-retries", type=int, default=1)
    parser.add_argument("--mcp-tool-retries", type=int, default=0)
    parser.add_argument("--mcp-init-timeout", type=float, default=10)
    parser.add_argument("--mcp-read-timeout", type=float, default=300)
    parser.add_argument("--include-mcp-instructions", action="store_true")
    parser.add_argument("--instruction-style", choices=sorted(OR_AGENT_INSTRUCTION_STYLES), default="legacy")
    parser.add_argument(
        "--score-accepted-only",
        action="store_true",
        help="Legacy diagnostic mode: score only accepted tool calls instead of every attempted call.",
    )
    parser.add_argument("--tool-hints", action="store_true")
    parser.add_argument("--tool-guardrails", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--strict-tools", action="store_true")
    parser.add_argument("--sequential-tools", action="store_true")
    parser.add_argument("--parallel-tool-calls", choices=["true", "false"])
    parser.add_argument("--tool-choice", choices=["auto", "required", "none"])
    parser.add_argument("--openai-strict-tools", action="store_true")
    parser.add_argument("--no-openai-tool-choice-required", dest="openai_tool_choice_required", action="store_false")
    parser.set_defaults(openai_tool_choice_required=True)
    parser.add_argument("--out-root", type=Path, default=Path.home() / "agent_arena_results")
    parser.add_argument("--no-server-debug-log", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    if args.parallel_tool_calls is not None:
        args.parallel_tool_calls = args.parallel_tool_calls == "true"
    return args


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = {item.strip() for item in args.case_ids.split(",")} if args.case_ids else None
    groups = {item.strip() for item in args.groups.split(",") if item.strip()} if args.groups else set()
    cases = []
    for arena_case in OR_ARENA_CASES:
        if wanted is not None and arena_case["id"] not in wanted:
            continue
        if groups and arena_case["group"] not in groups:
            continue
        if args.max_level is not None and int(arena_case["level"]) > args.max_level:
            continue
        cases.append(arena_case)
    return cases


async def amain() -> int:
    args = parse_args()
    if args.list_cases:
        for arena_case in OR_ARENA_CASES:
            print(
                f"{arena_case['id']}\tlevel={arena_case['level']}\t"
                f"group={arena_case['group']}\tcase={arena_case['event']['case_id']}\t"
                f"sterile={arena_case.get('sterile_zone_issue', False)}"
            )
        return 0
    apply_env_defaults(args)
    Agent, OpenAIChatModel, OpenAIProvider = require_pydantic_ai()
    MCPToolset = None
    StdioTransport = None
    if args.transport == "mcp":
        MCPToolset, StdioTransport = require_mcp()
    cases = select_cases(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = (
        args.out_root.expanduser()
        / f"{timestamp}__pydantic_or_arena__{args.model_label}__{args.mode}__{args.provider}__{args.transport}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    results = [
        await run_case_async(
            args,
            arena_case,
            Agent,
            OpenAIChatModel,
            OpenAIProvider,
            result_dir,
            MCPToolset,
            StdioTransport,
        )
        for arena_case in cases
    ]
    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(result_dir / "summary.md", "Pydantic AI OR-Style Agent Arena", results)
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())
    if server_debug_log_enabled(args):
        print(f"SERVER_DEBUG_ROOT={root_from_base_url(args.base_url)}/debug/requests")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
