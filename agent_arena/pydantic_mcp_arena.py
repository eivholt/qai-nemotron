#!/usr/bin/env python3
"""Run the arena through Pydantic AI with real MCP tool servers."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_arena.model_client import score_text, write_summary
from agent_arena.pydantic_arena import (
    get_server_log,
    require_pydantic_ai,
    reset_server_log,
    root_from_base_url,
)
from agent_arena.tool_arena import TOOL_CASES


AGENT_INSTRUCTIONS = (
    "You are a practical Linux/HTTP/data agent. Use the available MCP tools "
    "when needed. Do not guess values that should come from files, HTTP fixtures, "
    "CSV, JSON, logs, or calculation. Return only the final answer in the format "
    "requested by the user."
)


CASE_TOOLSETS: dict[str, list[str]] = {
    "tool_00_direct_final_protocol": [],
    "tool_01_single_json_lookup": ["lookup_json_record", "json_query", "read_file"],
    "tool_02_multi_step_inventory": ["csv_max_difference", "read_file", "list_files"],
    "tool_03_fixture_http": ["http_get"],
    "tool_04_context_breaking_log": ["search_text", "read_file"],
    "tool_05_calculator_then_final": ["calculator"],
    "tool_06_small_file_read": ["read_file"],
    "tool_07_easy_calculator_add": ["calculator"],
    "tool_08_easy_read_color": ["read_file"],
    "tool_09_easy_json_query_name": ["json_query", "read_file"],
    "tool_10_easy_http_ping": ["http_get"],
    "tool_11_easy_short_search": ["search_text", "read_file"],
    "tool_12_easy_list_files": ["list_files"],
}


def require_mcp() -> tuple[Any, Any]:
    try:
        from pydantic_ai.mcp import MCPToolset, StdioTransport
    except ImportError as exc:
        raise SystemExit(
            "Missing Pydantic AI MCP dependency. Install on the host with:\n"
            '  python -m pip install "pydantic-ai-slim[mcp]"\n'
            f"Original import error: {exc}"
        ) from exc
    return MCPToolset, StdioTransport


def tool_names_for_case(case: dict[str, Any], pruning: str) -> list[str]:
    if pruning == "none":
        return [
            "lookup_json_record",
            "csv_max_difference",
            "list_files",
            "read_file",
            "search_text",
            "json_query",
            "http_get",
            "calculator",
        ]
    return CASE_TOOLSETS.get(case["id"], [])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"ok": False, "error": f"invalid jsonl: {line[:200]}"})
    return rows


def compact_values(values: list[str], default: str = "none") -> str:
    clean = sorted({value for value in values if value})
    return ",".join(clean) if clean else default


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def apply_env_defaults(args: argparse.Namespace) -> None:
    load_env_file(args.env_file.expanduser())
    args.provider = args.provider or first_env("AGENT_ARENA_PROVIDER") or "openai-compatible"
    args.base_url = args.base_url or first_env("AGENT_ARENA_BASE_URL", "OPENAI_BASE_URL")
    args.api_key = args.api_key or first_env(
        "AGENT_ARENA_API_KEY",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_API_KEY",
    )
    args.azure_endpoint = args.azure_endpoint or first_env("AZURE_OPENAI_ENDPOINT")
    args.azure_api_version = args.azure_api_version or first_env("AZURE_OPENAI_API_VERSION")
    args.anthropic_foundry_base_url = args.anthropic_foundry_base_url or first_env(
        "ANTHROPIC_FOUNDRY_BASE_URL"
    )
    args.model_name = args.model_name or first_env(
        "AGENT_ARENA_MODEL_NAME",
        "AZURE_OPENAI_DEPLOYMENT",
        "ANTHROPIC_MODEL",
        "OPENAI_MODEL",
    )
    args.model_label = args.model_label or first_env("AGENT_ARENA_MODEL_LABEL")
    if not args.model_label and args.model_name:
        args.model_label = args.model_name.replace("/", "_").replace(":", "_")
    if args.provider == "azure" and args.model_name.lower().startswith("claude-"):
        args.provider = "azure-foundry-anthropic"
    if args.provider == "azure":
        missing = [
            name
            for name, value in [
                ("AZURE_OPENAI_ENDPOINT", args.azure_endpoint),
                ("AZURE_OPENAI_API_VERSION", args.azure_api_version),
                ("AZURE_OPENAI_API_KEY or AGENT_ARENA_API_KEY", args.api_key),
                ("AZURE_OPENAI_DEPLOYMENT or AGENT_ARENA_MODEL_NAME", args.model_name),
            ]
            if not value
        ]
        if missing:
            raise SystemExit(f"Missing Azure configuration: {', '.join(missing)}")
        return
    if args.provider == "azure-foundry-anthropic":
        if not args.anthropic_foundry_base_url and args.azure_endpoint:
            hostname = urlparse(args.azure_endpoint).hostname or ""
            resource = hostname.split(".")[0] if hostname else ""
            if resource:
                args.anthropic_foundry_base_url = (
                    f"https://{resource}.services.ai.azure.com/anthropic/"
                )
        missing = [
            name
            for name, value in [
                (
                    "ANTHROPIC_FOUNDRY_BASE_URL or AZURE_OPENAI_ENDPOINT",
                    args.anthropic_foundry_base_url,
                ),
                (
                    "ANTHROPIC_FOUNDRY_API_KEY, AZURE_OPENAI_API_KEY, or AGENT_ARENA_API_KEY",
                    args.api_key,
                ),
                ("AZURE_OPENAI_DEPLOYMENT or AGENT_ARENA_MODEL_NAME", args.model_name),
            ]
            if not value
        ]
        if missing:
            raise SystemExit(f"Missing Azure Foundry Anthropic configuration: {', '.join(missing)}")
        return
    if args.provider == "anthropic":
        missing = [
            name
            for name, value in [
                ("ANTHROPIC_API_KEY or AGENT_ARENA_API_KEY", args.api_key),
                ("ANTHROPIC_MODEL or AGENT_ARENA_MODEL_NAME", args.model_name),
            ]
            if not value
        ]
        if missing:
            raise SystemExit(f"Missing Anthropic configuration: {', '.join(missing)}")
        return
    if args.provider == "openai-compatible":
        args.api_key = args.api_key or "agent-arena"
        missing = [
            name
            for name, value in [
                ("AGENT_ARENA_BASE_URL or OPENAI_BASE_URL", args.base_url),
                ("AGENT_ARENA_MODEL_NAME or OPENAI_MODEL", args.model_name),
            ]
            if not value
        ]
        if missing:
            raise SystemExit(f"Missing OpenAI-compatible configuration: {', '.join(missing)}")
        return
    raise SystemExit(f"Unsupported provider: {args.provider}")


def build_provider(args: argparse.Namespace, OpenAIProvider: Any) -> Any:
    if args.provider == "azure":
        try:
            from pydantic_ai.providers.azure import AzureProvider
        except ImportError as exc:
            raise SystemExit(f"Missing Pydantic AI Azure provider: {exc}") from exc
        return AzureProvider(
            azure_endpoint=args.azure_endpoint,
            api_version=args.azure_api_version,
            api_key=args.api_key,
        )
    return OpenAIProvider(base_url=args.base_url, api_key=args.api_key)


def build_model(args: argparse.Namespace, OpenAIChatModel: Any, OpenAIProvider: Any) -> Any:
    if args.provider == "azure-foundry-anthropic":
        try:
            from anthropic import AsyncAnthropicFoundry
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider
        except ImportError as exc:
            raise SystemExit(
                "Missing Anthropic dependencies. Install on the host with:\n"
                '  python -m pip install "pydantic-ai-slim[anthropic]"\n'
                f"Original import error: {exc}"
            ) from exc
        client = AsyncAnthropicFoundry(
            api_key=args.api_key,
            base_url=args.anthropic_foundry_base_url,
        )
        return AnthropicModel(
            args.model_name,
            provider=AnthropicProvider(anthropic_client=client),
        )
    if args.provider == "anthropic":
        try:
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider
        except ImportError as exc:
            raise SystemExit(
                "Missing Anthropic dependencies. Install on the host with:\n"
                '  python -m pip install "pydantic-ai-slim[anthropic]"\n'
                f"Original import error: {exc}"
            ) from exc
        return AnthropicModel(
            args.model_name,
            provider=AnthropicProvider(api_key=args.api_key),
        )
    return OpenAIChatModel(
        args.model_name,
        provider=build_provider(args, OpenAIProvider),
    )


def server_debug_log_enabled(args: argparse.Namespace) -> bool:
    return args.provider == "openai-compatible" and not args.no_server_debug_log


def summarize_messages(result: Any) -> list[str]:
    try:
        return [repr(message) for message in result.all_messages()]
    except Exception:
        return []


def telemetry(score: dict[str, Any], exception: str, server_requests: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    raw_protocol_valid = all(item.get("parsed_ok") for item in server_requests) if server_requests else not exception
    tool_call_attempted = (
        any(item.get("finish_reason") == "tool_calls" for item in server_requests)
        if server_requests
        else bool(tool_calls)
    )
    tool_call_executed = bool(tool_calls)
    model_timeout = any(item.get("timed_out") for item in server_requests)
    empty_answer = any(item.get("parse_status") == "final" and not str(item.get("final_answer", "")).strip() for item in server_requests)
    placeholder_answer = any(
        "<" in str(item.get("raw_answer", "")) or "placeholder" in str(item.get("raw_answer", "")).lower()
        for item in server_requests
    )
    return {
        "final_pass": score["passed"],
        "raw_protocol_valid": raw_protocol_valid,
        "tool_call_attempted": tool_call_attempted,
        "tool_call_executed": tool_call_executed,
        "client_repaired": False,
        "model_timeout": model_timeout,
        "empty_answer": empty_answer,
        "placeholder_answer": placeholder_answer,
        "parse_failures": sum(1 for item in server_requests if not item.get("parsed_ok")),
        "model_requests": len(server_requests),
        "mcp_tool_calls": len(tool_calls),
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


async def run_case_async(
    args: argparse.Namespace,
    case: dict[str, Any],
    Agent: Any,
    OpenAIChatModel: Any,
    OpenAIProvider: Any,
    MCPToolset: Any,
    StdioTransport: Any,
    result_dir: Path,
) -> dict[str, Any]:
    if server_debug_log_enabled(args):
        reset_server_log(args.base_url)
    output = ""
    exception = ""
    messages: list[str] = []
    started = time.monotonic()
    allowed_tools = tool_names_for_case(case, args.tool_pruning)

    with tempfile.TemporaryDirectory(prefix=f"pydantic_mcp__{case['id']}__") as temp:
        temp_root = Path(temp)
        case_path = temp_root / "case.json"
        tool_log_path = temp_root / "mcp_tool_calls.jsonl"
        case_path.write_text(json.dumps(case))
        toolsets = []
        if allowed_tools:
            toolsets.append(
                MCPToolset(
                    StdioTransport(
                        sys.executable,
                        args=[
                            "-m",
                            "agent_arena.mcp_tool_server",
                            "--case-file",
                            str(case_path),
                            "--root",
                            str(temp_root / "sandbox"),
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
            )

        model = build_model(args, OpenAIChatModel, OpenAIProvider)
        agent = Agent(
            model,
            instructions=AGENT_INSTRUCTIONS,
            toolsets=toolsets,
            retries=args.agent_retries,
        )

        try:
            async with agent:
                result = await agent.run(case["goal"])
            output = str(result.output)
            messages = summarize_messages(result)
        except Exception as exc:
            exception = repr(exc)
        tool_calls = read_jsonl(tool_log_path)

    elapsed_s = time.monotonic() - started
    server_requests = get_server_log(args.base_url) if server_debug_log_enabled(args) else []
    score = score_text(output, case["required_regex"])
    facts = telemetry(score, exception, server_requests, tool_calls)
    parser_modes = compact_values([str(item.get("parser", "")) for item in server_requests])
    parse_statuses = compact_values([str(item.get("parse_status", "")) for item in server_requests])
    parsed_tools = compact_values(
        [
            str(item.get("parsed_action", {}).get("name", ""))
            for item in server_requests
            if isinstance(item.get("parsed_action"), dict)
        ]
    )
    executed_tools = compact_values([str(item.get("name", "")) for item in tool_calls])
    record = {
        "model": args.model_label,
        "mode": args.mode,
        "client": "pydantic_ai_mcp",
        "provider": args.provider,
        "tool_transport": "mcp_stdio",
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
        "mcp_tool_calls": tool_calls,
        "server_requests": server_requests,
        "messages": messages,
        "notes": (
            f"model_requests={facts['model_requests']},"
            f"mcp_tool_calls={facts['mcp_tool_calls']},"
            f"parse_failures={facts['parse_failures']},"
            f"parser={parser_modes},"
            f"parse_status={parse_statuses},"
            f"parsed_tools={parsed_tools},"
            f"executed_tools={executed_tools},"
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
    parser.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://evk:8001/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--azure-endpoint")
    parser.add_argument("--azure-api-version")
    parser.add_argument("--anthropic-foundry-base-url")
    parser.add_argument("--model-name")
    parser.add_argument("--model-label")
    parser.add_argument("--mode", choices=["stock", "thinking_off", "thinking_on"], default="thinking_off")
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
            print(f"{case['id']}\tdifficulty={case['difficulty']}\ttools={','.join(tool_names_for_case(case, 'case'))}")
        return 0
    apply_env_defaults(args)
    Agent, OpenAIChatModel, OpenAIProvider = require_pydantic_ai()
    MCPToolset, StdioTransport = require_mcp()
    wanted = {item.strip() for item in args.case_ids.split(",")} if args.case_ids else None
    cases = [case for case in TOOL_CASES if wanted is None or case["id"] in wanted]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.out_root.expanduser() / f"{timestamp}__pydantic_mcp_arena__{args.model_label}__{args.mode}__{args.provider}__{args.tool_pruning}"
    result_dir.mkdir(parents=True, exist_ok=True)
    results = [
        await run_case_async(args, case, Agent, OpenAIChatModel, OpenAIProvider, MCPToolset, StdioTransport, result_dir)
        for case in cases
    ]
    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(result_dir / "summary.md", "Pydantic AI MCP Agent Arena", results)
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())
    if server_debug_log_enabled(args):
        print(f"SERVER_DEBUG_ROOT={root_from_base_url(args.base_url)}/debug/requests")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
