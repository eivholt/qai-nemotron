"""Run a console shipping coordinator through an OpenAI-compatible model."""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from shipping_agent.runtime import SCENARIOS, ShippingRuntime, run_scripted


INSTRUCTIONS = """You are an autonomous shipping coordinator.

Complete the user's task by choosing and calling the available tools. Treat tool
results as the only source of operational truth. Work one step at a time: in each
response, call exactly one listed tool, then stop and wait for its result before
deciding the next action. Never invent tool names, identifiers, or facts, and do
not repeat successful work. When the shipment has a safe recorded disposition and
dispatch has been notified, finish with a concise summary of what you completed.
"""


@dataclass
class ShippingDeps:
    runtime: ShippingRuntime


def build_model(base_url: str, api_key: str, model_name: str) -> Any:
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.profiles.openai import OpenAIModelProfile
        from pydantic_ai.providers.openai import OpenAIProvider
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Pydantic AI is not installed. Run: "
            "python3 -m pip install -r shipping_agent/requirements.txt"
        ) from exc

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        profile=OpenAIModelProfile(
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=True,
        ),
    )


def model_settings() -> Any:
    from pydantic_ai import ModelSettings

    return ModelSettings(
        temperature=0.0,
        max_tokens=512,
        parallel_tool_calls=False,
    )


def build_direct_agent(base_url: str, api_key: str, model_name: str) -> Any:
    try:
        from pydantic_ai import Agent, RunContext
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Pydantic AI is not installed. Run: "
            "python3 -m pip install -r shipping_agent/requirements.txt"
        ) from exc

    agent = Agent(
        build_model(base_url, api_key, model_name),
        instructions=INSTRUCTIONS,
        deps_type=ShippingDeps,
        retries=0,
        model_settings=model_settings(),
    )

    @agent.tool(sequential=True)
    def get_pending_shipment(ctx: RunContext[ShippingDeps]) -> dict[str, Any]:
        """Return the pending shipment, including ID, load, handling, and deadline."""
        runtime = ctx.deps.runtime
        return runtime.call("get_pending_shipment", {}, runtime.get_pending_shipment)

    @agent.tool(sequential=True)
    def get_shipping_options(ctx: RunContext[ShippingDeps]) -> dict[str, Any]:
        """Group current-shipment options into usable, blocked, and excluded lists."""
        runtime = ctx.deps.runtime
        shipment_id = runtime.shipment["shipment_id"]
        return runtime.call(
            "get_shipping_options",
            {},
            lambda: runtime.get_shipping_options(shipment_id),
        )

    @agent.tool(sequential=True)
    def schedule_shipment(
        ctx: RunContext[ShippingDeps],
        carrier_id: str,
        dock_id: str,
    ) -> dict[str, Any]:
        """Schedule IDs from usable_carriers and usable_docks; never use other lists."""
        runtime = ctx.deps.runtime
        shipment_id = runtime.shipment["shipment_id"]
        arguments = {"carrier_id": carrier_id, "dock_id": dock_id}
        return runtime.call(
            "schedule_shipment",
            arguments,
            lambda: runtime.schedule_shipment(shipment_id, carrier_id, dock_id),
        )

    @agent.tool(sequential=True)
    def hold_shipment(
        ctx: RunContext[ShippingDeps],
        reason: str,
    ) -> dict[str, Any]:
        """Temporarily hold only when an otherwise usable carrier is expected to recover soon."""
        runtime = ctx.deps.runtime
        shipment_id = runtime.shipment["shipment_id"]
        arguments = {"reason": reason}
        return runtime.call(
            "hold_shipment",
            arguments,
            lambda: runtime.hold_shipment(shipment_id, reason),
        )

    @agent.tool(sequential=True)
    def escalate_shipment(
        ctx: RunContext[ShippingDeps],
        reason: str,
    ) -> dict[str, Any]:
        """Escalate when no compatible carrier is available now or expected to recover."""
        runtime = ctx.deps.runtime
        shipment_id = runtime.shipment["shipment_id"]
        arguments = {"reason": reason}
        return runtime.call(
            "escalate_shipment",
            arguments,
            lambda: runtime.escalate_shipment(shipment_id, reason),
        )

    @agent.tool(sequential=True)
    def notify_dispatch(ctx: RunContext[ShippingDeps]) -> dict[str, Any]:
        """Notify dispatch after the current shipment is scheduled, held, or escalated."""
        runtime = ctx.deps.runtime
        shipment_id = runtime.shipment["shipment_id"]
        return runtime.call(
            "notify_dispatch",
            {},
            lambda: runtime.notify_dispatch(shipment_id),
        )

    return agent


def print_scenario_heading(runtime: ShippingRuntime, name: str, run_number: int, transport: str) -> None:
    print(
        f"\n=== run {run_number} / {name} / {transport}: "
        f"{runtime.state['title']} ==="
    )
    print(f"TASK         {runtime.state['task']}")


def run_direct_scenario(args: argparse.Namespace, name: str, run_number: int) -> bool:
    from pydantic_ai import UsageLimits

    runtime = ShippingRuntime.from_scenario(name, verbose=True)
    print_scenario_heading(runtime, name, run_number, "direct")
    agent = build_direct_agent(args.base_url, args.api_key, args.model)
    result = agent.run_sync(
        runtime.state["task"],
        deps=ShippingDeps(runtime),
        usage_limits=UsageLimits(request_limit=args.request_limit),
    )
    print(f"FINAL ANSWER {result.output}")
    summary = runtime.summary()
    print(f"FINAL STATE  {json.dumps(summary, sort_keys=True)}")
    return bool(summary["passed"])


def normalize_mcp_tool_schemas(_ctx: Any, tool_defs: list[Any]) -> list[Any]:
    """Match MCP-discovered schemas to Pydantic's concise direct-tool shape."""

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            result = {
                key: normalize(item)
                for key, item in value.items()
                if key != "title"
            }
            if result.get("type") == "object":
                result.setdefault("additionalProperties", False)
            return result
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return [
        replace(
            tool_def,
            parameters_json_schema=normalize(
                deepcopy(tool_def.parameters_json_schema)
            ),
        )
        for tool_def in tool_defs
    ]


async def run_mcp_scenario_async(
    args: argparse.Namespace,
    name: str,
    run_number: int,
) -> bool:
    try:
        from pydantic_ai import Agent, UsageLimits
        from pydantic_ai.mcp import MCPToolset, StdioTransport
        from pydantic_ai.toolsets import PreparedToolset
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "MCP support is not installed. Run: "
            "python3 -m pip install -r shipping_agent/requirements.txt"
        ) from exc

    display_runtime = ShippingRuntime.from_scenario(name, verbose=False)
    print_scenario_heading(display_runtime, name, run_number, "mcp")

    with tempfile.TemporaryDirectory(prefix=f"shipping_mcp__{name}__") as temp:
        state_file = Path(temp) / "state.json"
        project_root = Path(__file__).resolve().parents[1]
        discovered_toolset = MCPToolset(
            StdioTransport(
                sys.executable,
                args=[
                    "-m",
                    "shipping_agent.mcp_server",
                    "--scenario",
                    name,
                    "--state-file",
                    str(state_file),
                ],
                cwd=str(project_root),
            ),
            init_timeout=args.mcp_init_timeout,
            read_timeout=args.mcp_read_timeout,
            max_retries=0,
            include_instructions=False,
        )
        toolset = PreparedToolset(
            discovered_toolset,
            normalize_mcp_tool_schemas,
        )
        agent = Agent(
            build_model(args.base_url, args.api_key, args.model),
            instructions=INSTRUCTIONS,
            toolsets=[toolset],
            retries=0,
            model_settings=model_settings(),
        )
        async with agent:
            result = await agent.run(
                display_runtime.state["task"],
                usage_limits=UsageLimits(request_limit=args.request_limit),
            )

        snapshot = json.loads(state_file.read_text())
        for call in snapshot["calls"]:
            print(
                "TOOL CALL    "
                + json.dumps(
                    {"name": call["name"], "arguments": call["arguments"]},
                    sort_keys=True,
                )
            )
            print("TOOL RESULT  " + json.dumps(call["result"], sort_keys=True))
        print(f"FINAL ANSWER {result.output}")
        summary = snapshot["summary"]
        print(f"FINAL STATE  {json.dumps(summary, sort_keys=True)}")
        return bool(summary["passed"])


def run_agent_scenario(args: argparse.Namespace, name: str, run_number: int) -> bool:
    if args.transport == "mcp":
        return asyncio.run(run_mcp_scenario_async(args, name, run_number))
    return run_direct_scenario(args, name, run_number)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="routine", choices=[*SCENARIOS, "all"])
    parser.add_argument("--transport", choices=["direct", "mcp"], default="direct")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--scripted", action="store_true", help="Test mock tools without an LLM")
    parser.add_argument(
        "--base-url",
        default=os.getenv("SHIPPING_AGENT_BASE_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("SHIPPING_AGENT_API_KEY", "local"))
    parser.add_argument("--model", default=os.getenv("SHIPPING_AGENT_MODEL", "ministral3-3b-q4"))
    parser.add_argument("--request-limit", type=int, default=10)
    parser.add_argument("--mcp-init-timeout", type=float, default=10.0)
    parser.add_argument("--mcp-read-timeout", type=float, default=300.0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each selected scenario to measure reliability",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_scenarios:
        for name, item in SCENARIOS.items():
            print(f"{name:<24} {item['title']}")
        return 0

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    results: list[dict[str, Any]] = []
    for run_number in range(1, args.repeat + 1):
        for name in names:
            try:
                if args.scripted:
                    print(f"\n=== run {run_number} / {name}: scripted mock-tool check ===")
                    runtime = run_scripted(name, verbose=True)
                    summary = runtime.summary()
                    print(f"FINAL STATE  {json.dumps(summary, sort_keys=True)}")
                    passed = bool(summary["passed"])
                else:
                    passed = run_agent_scenario(args, name, run_number)
                results.append(
                    {
                        "run": run_number,
                        "scenario": name,
                        "transport": args.transport,
                        "passed": passed,
                    }
                )
            except Exception as exc:
                print(f"RUN ERROR    {type(exc).__name__}: {exc}")
                results.append(
                    {
                        "run": run_number,
                        "scenario": name,
                        "transport": args.transport,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    passed_count = sum(bool(item["passed"]) for item in results)
    suite = {"passed": passed_count, "total": len(results), "results": results}
    print(f"\nSUITE SUMMARY {json.dumps(suite, sort_keys=True)}")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
