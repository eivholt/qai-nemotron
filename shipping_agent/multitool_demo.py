"""Probe whether an agent batches independent shipping observations."""

import argparse
import copy
import json
import os
from collections import Counter
from typing import Any

from shipping_agent.app import ShippingDeps, build_model
from shipping_agent.runtime import ShippingRuntime


MULTI_TOOL_INSTRUCTIONS = """You are an autonomous shipping coordinator.

Complete the user's task by choosing and calling the available tools. Treat tool
results as the only source of operational truth. Request independent read-only
observations together in one response when it is safe to do so. Never batch a
state-changing call with observations or other work whose result it depends on.
After the observations return, choose and record one safe disposition. Wait for
that result, notify dispatch, and then finish with a concise summary. Never invent
tool names, identifiers, or facts, and do not repeat successful work.
"""


MULTI_TOOL_SCENARIO: dict[str, Any] = {
    "title": "Parallel observation of a cold-chain shipment",
    "task": (
        "Plan the pending shipment with a compatible carrier and dock, record "
        "the disposition, and notify dispatch. Batch independent observations "
        "when safe."
    ),
    "shipment": {
        "shipment_id": "SHP-5001",
        "destination": "Regional pharmacy",
        "pallets": 3,
        "handling": "cold_chain",
        "temperature_c": "2-8",
        "deadline_minutes": 75,
        "status": "pending",
    },
    "carriers": [
        {
            "carrier_id": "TRUCK-7",
            "capacity_pallets": 8,
            "handling": ["standard"],
            "travel_minutes": 30,
            "route_open": True,
        },
        {
            "carrier_id": "REEFER-2",
            "capacity_pallets": 6,
            "handling": ["standard", "cold_chain"],
            "travel_minutes": 40,
            "route_open": True,
        },
    ],
    "docks": [
        {"dock_id": "D1", "available": True, "handling": ["standard"]},
        {"dock_id": "COLD-1", "available": True, "handling": ["cold_chain"]},
    ],
    "expected": {
        "action": "scheduled",
        "carrier_id": "REEFER-2",
        "dock_id": "COLD-1",
    },
}

READ_TOOLS = {
    "get_pending_shipment",
    "get_carrier_options",
    "get_dock_options",
}
WRITE_TOOLS = {
    "schedule_shipment",
    "hold_shipment",
    "escalate_shipment",
    "notify_dispatch",
}
EXPECTED_CALLS = [
    "get_pending_shipment",
    "get_carrier_options",
    "get_dock_options",
    "schedule_shipment",
    "notify_dispatch",
]


def model_settings() -> Any:
    from pydantic_ai import ModelSettings

    return ModelSettings(
        temperature=0.0,
        max_tokens=512,
        parallel_tool_calls=True,
    )


def build_multitool_agent(base_url: str, api_key: str, model_name: str) -> Any:
    try:
        from pydantic_ai import Agent, RunContext
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Pydantic AI is not installed. Run: "
            "python3 -m pip install -r shipping_agent/requirements.txt"
        ) from exc

    agent = Agent(
        build_model(base_url, api_key, model_name),
        instructions=MULTI_TOOL_INSTRUCTIONS,
        deps_type=ShippingDeps,
        retries=0,
        model_settings=model_settings(),
    )

    @agent.tool
    def get_pending_shipment(ctx: RunContext[ShippingDeps]) -> dict[str, Any]:
        """Read the pending shipment, including load, handling, and deadline."""
        runtime = ctx.deps.runtime
        return runtime.call("get_pending_shipment", {}, runtime.get_pending_shipment)

    @agent.tool
    def get_carrier_options(ctx: RunContext[ShippingDeps]) -> dict[str, Any]:
        """Assess every carrier for the current shipment without changing state."""
        runtime = ctx.deps.runtime
        return runtime.call(
            "get_carrier_options",
            {},
            runtime.get_carrier_options,
        )

    @agent.tool
    def get_dock_options(ctx: RunContext[ShippingDeps]) -> dict[str, Any]:
        """Assess every loading dock for the current shipment without changing state."""
        runtime = ctx.deps.runtime
        return runtime.call(
            "get_dock_options",
            {},
            runtime.get_dock_options,
        )

    @agent.tool(sequential=True)
    def schedule_shipment(
        ctx: RunContext[ShippingDeps],
        carrier_id: str,
        dock_id: str,
    ) -> dict[str, Any]:
        """Schedule IDs assessed as usable after reading shipment, carriers, and docks."""
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
        """Temporarily hold only when a usable route is expected to recover soon."""
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
        """Escalate when no safe carrier and dock combination can be scheduled."""
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
        """Notify dispatch after a safe disposition has been recorded."""
        runtime = ctx.deps.runtime
        shipment_id = runtime.shipment["shipment_id"]
        return runtime.call(
            "notify_dispatch",
            {},
            lambda: runtime.notify_dispatch(shipment_id),
        )

    return agent


def extract_tool_batches(result: Any) -> list[list[str]]:
    """Return tool calls grouped by individual assistant responses."""
    batches: list[list[str]] = []
    for message in result.all_messages():
        names = [
            part.tool_name
            for part in getattr(message, "parts", ())
            if part.__class__.__name__ == "ToolCallPart"
            and isinstance(getattr(part, "tool_name", None), str)
        ]
        if names:
            batches.append(names)
    return batches


def score_multitool(
    runtime: ShippingRuntime,
    tool_batches: list[list[str]],
) -> dict[str, Any]:
    call_names = [call["name"] for call in runtime.calls]
    tool_errors = [
        {"name": call["name"], "error": call["result"].get("error", "unknown")}
        for call in runtime.calls
        if not call["result"].get("ok", False)
    ]
    exact_calls = Counter(call_names) == Counter(EXPECTED_CALLS)
    ordered = False
    if exact_calls:
        read_indexes = [call_names.index(name) for name in READ_TOOLS]
        ordered = (
            max(read_indexes) < call_names.index("schedule_shipment")
            < call_names.index("notify_dispatch")
        )

    expected = MULTI_TOOL_SCENARIO["expected"]
    actual = {
        "action": runtime.shipment["status"],
        "carrier_id": runtime.shipment.get("carrier_id"),
        "dock_id": runtime.shipment.get("dock_id"),
    }
    state_correct = all(actual.get(key) == value for key, value in expected.items())
    task_passed = (
        runtime.notified
        and state_correct
        and exact_calls
        and ordered
        and not tool_errors
    )

    flattened_batches = [name for batch in tool_batches for name in batch]
    batches_match_calls = Counter(flattened_batches) == Counter(call_names)
    read_batch_sizes = [
        len(batch)
        for batch in tool_batches
        if batch and set(batch).issubset(READ_TOOLS)
    ]
    has_multi_read_batch = any(size >= 2 for size in read_batch_sizes)
    writes_are_separate = all(
        len(batch) == 1 for batch in tool_batches if set(batch) & WRITE_TOOLS
    )
    batching_passed = (
        batches_match_calls and has_multi_read_batch and writes_are_separate
    )

    return {
        "passed": task_passed and batching_passed,
        "task_passed": task_passed,
        "batching_passed": batching_passed,
        "multi_tool_observed": has_multi_read_batch,
        "expected": copy.deepcopy(expected),
        "actual": actual,
        "dispatch_notified": runtime.notified,
        "tool_calls": len(call_names),
        "call_names": call_names,
        "expected_calls": EXPECTED_CALLS,
        "tool_batches": tool_batches,
        "state_correct": state_correct,
        "trace_correct": exact_calls and ordered,
        "batches_match_calls": batches_match_calls,
        "writes_are_separate": writes_are_separate,
        "tool_errors": tool_errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("SHIPPING_AGENT_BASE_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("SHIPPING_AGENT_API_KEY", "local"))
    parser.add_argument("--model", default=os.getenv("SHIPPING_AGENT_MODEL", "ministral3-3b-q4"))
    parser.add_argument("--request-limit", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    from pydantic_ai import UsageLimits

    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    print("SYSTEM PROMPT")
    print(MULTI_TOOL_INSTRUCTIONS)
    results: list[dict[str, Any]] = []
    for run_number in range(1, args.repeat + 1):
        runtime = ShippingRuntime(copy.deepcopy(MULTI_TOOL_SCENARIO), verbose=True)
        print(
            f"\n=== run {run_number} / multi_tool_observation: "
            f"{runtime.state['title']} ==="
        )
        print(f"TASK         {runtime.state['task']}")
        try:
            agent = build_multitool_agent(args.base_url, args.api_key, args.model)
            result = agent.run_sync(
                runtime.state["task"],
                deps=ShippingDeps(runtime),
                usage_limits=UsageLimits(request_limit=args.request_limit),
            )
            tool_batches = extract_tool_batches(result)
            summary = score_multitool(runtime, tool_batches)
            print(f"FINAL ANSWER {result.output}")
        except Exception as exc:
            tool_batches = []
            summary = score_multitool(runtime, tool_batches)
            summary["error"] = f"{type(exc).__name__}: {exc}"
            print(f"RUN ERROR    {summary['error']}")
        print(f"TOOL BATCHES {json.dumps(tool_batches)}")
        print(f"FINAL STATE  {json.dumps(summary, sort_keys=True)}")
        results.append({"run": run_number, **summary})

    passed_count = sum(bool(item["passed"]) for item in results)
    task_passed_count = sum(bool(item["task_passed"]) for item in results)
    suite = {
        "passed": passed_count,
        "task_passed": task_passed_count,
        "total": len(results),
        "results": results,
    }
    print(f"\nSUITE SUMMARY {json.dumps(suite, sort_keys=True)}")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
