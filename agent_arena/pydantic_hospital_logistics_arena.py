#!/usr/bin/env python3
"""Run the hospital logistics edge-agent demo through Pydantic AI."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_arena.hospital_logistics_runtime import (
    HOSPITAL_LOGISTICS_INSTRUCTIONS,
    SCENARIOS,
    HospitalLogisticsRuntime,
    hospital_scenario,
    score_hospital_calls,
    HOSPITAL_TOOL_NAMES,
)
from agent_arena.model_client import write_summary
from agent_arena.pydantic_arena import get_server_log, require_pydantic_ai, reset_server_log, root_from_base_url
from agent_arena.pydantic_mcp_arena import apply_env_defaults, build_model, build_provider, compact_values, summarize_messages

try:
    from pydantic_ai import ModelSettings, RunContext, UsageLimits
    from pydantic_ai.exceptions import UsageLimitExceeded
except ModuleNotFoundError:
    ModelSettings = None  # type: ignore[assignment]
    RunContext = Any  # type: ignore[misc,assignment]
    UsageLimits = None  # type: ignore[assignment]
    UsageLimitExceeded = Exception  # type: ignore[assignment]


@dataclass
class HospitalDeps:
    runtime: HospitalLogisticsRuntime


def mode_instruction_prefix(mode: str) -> str:
    if mode == "thinking_off":
        return "detailed thinking off\nUse tools directly. Keep the final response concise.\n\n"
    if mode == "thinking_on":
        return "detailed thinking on\nUse a short reasoning budget, then act with tools. Keep the final response concise.\n\n"
    return ""


def build_hospital_model(args: argparse.Namespace, OpenAIChatModel: Any, OpenAIProvider: Any) -> Any:
    if args.provider in {"anthropic", "azure-foundry-anthropic"}:
        return build_model(args, OpenAIChatModel, OpenAIProvider)
    try:
        from pydantic_ai.profiles.openai import OpenAIModelProfile
    except ImportError:
        return OpenAIChatModel(args.model_name, provider=build_provider(args, OpenAIProvider))
    return OpenAIChatModel(
        args.model_name,
        provider=build_provider(args, OpenAIProvider),
        profile=OpenAIModelProfile(
            openai_supports_strict_tool_definition=args.openai_strict_tools,
            openai_supports_tool_choice_required=args.openai_tool_choice_required,
        ),
    )


def register_hospital_tools(agent: Any, *, strict: bool, sequential: bool, allowed_tools: list[str] | None = None) -> None:
    allowed = set(allowed_tools or HOSPITAL_TOOL_NAMES)

    def hospital_tool(name: str):
        def decorate(fn: Any) -> Any:
            if name in allowed:
                return agent.tool(strict=strict, sequential=sequential)(fn)
            return fn

        return decorate

    @hospital_tool("read_pending_job")
    def read_pending_job(ctx: RunContext[HospitalDeps]) -> dict[str, Any]:
        """Read the single pending hospital logistics job. No arguments needed."""
        return ctx.deps.runtime.log_call("read_pending_job", {}, ctx.deps.runtime.read_pending_job)

    @hospital_tool("assign_pending_porter")
    def assign_pending_porter(ctx: RunContext[HospitalDeps]) -> dict[str, Any]:
        """Assign the single pending job to an available porter. No arguments needed."""
        return ctx.deps.runtime.log_call("assign_pending_porter", {}, ctx.deps.runtime.assign_pending_porter)

    @hospital_tool("escalate_pending_job")
    def escalate_pending_job(ctx: RunContext[HospitalDeps]) -> dict[str, Any]:
        """Escalate the single pending job to a human coordinator. No arguments needed."""
        return ctx.deps.runtime.log_call("escalate_pending_job", {}, ctx.deps.runtime.escalate_pending_job)

    @hospital_tool("check_pending_cold_chain")
    def check_pending_cold_chain(ctx: RunContext[HospitalDeps]) -> dict[str, Any]:
        """Check cold-chain status for the single pending job. No arguments needed."""
        return ctx.deps.runtime.log_call("check_pending_cold_chain", {}, ctx.deps.runtime.check_pending_cold_chain)

    @hospital_tool("reserve_pending_elevator")
    def reserve_pending_elevator(ctx: RunContext[HospitalDeps]) -> dict[str, Any]:
        """Reserve the available elevator for the single pending job. No arguments needed."""
        return ctx.deps.runtime.log_call("reserve_pending_elevator", {}, ctx.deps.runtime.reserve_pending_elevator)
    @hospital_tool("get_pending_jobs")
    def get_pending_jobs(
        ctx: RunContext[HospitalDeps],
        jobs: Any = None,
        event: Any = None,
    ) -> dict[str, Any]:
        """Return the active hospital logistics jobs and the triggering event.

        Call this first. Optional jobs/event arguments are treated as client
        hints only; the tool remains the source of truth for the live queue.
        """
        args = {}
        return ctx.deps.runtime.log_call("get_pending_jobs", args, ctx.deps.runtime.get_pending_jobs)

    @hospital_tool("get_asset_location")
    def get_asset_location(ctx: RunContext[HospitalDeps], asset_id: str) -> dict[str, Any]:
        """Get the current location and readiness of a sample, tote, cart, or device."""
        return ctx.deps.runtime.log_call(
            "get_asset_location",
            {"asset_id": asset_id},
            lambda: ctx.deps.runtime.get_asset_location(asset_id),
        )

    @hospital_tool("check_elevator_status")
    def check_elevator_status(ctx: RunContext[HospitalDeps], from_unit: str = "", to_unit: str = "") -> dict[str, Any]:
        """Check which elevators can serve a route. Leave from_unit/to_unit blank to use the pending job route."""
        return ctx.deps.runtime.log_call(
            "check_elevator_status",
            {"from_unit": from_unit, "to_unit": to_unit},
            lambda: ctx.deps.runtime.check_elevator_status(from_unit, to_unit),
        )

    @hospital_tool("reserve_elevator")
    def reserve_elevator(ctx: RunContext[HospitalDeps], job_id: str = "pending", elevator_id: str = "available", duration_minutes: int = 5) -> dict[str, Any]:
        """Reserve an elevator. In single-job cases, job_id="pending" selects the pending job."""
        return ctx.deps.runtime.log_call(
            "reserve_elevator",
            {"elevator_id": elevator_id, "job_id": job_id, "duration_minutes": duration_minutes},
            lambda: ctx.deps.runtime.reserve_elevator(elevator_id, job_id, duration_minutes),
        )

    @hospital_tool("assign_porter")
    def assign_porter(ctx: RunContext[HospitalDeps], job_id: str = "pending", porter_id: str = "available") -> dict[str, Any]:
        """Assign a human porter. In single-job cases, job_id="pending" selects the pending job."""
        return ctx.deps.runtime.log_call(
            "assign_porter",
            {"porter_id": porter_id, "job_id": job_id},
            lambda: ctx.deps.runtime.assign_porter(porter_id, job_id),
        )

    @hospital_tool("assign_robot")
    def assign_robot(ctx: RunContext[HospitalDeps], robot_id: str, job_id: str) -> dict[str, Any]:
        """Assign an autonomous cart/robot to a job.

        Do not use a low-battery robot for urgent, STAT, blood-product, or
        cold-chain-critical jobs. The tool returns ok=false for low battery.
        """
        return ctx.deps.runtime.log_call(
            "assign_robot",
            {"robot_id": robot_id, "job_id": job_id},
            lambda: ctx.deps.runtime.assign_robot(robot_id, job_id),
        )

    @hospital_tool("check_cold_chain_window")
    def check_cold_chain_window(ctx: RunContext[HospitalDeps], job_id: str = "pending") -> dict[str, Any]:
        """Check remaining cold-chain time for medication totes or blood products."""
        return ctx.deps.runtime.log_call(
            "check_cold_chain_window",
            {"job_id": job_id},
            lambda: ctx.deps.runtime.check_cold_chain_window(job_id),
        )

    @hospital_tool("notify_ward")
    def notify_ward(ctx: RunContext[HospitalDeps], ward: str, message: str, job_id: str | None = None) -> dict[str, Any]:
        """Notify a ward/unit of logistics status, delays, preemption, or ETA changes."""
        return ctx.deps.runtime.log_call(
            "notify_ward",
            {"ward": ward, "message": message, "job_id": job_id},
            lambda: ctx.deps.runtime.notify_ward(ward, message, job_id),
        )

    @hospital_tool("escalate_to_human")
    def escalate_to_human(ctx: RunContext[HospitalDeps], job_id: str = "pending", reason: str = "no safe assignment available", urgency: str = "normal") -> dict[str, Any]:
        """Escalate to a human coordinator when no safe assignment exists or policy requires approval."""
        return ctx.deps.runtime.log_call(
            "escalate_to_human",
            {"job_id": job_id, "reason": reason, "urgency": urgency},
            lambda: ctx.deps.runtime.escalate_to_human(job_id, reason, urgency),
        )

    @hospital_tool("update_job_status")
    def update_job_status(ctx: RunContext[HospitalDeps], job_id: str, status: str, note: str = "") -> dict[str, Any]:
        """Update a logistics job status such as planned, assigned, delayed, blocked, or completed."""
        return ctx.deps.runtime.log_call(
            "update_job_status",
            {"job_id": job_id, "status": status, "note": note},
            lambda: ctx.deps.runtime.update_job_status(job_id, status, note),
        )

    @hospital_tool("query_policy")
    def query_policy(ctx: RunContext[HospitalDeps], topic: str) -> dict[str, Any]:
        """Query local hospital logistics policy for priority, preemption, or safety constraints."""
        return ctx.deps.runtime.log_call(
            "query_policy",
            {"topic": topic},
            lambda: ctx.deps.runtime.query_policy(topic),
        )


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()} if args.case_ids else None
    cases = []
    for case_id in SCENARIOS:
        if wanted and case_id not in wanted:
            continue
        cases.append({"id": case_id, **hospital_scenario(case_id)})
    return cases



def attempted_tool_calls_from_server(server_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for request in server_requests:
        action = request.get("parsed_action") or {}
        if not isinstance(action, dict) or action.get("type") != "tool":
            continue
        calls = action.get("tool_calls")
        call_list = calls if isinstance(calls, list) and calls else [action]
        for call in call_list:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = call.get("arguments", {}) if isinstance(call.get("arguments", {}), dict) else {}
            if name == "get_pending_jobs":
                arguments = {}
            attempts.append(
                {
                    "request_index": request.get("request_index"),
                    "parse_status": request.get("parse_status"),
                    "name": name,
                    "arguments": arguments,
                }
            )
    return attempts



def filter_server_requests_for_case(_arena_case: dict[str, Any], server_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(server_requests)

def score_hospital_calls_strict(
    scenario: dict[str, Any],
    executed_calls: list[dict[str, Any]],
    attempted_calls: list[dict[str, Any]],
    *,
    infrastructure_error: bool,
    context_exhaustion: bool = False,
) -> dict[str, Any]:
    loose = score_hospital_calls(scenario, executed_calls)
    expected = scenario["expected"]
    expected_names = list(expected.get("must_call", []))
    optional_names = set(expected.get("optional_call", []))
    optional_before_names = set(expected.get("optional_before_must", []))
    optional_names.update(optional_before_names)
    if scenario.get("focused_tools") and "get_pending_jobs" in scenario.get("focused_tools", []) and "get_pending_jobs" not in expected_names:
        optional_names.add("get_pending_jobs")
    allowed = set(expected_names) | optional_names
    merged_attempts = list(attempted_calls)
    unmatched_attempt_indices = list(range(len(merged_attempts)))
    for executed in executed_calls:
        executed_args = executed.get("arguments", {}) if isinstance(executed.get("arguments", {}), dict) else {}
        match_index = next(
            (
                idx
                for idx in unmatched_attempt_indices
                if merged_attempts[idx].get("name") == executed.get("name")
                and merged_attempts[idx].get("arguments", {}) == executed_args
            ),
            None,
        )
        if match_index is not None:
            unmatched_attempt_indices.remove(match_index)
            continue
        merged_attempts.append(
            {
                "name": executed["name"],
                "arguments": executed_args,
                "request_index": None,
                "source": "executed_tool_ledger",
            }
        )
    attempted_calls = merged_attempts
    attempted_names = [call["name"] for call in attempted_calls]
    executed_names = [call["name"] for call in executed_calls]

    excess_attempts = [
        {"name": call["name"], "arguments": call.get("arguments", {}), "request_index": call.get("request_index")}
        for call in attempted_calls
        if call["name"] not in allowed
    ]
    forbidden_attempts = [
        {"name": call["name"], "arguments": call.get("arguments", {}), "request_index": call.get("request_index")}
        for call in attempted_calls
        if call["name"] in set(expected.get("must_not_call", []))
    ]
    completion_request_indices: list[int] = []
    for name in expected_names:
        matching_executed = next((call for call in executed_calls if call.get("name") == name), None)
        if matching_executed is None:
            continue
        executed_args = matching_executed.get("arguments", {})
        matching_attempt = next(
            (
                call
                for call in attempted_calls
                if call.get("name") == name
                and call.get("arguments", {}) == executed_args
                and isinstance(call.get("request_index"), int)
            ),
            None,
        )
        if matching_attempt is not None:
            completion_request_indices.append(matching_attempt["request_index"])
    completion_request_index = max(completion_request_indices, default=None)
    out_of_order_attempts = [
        {"name": call["name"], "arguments": call.get("arguments", {}), "request_index": call.get("request_index")}
        for call in attempted_calls
        if completion_request_index is not None
        and call["name"] in optional_before_names
        and isinstance(call.get("request_index"), int)
        and call["request_index"] > completion_request_index
    ]
    duplicate_attempts: list[dict[str, Any]] = []
    for name in sorted(set(attempted_names)):
        allowed_count = expected_names.count(name) + (1 if name in optional_names else 0)
        actual_count = attempted_names.count(name)
        if actual_count > max(1, allowed_count):
            duplicate_attempts.append({"name": name, "count": actual_count, "allowed": max(1, allowed_count)})

    unmatched_attempts = attempted_calls.copy()
    unexecuted_attempts: list[dict[str, Any]] = []
    for executed in executed_calls:
        executed_args = executed.get("arguments", {}) if isinstance(executed.get("arguments", {}), dict) else {}
        match_index = next(
            (
                idx
                for idx, attempt in enumerate(unmatched_attempts)
                if attempt["name"] == executed["name"]
                and all(executed_args.get(key) == value for key, value in attempt.get("arguments", {}).items())
            ),
            None,
        )
        if match_index is None:
            continue
        unmatched_attempts.pop(match_index)
    for attempt in unmatched_attempts:
        if attempt["name"] in HOSPITAL_TOOL_NAMES:
            unexecuted_attempts.append(
                {"name": attempt["name"], "arguments": attempt.get("arguments", {}), "request_index": attempt.get("request_index")}
            )

    duplicate_failures = duplicate_attempts
    unexecuted_failures = unexecuted_attempts
    strict_failures = (
        len(loose.get("missing", []))
        + len(loose.get("arg_errors", []))
        + len(excess_attempts)
        + len(forbidden_attempts)
        + len(duplicate_failures)
        + len(unexecuted_failures)
        + len(out_of_order_attempts)
        + (1 if infrastructure_error or context_exhaustion else 0)
    )
    strict_units = max(1, len(expected_names) + len(expected.get("arguments", {})))
    strict_score = max(0.0, min(1.0, loose.get("score", 0.0) - (strict_failures / strict_units)))
    passed = (
        not infrastructure_error
        and not context_exhaustion
        and not loose.get("missing")
        and not loose.get("arg_errors")
        and not excess_attempts
        and not forbidden_attempts
        and not duplicate_failures
        and not unexecuted_failures
        and not out_of_order_attempts
    )
    return {
        **loose,
        "passed": passed,
        "score": 1.0 if passed else strict_score,
        "strict": True,
        "attempted_tool_calls": attempted_names,
        "executed_tool_calls": executed_names,
        "excess_attempts": excess_attempts,
        "forbidden_attempts": forbidden_attempts,
        "duplicate_attempts": duplicate_attempts,
        "unexecuted_attempts": unexecuted_attempts,
        "out_of_order_attempts": out_of_order_attempts,
        "infrastructure_error": infrastructure_error,
        "context_exhaustion": context_exhaustion,
    }

def diagnose_timeout_or_runtime(
    exception: str,
    server_requests: list[dict[str, Any]],
    attempted_calls: list[dict[str, Any]],
    executed_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    if is_context_exhaustion(exception) or any(
        is_context_exhaustion(item.get("raw_answer", "")) for item in server_requests
    ):
        return {
            "kind": "context_exhaustion",
            "detail": "The accumulated prompt/tool transcript exceeded the model context window",
        }
    if is_infrastructure_error(exception) or any(
        is_infrastructure_error(item.get("raw_answer", "")) for item in server_requests
    ):
        return {"kind": "infrastructure_error", "detail": "Inference runtime or device initialization failure"}
    if any(item.get("timed_out") for item in server_requests):
        return {"kind": "genie_generation_timeout", "detail": "genie-t2t-run exceeded server timeout"}
    if "Failed to parse tool call arguments as JSON" in exception:
        return {
            "kind": "tool_protocol_error",
            "detail": "Model emitted malformed tool-call JSON that the inference server rejected",
        }
    if "UsageLimitExceeded" in exception:
        return {"kind": "agent_request_limit", "detail": "Pydantic AI stopped the loop at the per-pass request limit"}
    if "TimeoutError" in exception:
        attempted_names = [call["name"] for call in attempted_calls]
        executed_names = [call["name"] for call in executed_calls]
        repeated = sorted({name for name in attempted_names if attempted_names.count(name) > 1})
        if repeated:
            return {
                "kind": "agent_loop_timeout_repeated_tools",
                "detail": "Model kept requesting tools and did not self-complete",
                "repeated_tools": repeated,
            }
        if attempted_names and not executed_names:
            return {
                "kind": "agent_loop_timeout_invalid_tools",
                "detail": "Model proposed tools that did not successfully execute before host pass timeout",
            }
        return {"kind": "host_agent_pass_timeout", "detail": "Host-side agent pass exceeded timeout"}
    if exception:
        return {"kind": "agent_exception", "detail": exception}
    return {"kind": "none", "detail": ""}
def classify_failure(score: dict[str, Any], exception: str, server_requests: list[dict[str, Any]]) -> str:
    if score["passed"]:
        return "passed"
    if score.get("context_exhaustion") or is_context_exhaustion(exception) or any(
        is_context_exhaustion(item.get("raw_answer", "")) for item in server_requests
    ):
        return "context_exhaustion"
    if score.get("infrastructure_error") or is_infrastructure_error(exception) or any(
        is_infrastructure_error(item.get("raw_answer", "")) for item in server_requests
    ):
        return "infrastructure_error"
    if "Failed to parse tool call arguments as JSON" in exception:
        return "protocol_error"
    if exception:
        return "agent_exception"
    if any(item.get("timed_out") for item in server_requests):
        return "model_timeout"
    if any(not item.get("parsed_ok") for item in server_requests):
        return "protocol_error"
    if score.get("missing"):
        return "missing_required_tool"
    if score.get("forbidden"):
        return "unsafe_or_forbidden_tool"
    if score.get("arg_errors"):
        return "wrong_tool_arguments"
    return "wrong_tool_behavior"


def is_infrastructure_error(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    signatures = [
        "Failed to create device: 14001",
        "Device Creation failure",
        "Failure to initialize model",
        "Failed to create the dialog",
        "Could not create context from binary",
        "Create From Binary FAILED",
    ]
    return any(signature in text for signature in signatures)


def is_context_exhaustion(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    signatures = [
        "Context Size was exceeded",
        "exceeds the available context size",
        "maximum context length",
    ]
    return any(signature in text for signature in signatures)


def completion_checklist(arena_case: dict[str, Any], calls: list[dict[str, Any]]) -> list[str]:
    jobs = list(arena_case.get("jobs", {}).values())
    has_cold_chain = any(job.get("requires_cold_chain") for job in jobs)
    has_assets = any(job.get("asset_id") for job in jobs)
    has_replan = arena_case.get("event", {}).get("kind") == "replan"
    called = {call["name"] for call in calls}
    checklist = [
        "Start from get_pending_jobs if the current queue is not known.",
        "Use tool results as state; do not rely on prompt assumptions for resources or constraints.",
        "If a route uses elevators, check status and reserve an available elevator before assigning a carrier.",
        "Assign an available porter or suitable robot only after route and timing constraints are verified.",
        "If a tool returned ok=false, fix the prerequisite shown in that result and retry the action if it is still needed.",
        "Finish by setting each affected job to an operational status such as assigned, delayed, blocked, or escalated.",
        "Notify affected wards/units when a delivery is assigned, delayed, blocked, or escalated.",
    ]
    if has_assets and "get_asset_location" not in called:
        checklist.append("Check asset location/readiness before acting on asset-dependent jobs.")
    if has_cold_chain and "check_cold_chain_window" not in called:
        checklist.append("Check cold-chain window for medication totes or blood products.")
    if has_replan:
        checklist.append("For replanning, protect the higher-priority job and update the lower-priority job if it is delayed.")
    return checklist

async def run_case_async(
    args: argparse.Namespace,
    arena_case: dict[str, Any],
    Agent: Any,
    OpenAIChatModel: Any,
    OpenAIProvider: Any,
    result_dir: Path,
) -> dict[str, Any]:
    reset_server_log(args.base_url)
    baseline_request_ids = {item.get("request_index") for item in get_server_log(args.base_url)}
    started = time.monotonic()
    runtime = HospitalLogisticsRuntime(scenario=arena_case)
    deps = HospitalDeps(runtime=runtime)
    output = ""
    exception = ""
    messages: list[str] = []

    model = build_hospital_model(args, OpenAIChatModel, OpenAIProvider)
    case_instructions = HOSPITAL_LOGISTICS_INSTRUCTIONS
    if arena_case.get("focused_tools"):
        case_instructions = (
            "You are a hospital logistics coordinator running locally on an edge device.\n"
            "Use the available tools to complete the user task. Prefer the most direct available tool.\n"
            "If a tool says no arguments are needed, call it with empty arguments. Use at most one tool call per response. After a successful tool response, answer final and do not call another tool."
        )
    case_instructions = mode_instruction_prefix(args.mode) + case_instructions
    if args.system_prompt_file:
        case_instructions = (
            args.system_prompt_file.expanduser().read_text().strip() + "\n\n" + case_instructions
        )
    agent_kwargs: dict[str, Any] = {
        "instructions": case_instructions,
        "deps_type": HospitalDeps,
        "retries": args.agent_retries,
    }
    if ModelSettings is not None:
        settings: dict[str, Any] = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if args.top_p is not None:
            settings["top_p"] = args.top_p
        if args.parallel_tool_calls is not None:
            settings["parallel_tool_calls"] = args.parallel_tool_calls
        if args.tool_choice:
            settings["tool_choice"] = args.tool_choice
        agent_kwargs["model_settings"] = ModelSettings(**settings)
    agent = Agent(model, **agent_kwargs)
    register_hospital_tools(agent, strict=args.strict_tools, sequential=args.sequential_tools, allowed_tools=arena_case.get("focused_tools"))

    if arena_case.get("focused_tools"):
        base_prompt = {
            "task": arena_case.get("task_prompt", "Use the available tools to complete this logistics task."),
            "job_reference": "For these focused cases, tools may use job_id=\"pending\" for the single pending job.",
        }
    else:
        base_prompt = {
            "scenario": arena_case["description"],
            "event": arena_case["event"],
            "instruction": (
                "Use the tools to gather all resource facts. Do not assume elevator, "
                "porter, robot, asset, cold-chain, or policy state from the prompt. "
                "Continue calling tools until the job is assigned, delayed, blocked, "
                "or escalated and the affected wards are notified."
            ),
        }
    score = {"passed": False, "missing": [], "forbidden": [], "arg_errors": [], "score": 0.0}
    pass_outputs: list[str] = []
    all_messages: list[str] = []
    for workflow_pass in range(args.workflow_passes):
        if workflow_pass == 0:
            user_prompt = json.dumps(base_prompt, indent=2)
        else:
            user_prompt = json.dumps(
                {
                    **base_prompt,
                    "continuation": (
                        "Continue with the next required action only. "
                        "Do not repeat tools whose facts are already available unless the status changed."
                    ) if arena_case.get("focused_tools") else (
                        "You stopped before the logistics workflow was complete. "
                        "Use the remaining tools needed to satisfy the operational constraints. "
                        "Do not repeat tools whose facts are already available unless the status changed."
                    ),
                    "tool_calls_so_far": [
                        {"name": call["name"], "arguments": call.get("arguments", {}), "result": call.get("result", {})}
                        for call in runtime.calls
                    ],
                    "completion_checklist": completion_checklist(arena_case, runtime.calls),
                },
                indent=2,
            )
        try:
            usage_limits = UsageLimits(request_limit=args.internal_request_limit) if UsageLimits is not None else None
            result = await asyncio.wait_for(agent.run(user_prompt, deps=deps, usage_limits=usage_limits), timeout=args.pass_timeout_s)
            output = str(result.output)
            pass_outputs.append(output)
            all_messages.extend(summarize_messages(result))
        except TimeoutError:
            exception = f"TimeoutError('agent pass exceeded {args.pass_timeout_s}s')"
            continue
        except UsageLimitExceeded as exc:
            exception = repr(exc)
            continue
        except Exception as exc:
            exception = repr(exc)
            break
        score = score_hospital_calls(arena_case, runtime.calls)
        if score["passed"] or not args.workflow_continue:
            break
    score = score_hospital_calls(arena_case, runtime.calls)
    output = "\n\n---\n\n".join(pass_outputs) if pass_outputs else output
    messages = all_messages

    elapsed_s = time.monotonic() - started
    server_requests = filter_server_requests_for_case(
        arena_case,
        [
            item for item in get_server_log(args.base_url)
            if item.get("request_index") not in baseline_request_ids
        ],
    )
    infra_error = is_infrastructure_error(exception) or any(
        is_infrastructure_error(item.get("raw_answer", "")) for item in server_requests
    )
    context_exhaustion = is_context_exhaustion(exception) or any(
        is_context_exhaustion(item.get("raw_answer", "")) for item in server_requests
    )
    attempted_calls = attempted_tool_calls_from_server(server_requests)
    score = score_hospital_calls_strict(
        arena_case,
        runtime.calls,
        attempted_calls,
        infrastructure_error=infra_error,
        context_exhaustion=context_exhaustion,
    )
    runtime_diagnosis = diagnose_timeout_or_runtime(exception, server_requests, attempted_calls, runtime.calls)
    parser_modes = compact_values([str(item.get("parser", "")) for item in server_requests])
    parse_statuses = compact_values([str(item.get("parse_status", "")) for item in server_requests])
    executed_tools = compact_values([call["name"] for call in runtime.calls])
    record = {
        "model": args.model_label,
        "mode": args.mode,
        "client": "pydantic_ai_hospital_logistics",
        "provider": args.provider,
        "case_id": arena_case["id"],
        "description": arena_case["description"],
        "event": arena_case["event"],
        "output": output,
        "score": score,
        "attempted_tool_calls": attempted_calls,
        "workflow_passes_used": len(pass_outputs),
        "failure_kind": classify_failure(score, exception, server_requests),
        "runtime_diagnosis": runtime_diagnosis,
        "exception": exception,
        "elapsed_s": round(elapsed_s, 3),
        "tool_calls": runtime.calls,
        "server_requests": server_requests,
        "messages": messages,
        "notes": (
            f"model_requests={len(server_requests)},"
            f"tool_calls={len(runtime.calls)},"
            f"parser={parser_modes},"
            f"parse_status={parse_statuses},"
            f"executed_tools={executed_tools}"
        ),
    }
    (result_dir / f"{args.model_label}__{args.mode}__{arena_case['id']}.json").write_text(
        json.dumps(record, indent=2)
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("agent_arena/.env"))
    parser.add_argument("--provider", choices=["openai-compatible", "azure", "azure-foundry-anthropic", "anthropic"])
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--azure-endpoint")
    parser.add_argument("--azure-api-version")
    parser.add_argument("--anthropic-foundry-base-url")
    parser.add_argument("--model-name")
    parser.add_argument("--model-label")
    parser.add_argument("--mode", default="stock")
    parser.add_argument("--case-ids")
    parser.add_argument("--agent-retries", type=int, default=1)
    parser.add_argument("--workflow-passes", type=int, default=3)
    parser.add_argument("--pass-timeout-s", type=float, default=240.0)
    parser.add_argument("--internal-request-limit", type=int, default=6)
    parser.add_argument("--no-workflow-continue", dest="workflow_continue", action="store_false")
    parser.set_defaults(workflow_continue=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--strict-tools", action="store_true")
    parser.add_argument("--sequential-tools", action="store_true")
    parser.add_argument("--parallel-tool-calls", choices=["true", "false"])
    parser.add_argument("--tool-choice", choices=["auto", "required", "none"])
    parser.add_argument("--openai-strict-tools", action="store_true")
    parser.add_argument("--no-openai-tool-choice-required", dest="openai_tool_choice_required", action="store_false")
    parser.set_defaults(openai_tool_choice_required=True)
    parser.add_argument("--out-root", type=Path, default=Path.home() / "agent_arena_results")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    if args.parallel_tool_calls is not None:
        args.parallel_tool_calls = args.parallel_tool_calls == "true"
    return args


async def amain() -> int:
    args = parse_args()
    if args.list_cases:
        for case_id, scenario in SCENARIOS.items():
            print(f"{case_id}\t{scenario['description']}")
        return 0
    apply_env_defaults(args)
    Agent, OpenAIChatModel, OpenAIProvider = require_pydantic_ai()
    cases = select_cases(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = (
        args.out_root.expanduser()
        / f"{timestamp}__pydantic_hospital_logistics__{args.model_label}__{args.mode}__{args.provider}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    results = [
        await run_case_async(args, arena_case, Agent, OpenAIChatModel, OpenAIProvider, result_dir)
        for arena_case in cases
    ]
    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(result_dir / "summary.md", "Pydantic AI Hospital Logistics Arena", results)
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())
    print(f"SERVER_DEBUG_ROOT={root_from_base_url(args.base_url)}/debug/requests")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
