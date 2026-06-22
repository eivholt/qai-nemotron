#!/usr/bin/env python3
"""OR-agent-style fixtures and tools for edge agent benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_arena.mcp_tool_server import append_log


OR_AGENT_INSTRUCTIONS = """\
You are a local OR logistics agent (synthetic demo). Coordinate supply and review workflows. Never diagnose or prescribe.

WORKFLOW - always follow this order:
1. Call get_case(case_id) to fetch the EMR case and required equipment.
2. Call check_supplies() to compare detected items against requirements.
3. Call inspect_scene to check the sterile zone (see its docstring for when to call).
   The image_path MUST be taken from the event's "image_path" field - do NOT invent a path.
4. Now take ALL applicable actions together:
   a) Call set_stacklight:
      - set red if inspect_scene verdict is true (sterile zone issue).
      - set yellow if check_supplies shows deficits.
      - set green ONLY when: no sterile issue AND no deficits.
   b) If inspect_scene verdict is true -> call create_task(human_review) for the sterile zone issue.
   c) Deficit items -> call request_resupply for each one. That is all - move on.

ACTION MAPPING (use exactly the right tool for each situation):
- deficit item              -> request_resupply (the ONLY correct tool for deficits)
- sterile zone verdict=true -> create_task

RULES:
- Do NOT create tasks for deficit items. Only use request_resupply for deficits.
- Every run MUST include exactly one set_stacklight call.
- Never ask the user for input. Always decide and act autonomously.
- Your final text response must be a SHORT summary (2-4 sentences max)."""


OR_AGENT_INSTRUCTIONS_SIMPLE = """\
You are a local OR logistics agent (synthetic demo). Coordinate supply and review workflows. Never diagnose or prescribe.

First gather facts:
1. Call get_case(case_id).
2. Call check_supplies().
3. If visible_items is not empty, call inspect_scene(image_path from the event).

Then act from this decision table:
- inspect_scene.verdict=true -> set_stacklight red, create_task human_review, and also request_resupply for any supply deficits.
- inspect_scene.verdict=false and check_supplies.deficits is not empty -> set_stacklight yellow and request_resupply for each deficit. Do not create_task.
- inspect_scene.verdict=false and check_supplies.deficits is empty -> set_stacklight green only.

Tool meanings:
- create_task is only for sterile zone verdict=true.
- request_resupply is only for supply deficits.
- Every run must include exactly one set_stacklight call.

Final response: short summary, 2-4 sentences."""


OR_AGENT_INSTRUCTIONS_RELAXED = """\
You are a local OR logistics agent (synthetic demo). Coordinate supply and review workflows. Never diagnose or prescribe.

Use the available tools to understand the case, observed supplies, and OR scene. Base every action on tool results, not guesses.

Helpful approach:
- Fetch the case before deciding what supplies are missing.
- Compare observed supplies with case requirements.
- Inspect the scene when instruments are visible.
- If supplies are missing, request resupply for the missing items.
- If the scene inspection reports a sterile-zone issue, set the stacklight red and create a human_review task.
- If there is no sterile issue, set yellow for supply deficits or green when everything is ready.

Final response: short summary, 2-4 sentences."""


OR_AGENT_INSTRUCTIONS_TOOL_DOC_FIRST = """\
You are a local OR logistics agent (synthetic demo). Coordinate supply and review workflows. Never diagnose or prescribe.

Use the available tool descriptions as the source of truth for what each tool does and when to use it. Do not guess values that a tool can return.

Gather the facts you need, then take the required logistics actions. Keep the final response short."""


OR_AGENT_INSTRUCTIONS_STEPWISE = """\
You are a local OR logistics agent (synthetic demo). Coordinate supply and review workflows. Never diagnose or prescribe.

Work step by step. On each assistant turn, call only the next tool or small set of tools whose arguments are already known. After tool results return, use those results to decide the next action. Do not try to plan or emit the whole workflow before seeing tool results.

Start by fetching the case. Then compare supplies. Then inspect the scene if instruments are visible. After those facts are known, take the needed logistics actions:
- sterile-zone issue -> set red and create a human_review task
- supply deficits -> set yellow unless red is needed, and request resupply for each deficit
- no sterile issue and no deficits -> set green

Do not request resupply for items that are not listed as deficits. Keep the final response short."""


OR_AGENT_INSTRUCTION_STYLES: dict[str, str] = {
    "legacy": OR_AGENT_INSTRUCTIONS,
    "simple": OR_AGENT_INSTRUCTIONS_SIMPLE,
    "relaxed": OR_AGENT_INSTRUCTIONS_RELAXED,
    "stepwise": OR_AGENT_INSTRUCTIONS_STEPWISE,
    "tool_doc_first": OR_AGENT_INSTRUCTIONS_TOOL_DOC_FIRST,
}


OR_MCP_INSTRUCTIONS_BY_STYLE: dict[str, str] = {
    "legacy": (
        "Use these OR logistics tools in workflow order: get_case, "
        "check_supplies, inspect_scene when objects are visible, then "
        "set_stacklight and any required action tools."
    ),
    "simple": (
        "Use these OR logistics tools to gather facts first, then act from "
        "tool results. Deficits use request_resupply. Sterile-zone verdicts "
        "use set_stacklight red and create_task human_review."
    ),
    "relaxed": (
        "Use these OR logistics tools to inspect the case, supplies, and scene. "
        "Base actions only on returned tool data."
    ),
    "stepwise": (
        "Use these OR logistics tools step by step. Call the next needed tool, "
        "wait for its result, then decide the next action from returned data."
    ),
    "tool_doc_first": (
        "Use the tool docstrings and schemas as the source of truth. Retrieve "
        "case, supply, and scene facts with tools before deciding actions."
    ),
}


def or_agent_instructions(style: str) -> str:
    try:
        return OR_AGENT_INSTRUCTION_STYLES[style]
    except KeyError as exc:
        raise ValueError(f"Unknown OR instruction style: {style}") from exc


def or_mcp_instructions(style: str) -> str:
    return OR_MCP_INSTRUCTIONS_BY_STYLE.get(style, OR_MCP_INSTRUCTIONS_BY_STYLE["legacy"])


OR_TOOL_NAMES = [
    "get_case",
    "check_supplies",
    "inspect_scene",
    "create_task",
    "request_resupply",
    "set_stacklight",
]


OR_RESOURCES = {
    "room_id": "OR-BENCH",
    "sterile_processing_robot": {"available": True, "eta_seconds": 180},
    "human_runner": {"available": True, "eta_seconds": 420},
    "porter": {"available": True, "eta_seconds": 300},
    "local_vlm": {"available": True, "estimated_latency_seconds": 5},
    "pc_gpu_vlm": {"available": True, "estimated_latency_seconds": 3},
}


OR_CASES_BY_ID: dict[str, dict[str, Any]] = {
    "CASE-1042": {
        "case_id": "CASE-1042",
        "patient_id": "SYN-PAT-8842",
        "procedure": "Laparoscopic Biopsy",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 3, "sponge": 6, "tweezers": 2},
    },
    "CASE-1044": {
        "case_id": "CASE-1044",
        "patient_id": "SYN-PAT-8844",
        "procedure": "Laparoscopic Biopsy",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 2, "sponge": 4, "tweezers": 1},
    },
    "CASE-1045": {
        "case_id": "CASE-1045",
        "patient_id": "SYN-PAT-8845",
        "procedure": "Laparoscopic Cholecystectomy",
        "priority": "normal",
        "required_items": {"tweezers": 2, "scalpel": 2, "sponge": 3, "scissors": 1},
    },
    "CASE-BENCH-1": {
        "case_id": "CASE-BENCH-1",
        "patient_id": "SYN-PAT-BENCH1",
        "procedure": "Minor Excision",
        "priority": "normal",
        "required_items": {"scalpel": 1, "scissors": 1, "sponge": 2},
    },
    "CASE-BENCH-2": {
        "case_id": "CASE-BENCH-2",
        "patient_id": "SYN-PAT-BENCH2",
        "procedure": "Laparoscopic Cholecystectomy",
        "priority": "normal",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
    },
    "CASE-BENCH-4": {
        "case_id": "CASE-BENCH-4",
        "patient_id": "SYN-PAT-BENCH4",
        "procedure": "Open Conversion (Changed from Laparoscopic Appendectomy)",
        "priority": "high",
        "required_items": {"scalpel": 3, "scissors": 2, "sponge": 6, "tweezers": 2},
    },
    "CASE-BENCH-5": {
        "case_id": "CASE-BENCH-5",
        "patient_id": "SYN-PAT-BENCH5",
        "procedure": "Laparoscopic Biopsy",
        "priority": "high",
        "required_items": {"scalpel": 2, "scissors": 2, "sponge": 4, "tweezers": 2},
    },
}


def _bench_event(case_id: str, visible_items: dict[str, int]) -> dict[str, Any]:
    return {
        "room_id": "OR-BENCH",
        "case_id": case_id,
        "visible_items": dict(visible_items),
        "image_path": "frames/frame_all_present.png",
    }


OR_BENCH_CASES: list[dict[str, Any]] = [
    {
        "id": "or_L1_01_all_present_green_light",
        "level": 1,
        "group": "or_benchmark",
        "description": "Everything present -> green light.",
        "event": _bench_event("CASE-BENCH-1", {"scalpel": 1, "scissors": 1, "sponge": 2}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L1_02_single_missing_item_resupply",
        "level": 1,
        "group": "or_benchmark",
        "description": "One missing item -> request_resupply.",
        "event": _bench_event("CASE-BENCH-1", {"scalpel": 1, "sponge": 2}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L1_03_no_action_surplus",
        "level": 1,
        "group": "or_benchmark",
        "description": "Surplus counts -> green light, no resupply.",
        "event": _bench_event("CASE-BENCH-1", {"scalpel": 3, "scissors": 2, "sponge": 5}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L2_01_two_missing_items",
        "level": 2,
        "group": "or_benchmark",
        "description": "Two deficits -> two resupply calls.",
        "event": _bench_event(
            "CASE-BENCH-2",
            {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 1},
        ),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L2_02_three_items_missing",
        "level": 2,
        "group": "or_benchmark",
        "description": "Three deficits -> three resupply calls.",
        "event": _bench_event(
            "CASE-BENCH-2",
            {"scalpel": 2, "scissors": 0, "sponge": 2, "tweezers": 1},
        ),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L2_03_unaccounted_not_flagged",
        "level": 2,
        "group": "or_benchmark",
        "description": "Only scalpel visible -> resupply all required deficits.",
        "event": _bench_event("CASE-BENCH-2", {"scalpel": 2}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L3_01_deficit_resupply_yellow",
        "level": 3,
        "group": "or_benchmark",
        "description": "Deficit -> resupply and yellow light.",
        "event": _bench_event("CASE-BENCH-1", {"scalpel": 1, "sponge": 2}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L3_02_deficit_triggers_action",
        "level": 3,
        "group": "or_benchmark",
        "description": "Sponge deficit -> resupply action.",
        "event": _bench_event("CASE-BENCH-1", {"scalpel": 1, "scissors": 1, "sponge": 1}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L3_03_high_count_resupply",
        "level": 3,
        "group": "or_benchmark",
        "description": "Scissors deficit despite high counts elsewhere.",
        "event": _bench_event(
            "CASE-BENCH-2",
            {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 2},
        ),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L3_04_no_deficit_no_task",
        "level": 3,
        "group": "or_benchmark",
        "description": "No deficit -> no deficit task.",
        "event": _bench_event("CASE-BENCH-1", {"scalpel": 2, "scissors": 1, "sponge": 3}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L4_01_all_present_green",
        "level": 4,
        "group": "or_benchmark",
        "description": "Complex case all present -> green.",
        "event": _bench_event(
            "CASE-BENCH-4",
            {"scalpel": 3, "scissors": 2, "sponge": 6, "tweezers": 2},
        ),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L4_02_deficits_with_resupply",
        "level": 4,
        "group": "or_benchmark",
        "description": "Complex case deficits -> yellow and resupply.",
        "event": _bench_event(
            "CASE-BENCH-4",
            {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 2},
        ),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L4_03_high_priority_propagation",
        "level": 4,
        "group": "or_benchmark",
        "description": "High priority case with deficits -> resupply.",
        "event": _bench_event(
            "CASE-BENCH-5",
            {"scalpel": 1, "scissors": 1, "sponge": 2, "tweezers": 1},
        ),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L5_01_deficit_resupply",
        "level": 5,
        "group": "or_benchmark",
        "description": "Adversarial repeat: deficit should trigger action.",
        "event": _bench_event("CASE-BENCH-1", {"scalpel": 1, "sponge": 2}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L5_02_all_items_surplus_no_resupply",
        "level": 5,
        "group": "or_benchmark",
        "description": "All surplus -> never resupply.",
        "event": _bench_event("CASE-BENCH-1", {"scalpel": 3, "scissors": 2, "sponge": 4}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L5_03_mixed_deficit_and_surplus",
        "level": 5,
        "group": "or_benchmark",
        "description": "Mixed scissors/tweezers deficits, sponge surplus.",
        "event": _bench_event(
            "CASE-BENCH-2",
            {"scalpel": 2, "scissors": 1, "sponge": 4, "tweezers": 0},
        ),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L5_04_zero_visible_items",
        "level": 5,
        "group": "or_benchmark",
        "description": "Empty table -> required items missing.",
        "event": _bench_event("CASE-BENCH-2", {}),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L5_05_spd_accompanies_deficit",
        "level": 5,
        "group": "or_benchmark",
        "description": "Deficits should request SPD/resupply.",
        "event": _bench_event(
            "CASE-BENCH-2",
            {"scalpel": 1, "scissors": 0, "sponge": 2, "tweezers": 1},
        ),
        "sterile_zone_issue": False,
    },
    {
        "id": "or_L5_06_all_present_no_deficit",
        "level": 5,
        "group": "or_benchmark",
        "description": "All present -> not yellow.",
        "event": _bench_event(
            "CASE-BENCH-4",
            {"scalpel": 3, "scissors": 2, "sponge": 6, "tweezers": 2},
        ),
        "sterile_zone_issue": False,
    },
]


OR_SCENARIO_CASES: list[dict[str, Any]] = [
    {
        "id": "or_scenario_all_present",
        "level": 2,
        "group": "or_scenario",
        "description": "OR integration scenario: all instruments present.",
        "event": {
            "room_id": "OR-2",
            "case_id": "CASE-1044",
            "image_path": "frames/frame_all_present.png",
            "visible_items": {"scalpel": 1, "scissors": 2, "sponge": 4, "tweezers": 1},
        },
        "sterile_zone_issue": False,
    },
    {
        "id": "or_scenario_missing_scissors",
        "level": 3,
        "group": "or_scenario",
        "description": "OR integration scenario: scissors and sponge deficits.",
        "event": {
            "room_id": "OR-2",
            "case_id": "CASE-1042",
            "image_path": "frames/frame_missing_scissors.png",
            "visible_items": {"scalpel": 2, "sponge": 3, "tweezers": 2},
        },
        "sterile_zone_issue": False,
    },
    {
        "id": "or_scenario_missing_something",
        "level": 3,
        "group": "or_scenario",
        "description": "OR integration scenario: multiple visible deficits.",
        "event": {
            "room_id": "OR-2",
            "case_id": "CASE-1042",
            "image_path": "frames/frame_missing_something.png",
            "visible_items": {"scissors": 2, "scalpel": 1, "sponge": 4, "tweezers": 1},
        },
        "sterile_zone_issue": False,
    },
    {
        "id": "or_scenario_instrument_out_of_zone",
        "level": 4,
        "group": "or_scenario",
        "description": "OR integration scenario: sterile zone violation.",
        "event": {
            "room_id": "OR-2",
            "case_id": "CASE-1045",
            "image_path": "frames/frame_instrument_out_of_zone.png",
            "visible_items": {"scalpel": 2, "scissors": 1, "sponge": 3, "tweezers": 2},
        },
        "sterile_zone_issue": True,
    },
    {
        "id": "or_scenario_sterile_zone_ambiguity",
        "level": 4,
        "group": "or_scenario",
        "description": "OR integration scenario: ambiguity but no sterile issue.",
        "event": {
            "room_id": "OR-2",
            "case_id": "CASE-1042",
            "image_path": "frames/frame_sterile_zone_ambiguity.png",
            "visible_items": {"scalpel": 1, "scissors": 2, "sponge": 3},
        },
        "sterile_zone_issue": False,
    },
]


OR_ARENA_CASES = OR_BENCH_CASES + OR_SCENARIO_CASES


@dataclass
class ORRuntime:
    arena_case: dict[str, Any]
    include_hints: bool = False
    enforce_guardrails: bool = False
    case: dict[str, Any] | None = None
    reconciliation: dict[str, Any] | None = None
    sterile_verdict: bool | None = None
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    log_path: Path | None = None

    @property
    def event(self) -> dict[str, Any]:
        return self.arena_case["event"]

    @property
    def resources(self) -> dict[str, Any]:
        resources = dict(OR_RESOURCES)
        resources["room_id"] = self.event.get("room_id", resources["room_id"])
        return resources

    def log_call(self, name: str, args: dict[str, Any], fn: Callable[[], Any]) -> Any:
        return logged_or_call(self.tool_log, self.log_path, name, args, fn)

    def get_case(self, case_id: str) -> dict[str, Any]:
        case = dict(OR_CASES_BY_ID[case_id])
        case.pop("patient_id", None)
        self.case = case
        return case

    def check_supplies(self) -> dict[str, Any]:
        if self.case is None:
            return {"error": "No case data - call get_case first."}
        deficits = reconcile(self.event, self.case)
        if deficits:
            summary = (
                f"{len(deficits)} deficit(s): "
                + ", ".join(f"{item['item']} ({item['have']}/{item['need']})" for item in deficits)
                + ". Action: request_resupply for each."
            )
        else:
            summary = "All instrument types match or exceed requirements."
        self.reconciliation = {"all_present": not deficits, "deficits": deficits, "summary": summary}
        if self.include_hints:
            self.reconciliation.update(
                {
                    "recommended_stacklight_if_no_sterile_issue": "yellow" if deficits else "green",
                    "recommended_deficit_actions": [
                        {
                            "tool": "request_resupply",
                            "item_name": item["item"],
                            "room_id": self.event.get("room_id", "OR-BENCH"),
                            "urgency": "high",
                        }
                        for item in deficits
                    ],
                    "do_not_create_task_for_supply_deficits": True,
                }
            )
        return self.reconciliation

    def inspect_scene(self, image_path: str) -> dict[str, Any]:
        verdict = bool(self.arena_case.get("sterile_zone_issue", False))
        self.sterile_verdict = verdict
        if verdict:
            answer = "An instrument is resting on the bare table beyond the sterile drape."
        else:
            answer = "All visible instruments are on the sterile drape; no bare-table violation is visible."
        result = {
            "image": image_path,
            "question": "Are any sponge, scissors, tweezers, scalpel on the bare table outside the sterile drape?",
            "answer": answer,
            "verdict": verdict,
        }
        if self.include_hints:
            result.update(
                {
                    "recommended_stacklight_if_true": "red",
                    "create_human_review_task": verdict,
                    "do_not_create_task_when_verdict_false": True,
                }
            )
        return result

    def create_task(
        self,
        case_id: str,
        task_type: str,
        priority: str,
        summary: str,
        reason: str,
    ) -> dict[str, Any]:
        if self.enforce_guardrails and task_type == "human_review" and self.sterile_verdict is False:
            raise ValueError(
                "Rejected create_task(human_review): inspect_scene.verdict is false, so there is no sterile zone issue."
            )
        return {
            "status": "created",
            "case_id": case_id,
            "task_type": task_type,
            "priority": priority,
            "summary": summary,
            "reason": reason,
        }

    def request_resupply(self, item_name: str, room_id: str, urgency: str) -> dict[str, Any]:
        if self.enforce_guardrails and self.reconciliation is not None:
            deficit_items = {
                str(item.get("item", "")).lower()
                for item in self.reconciliation.get("deficits", [])
                if isinstance(item, dict)
            }
            if item_name.lower() not in deficit_items:
                raise ValueError(f"Rejected request_resupply({item_name}): item is not in check_supplies.deficits.")
        return {
            "request_id": f"SPD-{item_name}-{room_id}",
            "item_name": item_name,
            "room_id": room_id,
            "urgency": urgency,
            "status": "requested",
        }

    def set_stacklight(self, room_id: str, color: str, reason: str) -> dict[str, Any]:
        if self.enforce_guardrails and self.reconciliation is not None and self.sterile_verdict is not None:
            deficits = self.reconciliation.get("deficits", [])
            expected = "red" if self.sterile_verdict else "yellow" if deficits else "green"
            if color.lower() != expected:
                why = (
                    "inspect_scene.verdict is true"
                    if self.sterile_verdict
                    else "check_supplies.deficits is non-empty"
                    if deficits
                    else "there are no sterile issues or supply deficits"
                )
                raise ValueError(f"Rejected stacklight color {color}: expected {expected} because {why}.")
        return {"room_id": room_id, "color": color, "reason": reason, "status": "set"}


def logged_or_call(
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
        result = {"ok": False, "error": str(exc)}
        record = {
            "name": name,
            "args": args,
            "ok": False,
            "elapsed_s": round(time.monotonic() - started, 5),
            "error": str(exc),
        }
    if log is not None:
        log.append(record)
    append_log(log_path, record)
    return result


def reconcile(event: dict[str, Any], case: dict[str, Any]) -> list[dict[str, Any]]:
    required = dict(case.get("required_items", {}))
    visible = dict(event.get("visible_items", {}))
    deficits = []
    for item, need in sorted(required.items()):
        have = int(visible.get(item, 0))
        if have < int(need):
            deficits.append({"item": item, "have": have, "need": int(need)})
    return deficits


def expected_or_state(arena_case: dict[str, Any]) -> dict[str, Any]:
    event = arena_case["event"]
    case = OR_CASES_BY_ID[event["case_id"]]
    deficits = reconcile(event, case)
    sterile = bool(arena_case.get("sterile_zone_issue", False))
    light = "red" if sterile else "yellow" if deficits else "green"
    return {
        "deficits": deficits,
        "deficit_items": [item["item"] for item in deficits],
        "sterile_zone_issue": sterile,
        "light": light,
        "inspect_required": bool(event.get("visible_items")),
    }


def _normal_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _tool_args(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("args", {})
    return args if isinstance(args, dict) else {}


def score_or_tool_calls(
    arena_case: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    *,
    accepted_only: bool = False,
) -> dict[str, Any]:
    expected = expected_or_state(arena_case)
    all_tool_calls = list(tool_calls)
    accepted_tool_calls = [call for call in all_tool_calls if call.get("ok", True)]
    rejected_tool_calls = [call for call in all_tool_calls if not call.get("ok", True)]
    scored_tool_calls = accepted_tool_calls if accepted_only else all_tool_calls
    names = [call.get("name") for call in scored_tool_calls]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for call in scored_tool_calls:
        by_name.setdefault(str(call.get("name")), []).append(call)

    event = arena_case["event"]
    expected_case_id = str(event["case_id"])
    expected_room_id = str(event.get("room_id", "OR-BENCH"))
    expected_image_path = str(event.get("image_path", ""))
    expected_deficit_items = sorted(_normal_text(item) for item in expected["deficit_items"])
    allowed_urgencies = {"low", "normal", "high"}

    checks: list[tuple[str, bool]] = []

    def add_check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    add_check("no_unknown_tools", all(str(name) in OR_TOOL_NAMES for name in names))
    if not accepted_only:
        add_check("no_rejected_tool_calls", not rejected_tool_calls)

    get_case_calls = by_name.get("get_case", [])
    add_check("called_get_case_once", len(get_case_calls) == 1)
    add_check(
        "get_case_case_id",
        len(get_case_calls) == 1 and str(_tool_args(get_case_calls[0]).get("case_id")) == expected_case_id,
    )

    check_supplies_calls = by_name.get("check_supplies", [])
    add_check("called_check_supplies_once", len(check_supplies_calls) == 1)
    add_check(
        "check_supplies_no_arguments",
        len(check_supplies_calls) == 1 and _tool_args(check_supplies_calls[0]) == {},
    )

    inspect_calls = by_name.get("inspect_scene", [])
    if expected["inspect_required"]:
        add_check("called_inspect_scene_once", len(inspect_calls) == 1)
        add_check(
            "inspect_scene_image_path",
            len(inspect_calls) == 1
            and str(_tool_args(inspect_calls[0]).get("image_path")) == expected_image_path,
        )
    else:
        add_check("no_unneeded_inspect_scene", len(inspect_calls) == 0)

    light_calls = by_name.get("set_stacklight", [])
    add_check("called_set_stacklight_once", len(light_calls) == 1)
    add_check(
        f"stacklight_{expected['light']}",
        len(light_calls) == 1 and _normal_text(_tool_args(light_calls[0]).get("color")) == expected["light"],
    )
    add_check(
        "stacklight_room_id",
        len(light_calls) == 1 and str(_tool_args(light_calls[0]).get("room_id")) == expected_room_id,
    )

    resupply_items = [
        _normal_text(_tool_args(call).get("item_name"))
        for call in by_name.get("request_resupply", [])
    ]
    resupply_calls = by_name.get("request_resupply", [])
    add_check("request_resupply_count", len(resupply_calls) == len(expected_deficit_items))
    add_check("request_resupply_items_exact", sorted(resupply_items) == expected_deficit_items)
    add_check(
        "request_resupply_arguments_valid",
        all(
            _normal_text(_tool_args(call).get("item_name"))
            and str(_tool_args(call).get("room_id")) == expected_room_id
            and _normal_text(_tool_args(call).get("urgency")) in allowed_urgencies
            for call in resupply_calls
        ),
    )

    create_tasks = by_name.get("create_task", [])
    human_reviews = [
        call
        for call in create_tasks
        if _tool_args(call).get("task_type") == "human_review"
    ]
    if expected["sterile_zone_issue"]:
        add_check("create_task_human_review_once", len(create_tasks) == 1 and len(human_reviews) == 1)
        add_check(
            "create_task_case_id",
            len(create_tasks) == 1 and str(_tool_args(create_tasks[0]).get("case_id")) == expected_case_id,
        )
    else:
        add_check("no_create_task_without_sterile_issue", len(create_tasks) == 0)

    passed_checks = [name for name, ok in checks if ok]
    missing = [name for name, ok in checks if not ok]
    score = len(passed_checks) / len(checks) if checks else 1.0
    return {
        "passed": not missing,
        "score": round(score, 3),
        "missing": missing,
        "passed_checks": passed_checks,
        "expected": expected,
        "tool_names": names,
        "accepted_tool_names": [call.get("name") for call in accepted_tool_calls],
        "rejected_tool_names": [call.get("name") for call in rejected_tool_calls],
        "actual_counts": {name: len(calls) for name, calls in sorted(by_name.items())},
        "scored_call_count": len(scored_tool_calls),
        "accepted_call_count": len(accepted_tool_calls),
        "rejected_call_count": len(rejected_tool_calls),
        "scoring": "accepted_only" if accepted_only else "all_attempted_calls",
    }
