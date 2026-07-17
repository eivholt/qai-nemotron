#!/usr/bin/env python3
"""Hospital logistics fixtures and mock tools for edge-agent demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Callable


HOSPITAL_LOGISTICS_INSTRUCTIONS = """\
You are a hospital logistics coordinator running locally on an edge device.
Coordinate non-emergency internal logistics. Do not diagnose, prescribe, or
directly drive robots. Use tools to check facts, reserve shared resources,
assign porters or robots, update job status, notify wards, and escalate
conflicts.

Work step by step:
1. Call get_pending_jobs to see the active queue.
2. For each job you plan, check relevant constraints:
   - cold-chain or blood-product jobs need check_cold_chain_window.
   - route conflicts need check_elevator_status before reserve_elevator.
   - asset-dependent jobs need get_asset_location.
   - uncertain policy choices need query_policy.
3. Assign a porter or robot only after checking that the route and constraints
   work. Do not assign a low-battery robot to an urgent or time-critical job.
   If a tool returns ok=false, the action is not complete; fix the prerequisite
   shown in the tool result and retry the failed action if it is still needed.
4. If a higher-priority event appears, replan: protect the higher-priority job,
   preserve time/cold-chain constraints, and update affected lower-priority jobs.
5. Escalate when no safe assignment exists or when policy requires human
   approval.

Final response: concise operational summary of what changed and why."""


HOSPITAL_TOOL_NAMES = [
    "get_pending_jobs",
    "get_asset_location",
    "check_elevator_status",
    "reserve_elevator",
    "assign_porter",
    "assign_robot",
    "check_cold_chain_window",
    "notify_ward",
    "escalate_to_human",
    "update_job_status",
    "query_policy",
    "read_pending_job",
    "assign_pending_porter",
    "escalate_pending_job",
    "check_pending_cold_chain",
    "reserve_pending_elevator",
]


SCENARIOS: dict[str, dict[str, Any]] = {
    "hospital_S0_read_pending_job": {
        "description": "Shortcut S0: read the single pending logistics job.",
        "mutation_group": "shortcut_single_tool",
        "focused_tools": ["read_pending_job"],
        "task_prompt": "Call read_pending_job with empty arguments to read the pending hospital logistics job.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:00", "jobs": ["JOB-MEAL-S0"]},
        "jobs": {"JOB-MEAL-S0": {"job_id": "JOB-MEAL-S0", "type": "meal_cart", "priority": "normal", "from": "Kitchen", "to": "Ward 1B", "deadline_minutes": 35, "requires_cold_chain": False, "asset_id": "meal-cart-s0", "status": "pending"}},
        "assets": {"meal-cart-s0": {"location": "Kitchen", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-S0": {"available": True, "eta_minutes": 5}},
        "robots": {},
        "expected": {"must_call": ["read_pending_job"], "must_not_call": ["assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"], "arguments": {}},
    },
    "hospital_S1_assign_pending_porter": {
        "description": "Shortcut S1: assign the single pending linen job to an available porter.",
        "mutation_group": "shortcut_single_tool",
        "focused_tools": ["assign_pending_porter"],
        "task_prompt": "Call assign_pending_porter with empty arguments to assign the pending linen delivery to a porter.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:05", "jobs": ["JOB-LINEN-S1"]},
        "jobs": {"JOB-LINEN-S1": {"job_id": "JOB-LINEN-S1", "type": "linen", "priority": "normal", "from": "Supply Room 1A", "to": "Ward 1A", "deadline_minutes": 45, "requires_cold_chain": False, "asset_id": "linen-cart-s1", "status": "pending"}},
        "assets": {"linen-cart-s1": {"location": "Supply Room 1A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-S1": {"available": True, "eta_minutes": 4}},
        "robots": {},
        "expected": {"must_call": ["assign_pending_porter"], "must_not_call": ["reserve_pending_elevator", "escalate_pending_job"], "arguments": {}},
    },
    "hospital_S2_escalate_pending_job": {
        "description": "Shortcut S2: escalate the single pending equipment job because no carrier is available.",
        "mutation_group": "shortcut_single_tool",
        "focused_tools": ["escalate_pending_job"],
        "task_prompt": "Call escalate_pending_job with empty arguments to escalate the pending equipment delivery to a human coordinator.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:10", "jobs": ["JOB-CHAIR-S2"]},
        "jobs": {"JOB-CHAIR-S2": {"job_id": "JOB-CHAIR-S2", "type": "equipment", "priority": "normal", "from": "Storage 2A", "to": "Ward 2A", "deadline_minutes": 30, "requires_cold_chain": False, "asset_id": "wheelchair-s2", "status": "pending"}},
        "assets": {"wheelchair-s2": {"location": "Storage 2A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-S2": {"available": False, "eta_minutes": 45}},
        "robots": {"ROBOT-S2": {"available": False, "battery_percent": 80}},
        "expected": {"must_call": ["escalate_pending_job"], "must_not_call": ["assign_pending_porter", "reserve_pending_elevator"], "arguments": {}},
    },
    "hospital_S3_check_pending_cold_chain": {
        "description": "Shortcut S3: check cold-chain status for the single pending medication tote.",
        "mutation_group": "shortcut_single_tool",
        "focused_tools": ["check_pending_cold_chain"],
        "task_prompt": "Call check_pending_cold_chain with empty arguments to check whether the pending medication tote is still inside its cold-chain window.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:15", "jobs": ["JOB-MED-S3"]},
        "jobs": {"JOB-MED-S3": {"job_id": "JOB-MED-S3", "type": "medication", "priority": "urgent", "from": "Pharmacy", "to": "Ward 3A", "deadline_minutes": 15, "requires_cold_chain": True, "cold_chain_minutes_remaining": 25, "asset_id": "med-tote-s3", "status": "pending"}},
        "assets": {"med-tote-s3": {"location": "Pharmacy", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-S3": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {"must_call": ["check_pending_cold_chain"], "must_not_call": ["assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"], "arguments": {}},
    },
    "hospital_S4_reserve_pending_elevator": {
        "description": "Shortcut S4: reserve the available elevator for the single pending lab sample route.",
        "mutation_group": "shortcut_single_tool",
        "focused_tools": ["reserve_pending_elevator"],
        "task_prompt": "Call reserve_pending_elevator with empty arguments to reserve the elevator for the pending lab sample route.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:20", "jobs": ["JOB-SAMPLE-S4"]},
        "jobs": {"JOB-SAMPLE-S4": {"job_id": "JOB-SAMPLE-S4", "type": "lab_sample", "priority": "urgent", "from": "Ward 4A", "to": "Central Lab", "deadline_minutes": 12, "requires_cold_chain": False, "asset_id": "sample-s4", "status": "pending"}},
        "assets": {"sample-s4": {"location": "Ward 4A", "ready": True}},
        "elevators": {"E4S": {"status": "available", "serves": ["Ward 4A", "Central Lab"]}},
        "porters": {"PORTER-S4": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {"must_call": ["reserve_pending_elevator"], "must_not_call": ["assign_pending_porter", "escalate_pending_job"], "arguments": {}},
    },
    "hospital_C1_choose_assign_porter": {
        "description": "Choice C1: choose assignment rather than escalation for an available porter.",
        "mutation_group": "shortcut_choice",
        "focused_tools": ["assign_pending_porter", "escalate_pending_job"],
        "task_prompt": "A pending linen delivery has an available porter. Because a porter is available, call assign_pending_porter. Do not call escalate_pending_job.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:25", "jobs": ["JOB-LINEN-C1"]},
        "jobs": {"JOB-LINEN-C1": {"job_id": "JOB-LINEN-C1", "type": "linen", "priority": "normal", "from": "Supply Room 1A", "to": "Ward 1A", "deadline_minutes": 45, "requires_cold_chain": False, "asset_id": "linen-cart-c1", "status": "pending"}},
        "assets": {"linen-cart-c1": {"location": "Supply Room 1A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-C1": {"available": True, "eta_minutes": 4}},
        "robots": {},
        "expected": {"must_call": ["assign_pending_porter"], "must_not_call": ["escalate_pending_job"], "arguments": {}},
    },
    "hospital_C2_choose_escalate": {
        "description": "Choice C2: choose escalation rather than assignment when no carrier is available.",
        "mutation_group": "shortcut_choice",
        "focused_tools": ["assign_pending_porter", "escalate_pending_job"],
        "task_prompt": "A pending equipment delivery has no available porter or robot. Because no carrier is available, call escalate_pending_job. Do not call assign_pending_porter.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:30", "jobs": ["JOB-CHAIR-C2"]},
        "jobs": {"JOB-CHAIR-C2": {"job_id": "JOB-CHAIR-C2", "type": "equipment", "priority": "normal", "from": "Storage 2A", "to": "Ward 2A", "deadline_minutes": 30, "requires_cold_chain": False, "asset_id": "wheelchair-c2", "status": "pending"}},
        "assets": {"wheelchair-c2": {"location": "Storage 2A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-C2": {"available": False, "eta_minutes": 45}},
        "robots": {"ROBOT-C2": {"available": False, "battery_percent": 80}},
        "expected": {"must_call": ["escalate_pending_job"], "must_not_call": ["assign_pending_porter"], "arguments": {}},
    },
    "hospital_C3_choose_cold_chain_check": {
        "description": "Choice C3: choose cold-chain check before assignment for medication.",
        "mutation_group": "shortcut_choice",
        "focused_tools": ["check_pending_cold_chain", "assign_pending_porter"],
        "task_prompt": "A pending medication tote requires cold-chain verification before assignment. First call check_pending_cold_chain. Assignment after a successful check is allowed.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:35", "jobs": ["JOB-MED-C3"]},
        "jobs": {"JOB-MED-C3": {"job_id": "JOB-MED-C3", "type": "medication", "priority": "urgent", "from": "Pharmacy", "to": "Ward 3A", "deadline_minutes": 15, "requires_cold_chain": True, "cold_chain_minutes_remaining": 25, "asset_id": "med-tote-c3", "status": "pending"}},
        "assets": {"med-tote-c3": {"location": "Pharmacy", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-C3": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {"must_call": ["check_pending_cold_chain"], "optional_call": ["assign_pending_porter"], "must_not_call": ["escalate_pending_job"], "arguments": {}},
    },
    "hospital_C4_choose_elevator_reservation": {
        "description": "Choice C4: choose elevator reservation rather than escalation for an available route.",
        "mutation_group": "shortcut_choice",
        "focused_tools": ["reserve_pending_elevator", "escalate_pending_job"],
        "task_prompt": "A pending lab sample route has an available elevator. Choose the correct available tool to reserve that elevator. Do not escalate when the route is available.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:40", "jobs": ["JOB-SAMPLE-C4"]},
        "jobs": {"JOB-SAMPLE-C4": {"job_id": "JOB-SAMPLE-C4", "type": "lab_sample", "priority": "urgent", "from": "Ward 4A", "to": "Central Lab", "deadline_minutes": 12, "requires_cold_chain": False, "asset_id": "sample-c4", "status": "pending"}},
        "assets": {"sample-c4": {"location": "Ward 4A", "ready": True}},
        "elevators": {"E4C": {"status": "available", "serves": ["Ward 4A", "Central Lab"]}},
        "porters": {"PORTER-C4": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {"must_call": ["reserve_pending_elevator"], "must_not_call": ["escalate_pending_job"], "arguments": {}},
    },    "hospital_O1_assign_with_many_options": {
        "description": "Option O1: choose porter assignment from four available hospital tools.",
        "mutation_group": "shortcut_multi_option",
        "focused_tools": ["read_pending_job", "assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"],
        "task_prompt": "A pending linen delivery has an available porter, no elevator need, and no emergency conflict. Choose the correct tool to start the delivery. Do not escalate or reserve an elevator.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:45", "jobs": ["JOB-LINEN-O1"]},
        "jobs": {"JOB-LINEN-O1": {"job_id": "JOB-LINEN-O1", "type": "linen", "priority": "normal", "from": "Supply Room 1A", "to": "Ward 1A", "deadline_minutes": 45, "requires_cold_chain": False, "asset_id": "linen-cart-o1", "status": "pending"}},
        "assets": {"linen-cart-o1": {"location": "Supply Room 1A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-O1": {"available": True, "eta_minutes": 4}},
        "robots": {},
        "expected": {"must_call": ["assign_pending_porter"], "optional_before_must": ["read_pending_job"], "must_not_call": ["reserve_pending_elevator", "escalate_pending_job"], "arguments": {}},
    },
    "hospital_O2_escalate_with_many_options": {
        "description": "Option O2: choose escalation from four available hospital tools.",
        "mutation_group": "shortcut_multi_option",
        "focused_tools": ["read_pending_job", "assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"],
        "task_prompt": "A pending equipment delivery has no available porter, no available robot, and no usable route. Choose the correct tool to get human help. Do not assign or reserve resources.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:50", "jobs": ["JOB-CHAIR-O2"]},
        "jobs": {"JOB-CHAIR-O2": {"job_id": "JOB-CHAIR-O2", "type": "equipment", "priority": "normal", "from": "Storage 2A", "to": "Ward 2A", "deadline_minutes": 30, "requires_cold_chain": False, "asset_id": "wheelchair-o2", "status": "pending"}},
        "assets": {"wheelchair-o2": {"location": "Storage 2A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-O2": {"available": False, "eta_minutes": 45}},
        "robots": {"ROBOT-O2": {"available": False, "battery_percent": 80}},
        "expected": {"must_call": ["escalate_pending_job"], "optional_before_must": ["read_pending_job"], "must_not_call": ["assign_pending_porter", "reserve_pending_elevator"], "arguments": {}},
    },
    "hospital_O3_cold_chain_with_many_options": {
        "description": "Option O3: choose cold-chain verification from four available hospital tools.",
        "mutation_group": "shortcut_multi_option",
        "focused_tools": ["read_pending_job", "check_pending_cold_chain", "assign_pending_porter", "escalate_pending_job"],
        "task_prompt": "A pending medication tote requires cold-chain verification before any assignment. Choose the correct first tool to verify the tote. Do not assign before the check.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "09:55", "jobs": ["JOB-MED-O3"]},
        "jobs": {"JOB-MED-O3": {"job_id": "JOB-MED-O3", "type": "medication", "priority": "urgent", "from": "Pharmacy", "to": "Ward 3A", "deadline_minutes": 15, "requires_cold_chain": True, "cold_chain_minutes_remaining": 25, "asset_id": "med-tote-o3", "status": "pending"}},
        "assets": {"med-tote-o3": {"location": "Pharmacy", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-O3": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {"must_call": ["check_pending_cold_chain"], "optional_before_must": ["read_pending_job"], "must_not_call": ["assign_pending_porter", "escalate_pending_job"], "arguments": {}},
    },
    "hospital_O4_elevator_with_many_options": {
        "description": "Option O4: choose elevator reservation from four available hospital tools.",
        "mutation_group": "shortcut_multi_option",
        "focused_tools": ["read_pending_job", "assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"],
        "task_prompt": "A pending lab sample route needs an elevator, and an elevator is available. Choose the correct tool to reserve the route before assignment. Do not assign first and do not escalate.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "10:00", "jobs": ["JOB-SAMPLE-O4"]},
        "jobs": {"JOB-SAMPLE-O4": {"job_id": "JOB-SAMPLE-O4", "type": "lab_sample", "priority": "urgent", "from": "Ward 4A", "to": "Central Lab", "deadline_minutes": 12, "requires_cold_chain": False, "asset_id": "sample-o4", "status": "pending"}},
        "assets": {"sample-o4": {"location": "Ward 4A", "ready": True}},
        "elevators": {"E4O": {"status": "available", "serves": ["Ward 4A", "Central Lab"]}},
        "porters": {"PORTER-O4": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {"must_call": ["reserve_pending_elevator"], "optional_before_must": ["read_pending_job"], "must_not_call": ["assign_pending_porter", "escalate_pending_job"], "arguments": {}},
    },
    "hospital_O5_read_only_with_many_options": {
        "description": "Option O5: choose read-only queue inspection from four available hospital tools.",
        "mutation_group": "shortcut_multi_option",
        "focused_tools": ["read_pending_job", "assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"],
        "task_prompt": "The coordinator only needs a queue snapshot for handoff notes. Choose the tool that reads the pending job. Do not assign, reserve, or escalate anything.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "10:05", "jobs": ["JOB-MEAL-O5"]},
        "jobs": {"JOB-MEAL-O5": {"job_id": "JOB-MEAL-O5", "type": "meal_cart", "priority": "normal", "from": "Kitchen", "to": "Ward 1B", "deadline_minutes": 35, "requires_cold_chain": False, "asset_id": "meal-cart-o5", "status": "pending"}},
        "assets": {"meal-cart-o5": {"location": "Kitchen", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-O5": {"available": True, "eta_minutes": 5}},
        "robots": {},
        "expected": {"must_call": ["read_pending_job"], "must_not_call": ["assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"], "arguments": {}},
    },    "hospital_P1_assign_from_action_options": {
        "description": "Action option P1: choose porter assignment from three action tools.",
        "mutation_group": "shortcut_action_option",
        "focused_tools": ["assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"],
        "task_prompt": "A pending linen delivery has an available porter and stays on the same floor. Call assign_pending_porter. Do not reserve an elevator or escalate.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "10:10", "jobs": ["JOB-LINEN-P1"]},
        "jobs": {"JOB-LINEN-P1": {"job_id": "JOB-LINEN-P1", "type": "linen", "priority": "normal", "from": "Supply Room 1A", "to": "Ward 1A", "deadline_minutes": 45, "requires_cold_chain": False, "asset_id": "linen-cart-p1", "status": "pending"}},
        "assets": {"linen-cart-p1": {"location": "Supply Room 1A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-P1": {"available": True, "eta_minutes": 4}},
        "robots": {},
        "expected": {"must_call": ["assign_pending_porter"], "must_not_call": ["reserve_pending_elevator", "escalate_pending_job"], "arguments": {}},
    },
    "hospital_P2_escalate_from_action_options": {
        "description": "Action option P2: choose escalation from three action tools.",
        "mutation_group": "shortcut_action_option",
        "focused_tools": ["assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"],
        "task_prompt": "A pending equipment delivery has no available porter, no robot, and no safe route. Call escalate_pending_job. Do not assign or reserve resources.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "10:15", "jobs": ["JOB-CHAIR-P2"]},
        "jobs": {"JOB-CHAIR-P2": {"job_id": "JOB-CHAIR-P2", "type": "equipment", "priority": "normal", "from": "Storage 2A", "to": "Ward 2A", "deadline_minutes": 30, "requires_cold_chain": False, "asset_id": "wheelchair-p2", "status": "pending"}},
        "assets": {"wheelchair-p2": {"location": "Storage 2A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-P2": {"available": False, "eta_minutes": 45}},
        "robots": {"ROBOT-P2": {"available": False, "battery_percent": 80}},
        "expected": {"must_call": ["escalate_pending_job"], "must_not_call": ["assign_pending_porter", "reserve_pending_elevator"], "arguments": {}},
    },
    "hospital_P3_cold_chain_from_action_options": {
        "description": "Action option P3: choose cold-chain verification from three action tools.",
        "mutation_group": "shortcut_action_option",
        "focused_tools": ["check_pending_cold_chain", "assign_pending_porter", "escalate_pending_job"],
        "task_prompt": "A pending medication tote must have cold-chain verified before any assignment. Call check_pending_cold_chain first. Assignment after a successful check is allowed.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "10:20", "jobs": ["JOB-MED-P3"]},
        "jobs": {"JOB-MED-P3": {"job_id": "JOB-MED-P3", "type": "medication", "priority": "urgent", "from": "Pharmacy", "to": "Ward 3A", "deadline_minutes": 15, "requires_cold_chain": True, "cold_chain_minutes_remaining": 25, "asset_id": "med-tote-p3", "status": "pending"}},
        "assets": {"med-tote-p3": {"location": "Pharmacy", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-P3": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {"must_call": ["check_pending_cold_chain"], "optional_call": ["assign_pending_porter"], "must_not_call": ["escalate_pending_job"], "arguments": {}},
    },
    "hospital_P4_elevator_from_action_options": {
        "description": "Action option P4: choose elevator reservation from three action tools.",
        "mutation_group": "shortcut_action_option",
        "focused_tools": ["assign_pending_porter", "reserve_pending_elevator", "escalate_pending_job"],
        "task_prompt": "A pending lab sample route needs an elevator and an elevator is available. Call reserve_pending_elevator before assignment. Do not assign first and do not escalate.",
        "scoring_mode": "idempotent_action",
        "event": {"kind": "new_jobs", "time": "10:25", "jobs": ["JOB-SAMPLE-P4"]},
        "jobs": {"JOB-SAMPLE-P4": {"job_id": "JOB-SAMPLE-P4", "type": "lab_sample", "priority": "urgent", "from": "Ward 4A", "to": "Central Lab", "deadline_minutes": 12, "requires_cold_chain": False, "asset_id": "sample-p4", "status": "pending"}},
        "assets": {"sample-p4": {"location": "Ward 4A", "ready": True}},
        "elevators": {"E4P": {"status": "available", "serves": ["Ward 4A", "Central Lab"]}},
        "porters": {"PORTER-P4": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {"must_call": ["reserve_pending_elevator"], "must_not_call": ["assign_pending_porter", "escalate_pending_job"], "arguments": {}},
    },    "hospital_E0_queue_snapshot": {
        "description": "Easy E0: read the current logistics queue and summarize the one pending job.",
        "mutation_group": "easy_focused",
        "focused_tools": ["get_pending_jobs"],
        "task_prompt": "Read the current hospital logistics queue. Then briefly summarize the pending job. Do not assign or update anything.",
        "event": {"kind": "new_jobs", "time": "07:55", "jobs": ["JOB-MEAL-E0"]},
        "jobs": {
            "JOB-MEAL-E0": {
                "job_id": "JOB-MEAL-E0",
                "type": "meal_cart",
                "priority": "normal",
                "from": "Kitchen",
                "to": "Ward 1B",
                "deadline_minutes": 35,
                "requires_cold_chain": False,
                "asset_id": "meal-cart-e0",
                "status": "pending",
            }
        },
        "assets": {"meal-cart-e0": {"location": "Kitchen", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-E0": {"available": True, "eta_minutes": 5}},
        "robots": {},
        "expected": {
            "must_call": ["get_pending_jobs"],
            "must_not_call": ["assign_porter", "assign_robot", "reserve_elevator", "escalate_to_human", "update_job_status"],
            "arguments": {},
        },
    },
    "hospital_E1_assign_porter": {
        "description": "Easy E1: assign an available porter to one same-floor linen job.",
        "mutation_group": "easy_focused",
        "focused_tools": ["get_pending_jobs", "assign_porter"],
        "task_prompt": "Assign an available porter to the pending linen delivery. Same floor, no elevator needed. Stop after the assignment tool succeeds.",
        "event": {"kind": "new_jobs", "time": "08:05", "jobs": ["JOB-LINEN-E1"]},
        "jobs": {
            "JOB-LINEN-E1": {
                "job_id": "JOB-LINEN-E1",
                "type": "linen",
                "priority": "normal",
                "from": "Supply Room 1A",
                "to": "Ward 1A",
                "deadline_minutes": 45,
                "requires_cold_chain": False,
                "asset_id": "linen-cart-e1",
                "status": "pending",
            }
        },
        "assets": {"linen-cart-e1": {"location": "Supply Room 1A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-E1": {"available": True, "eta_minutes": 4}},
        "robots": {},
        "expected": {
            "must_call": ["assign_porter"],
            "must_not_call": ["assign_robot", "reserve_elevator", "escalate_to_human"],
            "arguments": {"assign_porter": {"job_id": "JOB-LINEN-E1"}},
        },
    },
    "hospital_E2_blocked_escalate": {
        "description": "Easy E2: no carrier is available, so escalate one equipment job.",
        "mutation_group": "easy_focused",
        "focused_tools": ["get_pending_jobs", "escalate_to_human"],
        "task_prompt": "Check the queue. The equipment job has no available porter or robot. Escalate it to a human coordinator and stop.",
        "event": {"kind": "new_jobs", "time": "08:15", "jobs": ["JOB-CHAIR-E2"]},
        "jobs": {
            "JOB-CHAIR-E2": {
                "job_id": "JOB-CHAIR-E2",
                "type": "equipment",
                "priority": "normal",
                "from": "Storage 2A",
                "to": "Ward 2A",
                "deadline_minutes": 30,
                "requires_cold_chain": False,
                "asset_id": "wheelchair-e2",
                "status": "pending",
            }
        },
        "assets": {"wheelchair-e2": {"location": "Storage 2A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-E2": {"available": False, "eta_minutes": 45}},
        "robots": {"ROBOT-E2": {"available": False, "battery_percent": 80}},
        "expected": {
            "must_call": ["escalate_to_human"],
            "must_not_call": ["assign_porter", "assign_robot", "reserve_elevator"],
            "arguments": {"escalate_to_human": {"job_id": "JOB-CHAIR-E2"}},
        },
    },
    "hospital_E3_cold_chain_check": {
        "description": "Easy E3: check cold-chain time before assigning a medication tote.",
        "mutation_group": "easy_focused",
        "focused_tools": ["get_pending_jobs", "check_cold_chain_window", "assign_porter"],
        "task_prompt": "A medication tote needs cold-chain verification. Check the queue, check the cold-chain window, then assign an available porter if the tool says the window is ok.",
        "event": {"kind": "new_jobs", "time": "08:25", "jobs": ["JOB-MED-E3"]},
        "jobs": {
            "JOB-MED-E3": {
                "job_id": "JOB-MED-E3",
                "type": "medication",
                "priority": "urgent",
                "from": "Pharmacy",
                "to": "Ward 3A",
                "deadline_minutes": 15,
                "requires_cold_chain": True,
                "cold_chain_minutes_remaining": 25,
                "asset_id": "med-tote-e3",
                "status": "pending",
            }
        },
        "assets": {"med-tote-e3": {"location": "Pharmacy", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-E3": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {
            "must_call": ["check_cold_chain_window", "assign_porter"],
            "must_not_call": ["assign_robot", "reserve_elevator", "escalate_to_human"],
            "arguments": {
                "check_cold_chain_window": {"job_id": "JOB-MED-E3"},
                "assign_porter": {"job_id": "JOB-MED-E3"},
            },
        },
    },
    "hospital_E4_reserve_elevator": {
        "description": "Easy E4: check and reserve one elevator for a lab sample route.",
        "mutation_group": "easy_focused",
        "focused_tools": ["get_pending_jobs", "check_elevator_status", "reserve_elevator"],
        "task_prompt": "A lab sample route needs an elevator. Check the queue, check elevator status for the route, reserve the available elevator, and stop.",
        "event": {"kind": "new_jobs", "time": "08:35", "jobs": ["JOB-SAMPLE-E4"]},
        "jobs": {
            "JOB-SAMPLE-E4": {
                "job_id": "JOB-SAMPLE-E4",
                "type": "lab_sample",
                "priority": "urgent",
                "from": "Ward 4A",
                "to": "Central Lab",
                "deadline_minutes": 12,
                "requires_cold_chain": False,
                "asset_id": "sample-e4",
                "status": "pending",
            }
        },
        "assets": {"sample-e4": {"location": "Ward 4A", "ready": True}},
        "elevators": {"E4": {"status": "available", "serves": ["Ward 4A", "Central Lab"]}},
        "porters": {"PORTER-E4": {"available": True, "eta_minutes": 3}},
        "robots": {},
        "expected": {
            "must_call": ["check_elevator_status", "reserve_elevator"],
            "must_not_call": ["assign_robot", "escalate_to_human"],
            "arguments": {"reserve_elevator": {"elevator_id": "E4", "job_id": "JOB-SAMPLE-E4"}},
        },
    },    "hospital_M0_read_queue_only": {
        "description": "Mutation M0: read one pending linen job and summarize it.",
        "mutation_group": "low_complexity",
        "event": {
            "kind": "new_jobs",
            "time": "08:00",
            "jobs": ["JOB-LINEN-M0"],
        },
        "jobs": {
            "JOB-LINEN-M0": {
                "job_id": "JOB-LINEN-M0",
                "type": "linen",
                "priority": "normal",
                "from": "Laundry",
                "to": "Ward 1A",
                "deadline_minutes": 60,
                "requires_cold_chain": False,
                "asset_id": "linen-cart-m0",
                "status": "pending",
            }
        },
        "assets": {"linen-cart-m0": {"location": "Laundry", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-M0": {"available": True, "eta_minutes": 6}},
        "robots": {},
        "expected": {
            "must_call": ["get_pending_jobs"],
            "must_not_call": ["assign_robot", "reserve_elevator", "escalate_to_human"],
            "arguments": {},
        },
    },
    "hospital_M1_direct_porter_no_elevator": {
        "description": "Mutation M1: assign a porter for a simple linen delivery on the same floor.",
        "mutation_group": "low_complexity",
        "event": {
            "kind": "new_jobs",
            "time": "08:10",
            "jobs": ["JOB-LINEN-M1"],
        },
        "jobs": {
            "JOB-LINEN-M1": {
                "job_id": "JOB-LINEN-M1",
                "type": "linen",
                "priority": "normal",
                "from": "Supply Room 1A",
                "to": "Ward 1A",
                "deadline_minutes": 45,
                "requires_cold_chain": False,
                "asset_id": "linen-cart-m1",
                "status": "pending",
            }
        },
        "assets": {"linen-cart-m1": {"location": "Supply Room 1A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-M1": {"available": True, "eta_minutes": 4}},
        "robots": {},
        "expected": {
            "must_call": [
                "get_pending_jobs",
                "assign_porter",
                "update_job_status",
                "notify_ward",
            ],
            "must_not_call": ["assign_robot", "reserve_elevator", "escalate_to_human"],
            "arguments": {
                "assign_porter": {"porter_id": "PORTER-M1", "job_id": "JOB-LINEN-M1"},
                "update_job_status": {"job_id": "JOB-LINEN-M1", "status": "assigned"},
            },
        },
    },
    "hospital_M2_asset_then_porter_no_elevator": {
        "description": "Mutation M2: check asset readiness, then assign a porter for a same-floor equipment move.",
        "mutation_group": "low_complexity",
        "event": {
            "kind": "new_jobs",
            "time": "08:30",
            "jobs": ["JOB-PUMP-M2"],
        },
        "jobs": {
            "JOB-PUMP-M2": {
                "job_id": "JOB-PUMP-M2",
                "type": "equipment",
                "priority": "normal",
                "from": "Equipment Bay 2B",
                "to": "Ward 2B",
                "deadline_minutes": 50,
                "requires_cold_chain": False,
                "asset_id": "pump-m2",
                "status": "pending",
            }
        },
        "assets": {"pump-m2": {"location": "Equipment Bay 2B", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-M2": {"available": True, "eta_minutes": 5}},
        "robots": {},
        "expected": {
            "must_call": [
                "get_pending_jobs",
                "get_asset_location",
                "assign_porter",
                "update_job_status",
                "notify_ward",
            ],
            "must_not_call": ["assign_robot", "reserve_elevator", "escalate_to_human"],
            "arguments": {
                "get_asset_location": {"asset_id": "pump-m2"},
                "assign_porter": {"porter_id": "PORTER-M2", "job_id": "JOB-PUMP-M2"},
                "update_job_status": {"job_id": "JOB-PUMP-M2", "status": "assigned"},
            },
        },
    },
    "hospital_M3_simple_blocked_escalation": {
        "description": "Mutation M3: one normal equipment job has no available carrier, so escalate and mark blocked.",
        "mutation_group": "low_complexity",
        "event": {
            "kind": "new_jobs",
            "time": "08:45",
            "jobs": ["JOB-CHAIR-M3"],
        },
        "jobs": {
            "JOB-CHAIR-M3": {
                "job_id": "JOB-CHAIR-M3",
                "type": "equipment",
                "priority": "normal",
                "from": "Storage 3A",
                "to": "Ward 3A",
                "deadline_minutes": 30,
                "requires_cold_chain": False,
                "asset_id": "wheelchair-m3",
                "status": "pending",
            }
        },
        "assets": {"wheelchair-m3": {"location": "Storage 3A", "ready": True}},
        "elevators": {},
        "porters": {"PORTER-M3": {"available": False, "eta_minutes": 40}},
        "robots": {},
        "expected": {
            "must_call": [
                "get_pending_jobs",
                "escalate_to_human",
                "update_job_status",
            ],
            "must_not_call": ["assign_robot", "reserve_elevator"],
            "arguments": {
                "escalate_to_human": {"job_id": "JOB-CHAIR-M3"},
                "update_job_status": {"job_id": "JOB-CHAIR-M3", "status": "blocked"},
            },
        },
    },    "hospital_L0_simple_linen_delivery": {
        "description": "Simple linen cart delivery with one available elevator and porter.",
        "event": {
            "kind": "new_jobs",
            "time": "08:20",
            "jobs": ["JOB-LINEN-1"],
        },
        "jobs": {
            "JOB-LINEN-1": {
                "job_id": "JOB-LINEN-1",
                "type": "linen",
                "priority": "normal",
                "from": "Laundry",
                "to": "Ward 2A",
                "deadline_minutes": 40,
                "requires_cold_chain": False,
                "asset_id": "linen-cart-1",
                "status": "pending",
            }
        },
        "assets": {"linen-cart-1": {"location": "Laundry loading bay", "ready": True}},
        "elevators": {"E1": {"status": "available", "serves": ["Laundry", "Ward 2A"]}},
        "porters": {"PORTER-A": {"available": True, "eta_minutes": 5}},
        "robots": {},
        "expected": {
            "must_call": [
                "get_pending_jobs",
                "check_elevator_status",
                "reserve_elevator",
                "assign_porter",
                "update_job_status",
                "notify_ward",
            ],
            "must_not_call": ["escalate_to_human"],
            "arguments": {
                "reserve_elevator": {"elevator_id": "E1"},
                "assign_porter": {"porter_id": "PORTER-A", "job_id": "JOB-LINEN-1"},
            },
        },
    },
    "hospital_L1_sample_elevator_out": {
        "description": "Blood sample must reach lab in 12 minutes; one elevator is out.",
        "event": {
            "kind": "new_jobs",
            "time": "09:00",
            "jobs": ["JOB-SAMPLE-7"],
        },
        "jobs": {
            "JOB-SAMPLE-7": {
                "job_id": "JOB-SAMPLE-7",
                "type": "lab_sample",
                "priority": "urgent",
                "from": "Ward 4B",
                "to": "Central Lab",
                "deadline_minutes": 12,
                "requires_cold_chain": False,
                "asset_id": "sample-tube-7",
                "status": "pending",
            }
        },
        "assets": {
            "sample-tube-7": {"location": "Ward 4B nurses station", "ready": True}
        },
        "elevators": {
            "E1": {"status": "out", "serves": ["Ward 4B", "Central Lab"]},
            "E2": {"status": "available", "serves": ["Ward 4B", "Central Lab"]},
        },
        "porters": {"PORTER-A": {"available": True, "eta_minutes": 3}},
        "robots": {"ROBOT-1": {"available": True, "battery_percent": 78}},
        "expected": {
            "must_call": [
                "get_pending_jobs",
                "get_asset_location",
                "check_elevator_status",
                "reserve_elevator",
                "assign_porter",
                "update_job_status",
                "notify_ward",
            ],
            "must_not_call": ["assign_robot", "escalate_to_human"],
            "arguments": {
                "reserve_elevator": {"elevator_id": "E2"},
                "assign_porter": {"porter_id": "PORTER-A", "job_id": "JOB-SAMPLE-7"},
            },
        },
    },
    "hospital_L2_cold_chain_robot_low": {
        "description": "Medication tote has cold-chain limit; robot is low battery.",
        "event": {
            "kind": "new_jobs",
            "time": "10:10",
            "jobs": ["JOB-MED-22"],
        },
        "jobs": {
            "JOB-MED-22": {
                "job_id": "JOB-MED-22",
                "type": "medication_tote",
                "priority": "high",
                "from": "Pharmacy",
                "to": "Ward 6A",
                "deadline_minutes": 18,
                "requires_cold_chain": True,
                "cold_chain_minutes_remaining": 16,
                "asset_id": "med-tote-22",
                "status": "pending",
            }
        },
        "assets": {"med-tote-22": {"location": "Pharmacy cold room", "ready": True}},
        "elevators": {"E2": {"status": "available", "serves": ["Pharmacy", "Ward 6A"]}},
        "porters": {"PORTER-B": {"available": True, "eta_minutes": 4}},
        "robots": {"ROBOT-2": {"available": True, "battery_percent": 12}},
        "expected": {
            "must_call": [
                "get_pending_jobs",
                "check_cold_chain_window",
                "check_elevator_status",
                "reserve_elevator",
                "assign_porter",
                "update_job_status",
            ],
            "must_not_call": ["assign_robot", "escalate_to_human"],
            "arguments": {
                "assign_porter": {"porter_id": "PORTER-B", "job_id": "JOB-MED-22"}
            },
        },
    },
    "hospital_L3_replan_priority_medication": {
        "description": "Replan when a higher-priority medication delivery appears mid-plan.",
        "event": {
            "kind": "replan",
            "time": "11:30",
            "jobs": ["JOB-LINEN-5", "JOB-MED-STAT-9"],
            "new_priority_job": "JOB-MED-STAT-9",
        },
        "jobs": {
            "JOB-LINEN-5": {
                "job_id": "JOB-LINEN-5",
                "type": "linen",
                "priority": "normal",
                "from": "Laundry",
                "to": "Ward 3C",
                "deadline_minutes": 45,
                "requires_cold_chain": False,
                "asset_id": "linen-cart-5",
                "status": "planned",
            },
            "JOB-MED-STAT-9": {
                "job_id": "JOB-MED-STAT-9",
                "type": "medication_tote",
                "priority": "stat",
                "from": "Pharmacy",
                "to": "ICU",
                "deadline_minutes": 8,
                "requires_cold_chain": True,
                "cold_chain_minutes_remaining": 10,
                "asset_id": "stat-med-9",
                "status": "pending",
            },
        },
        "assets": {
            "linen-cart-5": {"location": "Laundry loading bay", "ready": True},
            "stat-med-9": {"location": "Pharmacy handoff desk", "ready": True},
        },
        "elevators": {
            "E2": {"status": "reserved", "reserved_for": "JOB-LINEN-5"},
            "E3": {"status": "available", "serves": ["Pharmacy", "ICU", "Ward 3C"]},
        },
        "porters": {"PORTER-C": {"available": True, "eta_minutes": 2}},
        "robots": {"ROBOT-3": {"available": True, "battery_percent": 66}},
        "policies": {
            "priority_preemption": "STAT medication jobs may preempt normal linen jobs. Notify affected ward and reschedule linen."
        },
        "expected": {
            "must_call": [
                "get_pending_jobs",
                "query_policy",
                "check_cold_chain_window",
                "check_elevator_status",
                "reserve_elevator",
                "assign_porter",
                "update_job_status",
                "notify_ward",
            ],
            "must_not_call": ["assign_robot", "escalate_to_human"],
            "arguments": {
                "assign_porter": {"porter_id": "PORTER-C", "job_id": "JOB-MED-STAT-9"},
                "update_job_status": {"job_id": "JOB-LINEN-5", "status": "delayed"},
            },
        },
    },
    "hospital_L4_blood_product_conflict_escalate": {
        "description": "Blood product has tight window and no route is available; escalate.",
        "event": {
            "kind": "new_jobs",
            "time": "13:05",
            "jobs": ["JOB-BLOOD-3"],
        },
        "jobs": {
            "JOB-BLOOD-3": {
                "job_id": "JOB-BLOOD-3",
                "type": "blood_product",
                "priority": "stat",
                "from": "Blood Bank",
                "to": "OR 2",
                "deadline_minutes": 7,
                "requires_cold_chain": True,
                "cold_chain_minutes_remaining": 6,
                "asset_id": "blood-3",
                "status": "pending",
            }
        },
        "assets": {"blood-3": {"location": "Blood Bank release fridge", "ready": True}},
        "elevators": {
            "E1": {"status": "out", "serves": ["Blood Bank", "OR 2"]},
            "E2": {"status": "reserved", "reserved_for": "emergency_transfer"},
        },
        "porters": {"PORTER-D": {"available": False, "eta_minutes": 12}},
        "robots": {"ROBOT-4": {"available": True, "battery_percent": 9}},
        "policies": {
            "blood_product_exception": "Blood products with less than 10 minutes remaining require human escalation if no verified route and carrier are available."
        },
        "expected": {
            "must_call": [
                "get_pending_jobs",
                "check_cold_chain_window",
                "check_elevator_status",
                "query_policy",
                "escalate_to_human",
                "update_job_status",
            ],
            "must_not_call": ["assign_robot", "assign_porter", "reserve_elevator"],
            "arguments": {
                "escalate_to_human": {"job_id": "JOB-BLOOD-3"},
                "update_job_status": {"job_id": "JOB-BLOOD-3", "status": "blocked"},
            },
        },
    },
}


@dataclass
class HospitalLogisticsRuntime:
    scenario: dict[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)

    def log_call(self, name: str, arguments: dict[str, Any], func: Callable[[], Any]) -> Any:
        read_only_tools = {
            "get_pending_jobs",
            "get_asset_location",
            "check_elevator_status",
            "check_cold_chain_window",
            "query_policy",
            "notify_ward",
            "read_pending_job",
            "assign_pending_porter",
            "escalate_pending_job",
            "check_pending_cold_chain",
            "reserve_pending_elevator",
        }
        if name in read_only_tools:
            for previous in reversed(self.calls):
                if previous["name"] == name and previous.get("arguments", {}) == arguments:
                    result = {
                        "ok": False,
                        "duplicate_call": True,
                        "reason": "no_new_information",
                        "previous_result": previous.get("result", {}),
                        "next_step": "Use the facts already returned and choose the next required action.",
                    }
                    self.calls.append({"name": name, "arguments": arguments, "result": result})
                    return result
        result = func()
        self.calls.append({"name": name, "arguments": arguments, "result": result})
        return result

    def get_pending_jobs(self) -> dict[str, Any]:
        return {
            "event": self.scenario["event"],
            "jobs": list(self.scenario["jobs"].values()),
            "available_porters": [
                {"porter_id": porter_id, **info}
                for porter_id, info in self.scenario.get("porters", {}).items()
            ],
            "available_robots": [
                {"robot_id": robot_id, **info}
                for robot_id, info in self.scenario.get("robots", {}).items()
            ],
        }

    def resolve_job_id(self, job_id: str) -> str:
        jobs = self.scenario.get("jobs", {})
        if job_id in jobs:
            return job_id
        matches = get_close_matches(job_id, list(jobs), n=1, cutoff=0.72)
        if matches:
            return matches[0]
        normalized = job_id.strip().upper()
        for candidate in jobs:
            if normalized and (candidate.endswith(normalized) or normalized in candidate):
                return candidate
        if len(jobs) == 1:
            return next(iter(jobs))
        return job_id

    def pending_job_id(self) -> str:
        return next(iter(self.scenario.get("jobs", {})), "pending")

    def read_pending_job(self) -> dict[str, Any]:
        queue = self.get_pending_jobs()
        return {"ok": True, "job": queue["jobs"][0] if queue["jobs"] else None, "available_porters": queue["available_porters"], "available_robots": queue["available_robots"]}

    def assign_pending_porter(self) -> dict[str, Any]:
        return self.assign_porter("available", self.pending_job_id())

    def escalate_pending_job(self) -> dict[str, Any]:
        return self.escalate_to_human(self.pending_job_id(), "no safe assignment available", "normal")

    def check_pending_cold_chain(self) -> dict[str, Any]:
        return self.check_cold_chain_window(self.pending_job_id())

    def reserve_pending_elevator(self) -> dict[str, Any]:
        return self.reserve_elevator("available", self.pending_job_id(), 5)
    def get_asset_location(self, asset_id: str) -> dict[str, Any]:
        assets = self.scenario.get("assets", {})
        resolved_asset_id = asset_id
        if asset_id not in assets:
            matches = get_close_matches(asset_id, list(assets), n=1, cutoff=0.82)
            if matches:
                resolved_asset_id = matches[0]
        info = assets.get(resolved_asset_id, {"location": "unknown", "ready": False})
        return {
            "asset_id": resolved_asset_id,
            "requested_asset_id": asset_id,
            "corrected": resolved_asset_id != asset_id,
            **info,
        }

    def check_elevator_status(self, from_unit: str, to_unit: str) -> dict[str, Any]:
        if not from_unit or not to_unit:
            first_job = next(iter(self.scenario.get("jobs", {}).values()), {})
            from_unit = from_unit or str(first_job.get("from", ""))
            to_unit = to_unit or str(first_job.get("to", ""))
        elevators = []
        recommended = None
        for elevator_id, info in self.scenario.get("elevators", {}).items():
            entry = {"elevator_id": elevator_id, **info}
            elevators.append(entry)
            if recommended is None and info.get("status") == "available":
                recommended = elevator_id
        return {"from": from_unit, "to": to_unit, "elevators": elevators, "recommended_elevator_id": recommended}

    def reserve_elevator(self, elevator_id: str, job_id: str, duration_minutes: int = 5) -> dict[str, Any]:
        job_id = self.resolve_job_id(job_id)
        if elevator_id.lower() in {"any", "available", "recommended", "best"}:
            elevator_id = next(
                (candidate for candidate, info in self.scenario.get("elevators", {}).items() if info.get("status") == "available"),
                elevator_id,
            )
        info = self.scenario.get("elevators", {}).get(elevator_id, {})
        ok = info.get("status") == "available" and job_id in self.scenario.get("jobs", {})
        if ok:
            info["status"] = "reserved"
            info["reserved_for"] = job_id
        return {"ok": ok, "elevator_id": elevator_id, "job_id": job_id, "duration_minutes": duration_minutes}

    def assign_porter(self, porter_id: str, job_id: str) -> dict[str, Any]:
        job_id = self.resolve_job_id(job_id)
        if porter_id.lower() in {"any", "nearest", "nearest_available", "available", "best"}:
            porter_id = next(
                (candidate for candidate, info in self.scenario.get("porters", {}).items() if info.get("available")),
                porter_id,
            )
        porter = self.scenario.get("porters", {}).get(porter_id, {})
        job = self.scenario.get("jobs", {}).get(job_id, {})
        route_elevators = [
            (elevator_id, info)
            for elevator_id, info in self.scenario.get("elevators", {}).items()
            if job.get("from") in info.get("serves", []) and job.get("to") in info.get("serves", [])
        ]
        reserved = next((elevator_id for elevator_id, info in route_elevators if info.get("reserved_for") == job_id), None)
        recommended = next((elevator_id for elevator_id, info in route_elevators if info.get("status") == "available"), None)
        if route_elevators and not reserved:
            return {
                "ok": False,
                "porter_id": porter_id,
                "job_id": job_id,
                "reason": "route_requires_elevator_reservation_first",
                "recommended_elevator_id": recommended,
            }
        ok = bool(porter.get("available"))
        return {"ok": ok, "porter_id": porter_id, "job_id": job_id, "eta_minutes": porter.get("eta_minutes"), "reserved_elevator_id": reserved}

    def assign_robot(self, robot_id: str, job_id: str) -> dict[str, Any]:
        job_id = self.resolve_job_id(job_id)
        if robot_id.lower() in {"any", "nearest", "nearest_available", "available", "best"}:
            robot_id = next(
                (
                    candidate
                    for candidate, info in self.scenario.get("robots", {}).items()
                    if info.get("available") and int(info.get("battery_percent", 0)) >= 20
                ),
                robot_id,
            )
        robot = self.scenario.get("robots", {}).get(robot_id, {})
        ok = bool(robot.get("available")) and int(robot.get("battery_percent", 0)) >= 20
        return {"ok": ok, "robot_id": robot_id, "job_id": job_id, "battery_percent": robot.get("battery_percent")}

    def check_cold_chain_window(self, job_id: str) -> dict[str, Any]:
        job_id = self.resolve_job_id(job_id)
        job = self.scenario.get("jobs", {}).get(job_id)
        if job is None:
            return {"ok": False, "job_id": job_id, "reason": "unknown_job_id"}
        remaining = int(job.get("cold_chain_minutes_remaining", 999))
        deadline = int(job.get("deadline_minutes", 999))
        return {
            "job_id": job_id,
            "requires_cold_chain": bool(job.get("requires_cold_chain")),
            "minutes_remaining": remaining,
            "deadline_minutes": deadline,
            "ok": remaining >= deadline,
        }

    def notify_ward(self, ward: str, message: str, job_id: str | None = None) -> dict[str, Any]:
        resolved_job_id = self.resolve_job_id(job_id) if job_id else None
        return {"ok": True, "ward": ward, "job_id": resolved_job_id, "message": message}

    def escalate_to_human(self, job_id: str, reason: str, urgency: str) -> dict[str, Any]:
        job_id = self.resolve_job_id(job_id)
        return {"ok": True, "job_id": job_id, "reason": reason, "urgency": urgency, "assignee": "charge_nurse"}

    def update_job_status(self, job_id: str, status: str, note: str = "") -> dict[str, Any]:
        job_id = self.resolve_job_id(job_id)
        ok = job_id in self.scenario.get("jobs", {})
        if ok:
            self.statuses[job_id] = status
        return {"ok": ok, "job_id": job_id, "status": status, "note": note}

    def query_policy(self, topic: str) -> dict[str, Any]:
        policies = self.scenario.get("policies", {})
        return {"topic": topic, "policy": policies.get(topic, "No special policy found. Use normal logistics priority rules.")}


def hospital_scenario(case_id: str) -> dict[str, Any]:
    if case_id not in SCENARIOS:
        raise KeyError(f"Unknown hospital logistics scenario: {case_id}")
    return SCENARIOS[case_id]


def score_hospital_calls(scenario: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    expected = scenario["expected"]
    names = [call["name"] for call in calls]
    missing = [name for name in expected.get("must_call", []) if name not in names]
    forbidden = [name for name in expected.get("must_not_call", []) if name in names]
    arg_errors: list[str] = []
    for tool_name, required_args in expected.get("arguments", {}).items():
        matching = [call for call in calls if call["name"] == tool_name]
        if not matching:
            continue
        matched = False
        for call in matching:
            args = call.get("arguments", {})
            result = call.get("result", {})
            checks = []
            for key, value in required_args.items():
                if key in {"porter_id", "robot_id", "elevator_id"}:
                    checks.append(args.get(key) == value or result.get(key) == value)
                elif key == "status":
                    checks.append(args.get(key) == value or result.get(key) == value)
                else:
                    checks.append(args.get(key) == value or result.get(key) == value)
            if all(checks) and result.get("ok", True):
                matched = True
                break
        if not matched:
            arg_errors.append(f"{tool_name} missing expected args {required_args}")
    required_count = len(expected.get("must_call", [])) + len(expected.get("arguments", {}))
    satisfied_count = max(0, required_count - len(missing) - len(arg_errors))
    penalty = len(forbidden)
    score = max(0.0, satisfied_count / required_count if required_count else 1.0)
    if penalty:
        score = max(0.0, score - 0.25 * penalty)
    passed = not missing and not forbidden and not arg_errors
    return {
        "passed": passed,
        "missing": missing,
        "forbidden": forbidden,
        "arg_errors": arg_errors,
        "tool_calls": names,
        "score": 1.0 if passed else score,
    }
