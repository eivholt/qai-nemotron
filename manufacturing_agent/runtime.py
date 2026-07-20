"""Deterministic manufacturing scenarios and strict action ledger."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable


SCENARIOS: dict[str, dict[str, Any]] = {
    "normal": {
        "title": "Healthy production batch",
        "context": {
            "batch_id": "BATCH-100",
            "product": "pump_housing",
            "machine_ids": ["CELL-1", "CELL-2"],
            "status": "analyzing",
        },
        "policy": {
            "max_temperature_c": 80.0,
            "max_vibration_rms": 5.0,
            "max_defect_rate": 0.05,
        },
        "quality": {"inspected": 100, "defects": 2},
        "sensors": {
            "CELL-1": {
                "temperature_c": [62.0, 64.0, 63.0, 65.0],
                "vibration_mm_s": [1.1, 1.4, 1.2, 1.3],
            },
            "CELL-2": {
                "temperature_c": [66.0, 67.0, 65.0, 68.0],
                "vibration_mm_s": [1.8, 1.7, 1.9, 1.6],
            },
        },
        "expected": {"action": "released", "inspection_machine_id": None},
    },
    "bearing_vibration": {
        "title": "Bearing vibration anomaly",
        "context": {
            "batch_id": "BATCH-201",
            "product": "pump_housing",
            "machine_ids": ["CELL-1", "CELL-2"],
            "status": "analyzing",
        },
        "policy": {
            "max_temperature_c": 80.0,
            "max_vibration_rms": 5.0,
            "max_defect_rate": 0.05,
        },
        "quality": {"inspected": 100, "defects": 2},
        "sensors": {
            "CELL-1": {
                "temperature_c": [63.0, 64.0, 65.0, 64.0],
                "vibration_mm_s": [1.2, 1.4, 1.3, 1.5],
            },
            "CELL-2": {
                "temperature_c": [68.0, 69.0, 70.0, 69.0],
                "vibration_mm_s": [6.2, 7.0, 7.8, 7.1],
            },
        },
        "expected": {"action": "held", "inspection_machine_id": "CELL-2"},
    },
    "thermal_drift": {
        "title": "Thermal process drift",
        "context": {
            "batch_id": "BATCH-302",
            "product": "valve_body",
            "machine_ids": ["CELL-1", "CELL-2"],
            "status": "analyzing",
        },
        "policy": {
            "max_temperature_c": 80.0,
            "max_vibration_rms": 5.0,
            "max_defect_rate": 0.05,
        },
        "quality": {"inspected": 80, "defects": 1},
        "sensors": {
            "CELL-1": {
                "temperature_c": [77.0, 81.0, 83.0, 84.0],
                "vibration_mm_s": [1.5, 1.7, 1.6, 1.8],
            },
            "CELL-2": {
                "temperature_c": [65.0, 66.0, 67.0, 66.0],
                "vibration_mm_s": [1.6, 1.5, 1.7, 1.6],
            },
        },
        "expected": {"action": "held", "inspection_machine_id": "CELL-1"},
    },
    "quality_spike": {
        "title": "Defect-rate spike",
        "context": {
            "batch_id": "BATCH-403",
            "product": "valve_body",
            "machine_ids": ["CELL-1", "CELL-2"],
            "status": "analyzing",
        },
        "policy": {
            "max_temperature_c": 80.0,
            "max_vibration_rms": 5.0,
            "max_defect_rate": 0.05,
        },
        "quality": {"inspected": 100, "defects": 9},
        "sensors": {
            "CELL-1": {
                "temperature_c": [64.0, 65.0, 66.0, 65.0],
                "vibration_mm_s": [1.4, 1.5, 1.3, 1.4],
            },
            "CELL-2": {
                "temperature_c": [67.0, 68.0, 69.0, 68.0],
                "vibration_mm_s": [1.7, 1.8, 1.6, 1.7],
            },
        },
        "expected": {"action": "quarantined", "inspection_machine_id": None},
    },
}


def scenario(name: str) -> dict[str, Any]:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown manufacturing scenario: {name}")
    return copy.deepcopy(SCENARIOS[name])


@dataclass
class ManufacturingRuntime:
    state: dict[str, Any]
    verbose: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    inspections: list[dict[str, str]] = field(default_factory=list)
    notified: bool = False

    @classmethod
    def from_scenario(
        cls,
        name: str,
        *,
        verbose: bool = False,
    ) -> "ManufacturingRuntime":
        return cls(scenario(name), verbose=verbose)

    @property
    def context(self) -> dict[str, Any]:
        return self.state["context"]

    def _emit(self, label: str, value: Any) -> None:
        if self.verbose:
            print(f"{label:<12} {json.dumps(value, sort_keys=True)}", flush=True)

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        self._emit("API CALL", {"name": name, "arguments": arguments})
        result = operation()
        self.calls.append(
            {
                "name": name,
                "arguments": copy.deepcopy(arguments),
                "result": copy.deepcopy(result),
            }
        )
        self._emit("API RESULT", result)
        return result

    def get_production_context(self) -> dict[str, Any]:
        return {"ok": True, "context": copy.deepcopy(self.context)}

    def get_policy(self) -> dict[str, Any]:
        return {"ok": True, "policy": copy.deepcopy(self.state["policy"])}

    def get_quality_counts(self) -> dict[str, Any]:
        return {
            "ok": True,
            "batch_id": self.context["batch_id"],
            **copy.deepcopy(self.state["quality"]),
        }

    def get_sensor_summary(self, machine_id: str) -> dict[str, Any]:
        sensors = self.state["sensors"].get(machine_id)
        if sensors is None:
            return {"ok": False, "error": "unknown_machine", "machine_id": machine_id}
        temperatures = sensors["temperature_c"]
        vibrations = sensors["vibration_mm_s"]
        return {
            "ok": True,
            "machine_id": machine_id,
            "max_temperature_c": max(temperatures),
            "vibration_rms": math.sqrt(
                sum(value * value for value in vibrations) / len(vibrations)
            ),
        }

    def schedule_inspection(self, machine_id: str, reason: str) -> dict[str, Any]:
        if machine_id not in self.context["machine_ids"]:
            return {"ok": False, "error": "unknown_machine", "machine_id": machine_id}
        if not reason.strip():
            return {"ok": False, "error": "reason_required"}
        if self.inspections:
            return {"ok": False, "error": "inspection_already_scheduled"}
        record = {"machine_id": machine_id, "reason": reason.strip()}
        self.inspections.append(record)
        return {"ok": True, "inspection": copy.deepcopy(record)}

    def release_batch(self, batch_id: str, evidence: str) -> dict[str, Any]:
        return self._set_disposition(batch_id, "released", evidence)

    def hold_batch(self, batch_id: str, reason: str) -> dict[str, Any]:
        if not self.inspections:
            return {"ok": False, "error": "inspection_required_before_hold"}
        return self._set_disposition(batch_id, "held", reason)

    def quarantine_batch(self, batch_id: str, reason: str) -> dict[str, Any]:
        return self._set_disposition(batch_id, "quarantined", reason)

    def _set_disposition(
        self,
        batch_id: str,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        if batch_id != self.context["batch_id"]:
            return {"ok": False, "error": "unknown_batch", "batch_id": batch_id}
        if self.context["status"] != "analyzing":
            return {
                "ok": False,
                "error": "batch_already_disposed",
                "status": self.context["status"],
            }
        if not reason.strip():
            return {"ok": False, "error": "reason_required"}
        self.context.update({"status": status, "reason": reason.strip()})
        return {
            "ok": True,
            "batch_id": batch_id,
            "status": status,
            "supervisor_notified": False,
        }

    def notify_supervisor(self, message: str) -> dict[str, Any]:
        if self.context["status"] == "analyzing":
            return {"ok": False, "error": "disposition_required_before_notification"}
        if self.notified:
            return {"ok": False, "error": "supervisor_already_notified"}
        if not message.strip():
            return {"ok": False, "error": "message_required"}
        self.notified = True
        return {
            "ok": True,
            "notified": True,
            "batch_id": self.context["batch_id"],
            "status": self.context["status"],
        }

    def summary(self) -> dict[str, Any]:
        expected = self.state["expected"]
        inspection_machine_id = (
            self.inspections[0]["machine_id"] if self.inspections else None
        )
        actual = {
            "action": self.context["status"],
            "inspection_machine_id": inspection_machine_id,
        }
        read_names = {
            "get_production_context",
            "get_policy",
            "get_quality_counts",
            "get_sensor_summary",
        }
        action = expected["action"]
        expected_calls = [
            "get_production_context",
            "get_policy",
            "get_quality_counts",
            "get_sensor_summary",
            "get_sensor_summary",
        ]
        if expected["inspection_machine_id"]:
            expected_calls.extend(["schedule_inspection", "hold_batch"])
        elif action == "quarantined":
            expected_calls.append("quarantine_batch")
        else:
            expected_calls.append("release_batch")
        expected_calls.append("notify_supervisor")

        call_names = [item["name"] for item in self.calls]
        tool_errors = [
            {"name": item["name"], "error": item["result"].get("error", "unknown")}
            for item in self.calls
            if not item["result"].get("ok", False)
        ]
        exact_calls = sorted(call_names) == sorted(expected_calls)
        ordered = False
        if exact_calls:
            last_read = max(
                index for index, name in enumerate(call_names) if name in read_names
            )
            first_write = min(
                index for index, name in enumerate(call_names) if name not in read_names
            )
            ordered = (
                last_read < first_write
                and call_names[-1] == "notify_supervisor"
            )
            if expected["inspection_machine_id"]:
                ordered = ordered and (
                    call_names.index("schedule_inspection")
                    < call_names.index("hold_batch")
                    < call_names.index("notify_supervisor")
                )

        state_correct = all(actual.get(key) == value for key, value in expected.items())
        passed = (
            self.notified
            and state_correct
            and exact_calls
            and ordered
            and not tool_errors
        )
        return {
            "passed": passed,
            "expected": copy.deepcopy(expected),
            "actual": actual,
            "notified": self.notified,
            "call_names": call_names,
            "expected_calls": expected_calls,
            "trace_correct": exact_calls and ordered,
            "state_correct": state_correct,
            "tool_errors": tool_errors,
            "inspections": copy.deepcopy(self.inspections),
        }
