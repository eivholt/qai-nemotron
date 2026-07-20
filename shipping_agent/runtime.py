"""Deterministic mock shipping tools used by the console agent."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable


SCENARIOS: dict[str, dict[str, Any]] = {
    "routine": {
        "title": "Routine pallet shipment",
        "task": "Plan the pending shipment and notify dispatch when the plan is recorded.",
        "shipment": {
            "shipment_id": "SHP-1001",
            "destination": "Bergen terminal",
            "pallets": 4,
            "handling": "standard",
            "deadline_minutes": 90,
            "status": "pending",
        },
        "carriers": [
            {
                "carrier_id": "VAN-3",
                "capacity_pallets": 2,
                "handling": ["standard"],
                "travel_minutes": 20,
                "route_open": True,
            },
            {
                "carrier_id": "TRUCK-7",
                "capacity_pallets": 8,
                "handling": ["standard"],
                "travel_minutes": 35,
                "route_open": True,
            },
        ],
        "docks": [
            {"dock_id": "D1", "available": True, "handling": ["standard"]},
            {"dock_id": "D2", "available": False, "handling": ["standard"]},
        ],
        "expected": {"action": "scheduled", "carrier_id": "TRUCK-7", "dock_id": "D1"},
    },
    "cold_chain": {
        "title": "Refrigerated medicine shipment",
        "task": "Plan the pending shipment without breaking its cold-chain requirement.",
        "shipment": {
            "shipment_id": "SHP-2001",
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
        "expected": {"action": "scheduled", "carrier_id": "REEFER-2", "dock_id": "COLD-1"},
    },
    "weather_hold": {
        "title": "Route temporarily closed by weather",
        "task": "Choose a safe disposition for the pending shipment and notify dispatch.",
        "shipment": {
            "shipment_id": "SHP-3001",
            "destination": "Mountain depot",
            "pallets": 5,
            "handling": "standard",
            "deadline_minutes": 180,
            "status": "pending",
        },
        "carriers": [
            {
                "carrier_id": "TRUCK-9",
                "capacity_pallets": 10,
                "handling": ["standard"],
                "travel_minutes": 55,
                "route_open": False,
                "route_reopens_minutes": 40,
            },
            {
                "carrier_id": "VAN-3",
                "capacity_pallets": 2,
                "handling": ["standard"],
                "travel_minutes": 35,
                "route_open": True,
            },
        ],
        "docks": [
            {"dock_id": "D1", "available": True, "handling": ["standard"]},
        ],
        "expected": {"action": "held"},
    },
    "no_compliant_carrier": {
        "title": "Hazardous material without a certified carrier",
        "task": "Choose a safe disposition for the pending shipment and notify dispatch.",
        "shipment": {
            "shipment_id": "SHP-4001",
            "destination": "Industrial customer",
            "pallets": 2,
            "handling": "hazmat",
            "deadline_minutes": 120,
            "status": "pending",
        },
        "carriers": [
            {
                "carrier_id": "TRUCK-7",
                "capacity_pallets": 8,
                "handling": ["standard"],
                "travel_minutes": 45,
                "route_open": True,
            },
            {
                "carrier_id": "REEFER-2",
                "capacity_pallets": 6,
                "handling": ["standard", "cold_chain"],
                "travel_minutes": 50,
                "route_open": True,
            },
        ],
        "docks": [
            {"dock_id": "HAZ-1", "available": True, "handling": ["hazmat"]},
        ],
        "expected": {"action": "escalated"},
    },
}


def scenario(name: str) -> dict[str, Any]:
    if name not in SCENARIOS:
        choices = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario {name!r}. Choose one of: {choices}")
    return copy.deepcopy(SCENARIOS[name])


@dataclass
class ShippingRuntime:
    """Mutable mock world with validation at every action boundary."""

    state: dict[str, Any]
    verbose: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)
    shipment_read: bool = False
    options_read: bool = False
    carrier_options_read: bool = False
    dock_options_read: bool = False
    notified: bool = False

    @classmethod
    def from_scenario(cls, name: str, *, verbose: bool = True) -> "ShippingRuntime":
        return cls(scenario(name), verbose=verbose)

    @property
    def shipment(self) -> dict[str, Any]:
        return self.state["shipment"]

    def _emit(self, label: str, value: Any) -> None:
        if self.verbose:
            print(f"{label:<12} {json.dumps(value, sort_keys=True)}", flush=True)

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        self._emit("TOOL CALL", {"name": name, "arguments": arguments})
        result = operation()
        self.calls.append({"name": name, "arguments": arguments, "result": copy.deepcopy(result)})
        self._emit("TOOL RESULT", result)
        return result

    def get_pending_shipment(self) -> dict[str, Any]:
        self.shipment_read = True
        return {"ok": True, "shipment": copy.deepcopy(self.shipment)}

    def get_shipping_options(self, shipment_id: str) -> dict[str, Any]:
        if shipment_id != self.shipment["shipment_id"]:
            return {"ok": False, "error": "unknown_shipment", "shipment_id": shipment_id}
        self.options_read = True
        carrier_options = self.get_carrier_options()
        dock_options = self.get_dock_options()
        return {
            "ok": True,
            "shipment_id": shipment_id,
            "usable_carriers": carrier_options["usable_carriers"],
            "temporarily_blocked_carriers": carrier_options[
                "temporarily_blocked_carriers"
            ],
            "excluded_carriers": carrier_options["excluded_carriers"],
            "usable_docks": dock_options["usable_docks"],
            "excluded_docks": dock_options["excluded_docks"],
        }

    def get_carrier_options(self) -> dict[str, Any]:
        """Assess carriers for the current shipment without changing world state."""
        self.carrier_options_read = True
        carriers = [
            self._assess_carrier(carrier) for carrier in self.state["carriers"]
        ]
        return {
            "ok": True,
            "shipment_id": self.shipment["shipment_id"],
            "usable_carriers": [
                item for item in carriers if item["assessment"] == "usable"
            ],
            "temporarily_blocked_carriers": [
                item
                for item in carriers
                if item["assessment"] == "temporarily_blocked"
            ],
            "excluded_carriers": [
                item for item in carriers if item["assessment"] == "incompatible"
            ],
        }

    def get_dock_options(self) -> dict[str, Any]:
        """Assess docks for the current shipment without changing world state."""
        self.dock_options_read = True
        docks = [self._assess_dock(dock) for dock in self.state["docks"]]
        return {
            "ok": True,
            "shipment_id": self.shipment["shipment_id"],
            "usable_docks": [
                item for item in docks if item["assessment"] == "usable"
            ],
            "excluded_docks": [
                item for item in docks if item["assessment"] == "incompatible"
            ],
        }

    def _assess_carrier(self, carrier: dict[str, Any]) -> dict[str, Any]:
        item = copy.deepcopy(carrier)
        issues: list[str] = []
        if carrier["capacity_pallets"] < self.shipment["pallets"]:
            issues.append("insufficient_capacity")
        if self.shipment["handling"] not in carrier["handling"]:
            issues.append("handling_not_supported")
        if carrier["travel_minutes"] > self.shipment["deadline_minutes"]:
            issues.append("deadline_missed")

        if not carrier["route_open"]:
            reopens = carrier.get("route_reopens_minutes")
            recoverable = (
                not issues
                and reopens is not None
                and reopens + carrier["travel_minutes"]
                <= self.shipment["deadline_minutes"]
            )
            issues.append("route_temporarily_closed" if recoverable else "route_closed")
            assessment = "temporarily_blocked" if recoverable else "incompatible"
        else:
            assessment = "incompatible" if issues else "usable"

        item.update({"assessment": assessment, "issues": issues})
        return item

    def _assess_dock(self, dock: dict[str, Any]) -> dict[str, Any]:
        item = copy.deepcopy(dock)
        issues: list[str] = []
        if not dock["available"]:
            issues.append("dock_unavailable")
        if self.shipment["handling"] not in dock["handling"]:
            issues.append("handling_not_supported")
        item.update(
            {
                "assessment": "usable" if not issues else "incompatible",
                "issues": issues,
            }
        )
        return item

    def schedule_shipment(self, shipment_id: str, carrier_id: str, dock_id: str) -> dict[str, Any]:
        error = self._planning_error(shipment_id)
        if error:
            return error
        carrier = next(
            (item for item in self.state["carriers"] if item["carrier_id"] == carrier_id),
            None,
        )
        dock = next((item for item in self.state["docks"] if item["dock_id"] == dock_id), None)
        if carrier is None:
            return {"ok": False, "error": "unknown_carrier", "carrier_id": carrier_id}
        if dock is None:
            return {"ok": False, "error": "unknown_dock", "dock_id": dock_id}
        if carrier["capacity_pallets"] < self.shipment["pallets"]:
            return {"ok": False, "error": "insufficient_capacity", "carrier_id": carrier_id}
        if self.shipment["handling"] not in carrier["handling"]:
            return {"ok": False, "error": "carrier_not_compliant", "carrier_id": carrier_id}
        if not carrier["route_open"]:
            return {
                "ok": False,
                "error": "route_closed",
                "carrier_id": carrier_id,
                "route_reopens_minutes": carrier.get("route_reopens_minutes"),
            }
        if carrier["travel_minutes"] > self.shipment["deadline_minutes"]:
            return {"ok": False, "error": "deadline_missed", "carrier_id": carrier_id}
        if not dock["available"]:
            return {"ok": False, "error": "dock_unavailable", "dock_id": dock_id}
        if self.shipment["handling"] not in dock["handling"]:
            return {"ok": False, "error": "dock_not_compliant", "dock_id": dock_id}
        self.shipment.update(
            {
                "status": "scheduled",
                "carrier_id": carrier_id,
                "dock_id": dock_id,
            }
        )
        dock["available"] = False
        return {
            "ok": True,
            "status": "scheduled",
            "shipment_id": shipment_id,
            "carrier_id": carrier_id,
            "dock_id": dock_id,
            "dispatch_notified": False,
        }

    def hold_shipment(self, shipment_id: str, reason: str) -> dict[str, Any]:
        error = self._planning_error(shipment_id)
        if error:
            return error
        if not reason.strip():
            return {"ok": False, "error": "reason_required"}
        self.shipment.update({"status": "held", "reason": reason.strip()})
        return {
            "ok": True,
            "status": "held",
            "shipment_id": shipment_id,
            "reason": reason.strip(),
            "dispatch_notified": False,
        }

    def escalate_shipment(self, shipment_id: str, reason: str) -> dict[str, Any]:
        error = self._planning_error(shipment_id)
        if error:
            return error
        if not reason.strip():
            return {"ok": False, "error": "reason_required"}
        self.shipment.update({"status": "escalated", "reason": reason.strip()})
        return {
            "ok": True,
            "status": "escalated",
            "shipment_id": shipment_id,
            "reason": reason.strip(),
            "dispatch_notified": False,
        }

    def notify_dispatch(self, shipment_id: str) -> dict[str, Any]:
        if shipment_id != self.shipment["shipment_id"]:
            return {"ok": False, "error": "unknown_shipment", "shipment_id": shipment_id}
        if self.shipment["status"] == "pending":
            return {"ok": False, "error": "plan_shipment_before_notification"}
        if self.notified:
            return {"ok": False, "error": "dispatch_already_notified"}
        self.notified = True
        return {
            "ok": True,
            "shipment_id": shipment_id,
            "notified": True,
            "dispatch_notified": True,
            "message": f"Shipment {shipment_id} is {self.shipment['status']}.",
        }

    def _planning_error(self, shipment_id: str) -> dict[str, Any] | None:
        if shipment_id != self.shipment["shipment_id"]:
            return {"ok": False, "error": "unknown_shipment", "shipment_id": shipment_id}
        if self.shipment["status"] != "pending":
            return {
                "ok": False,
                "error": "shipment_already_planned",
                "status": self.shipment["status"],
            }
        return None


    def summary(self) -> dict[str, Any]:
        expected = self.state["expected"]
        actual = {"action": self.shipment["status"]}
        if self.shipment["status"] == "scheduled":
            actual.update(
                {
                    "carrier_id": self.shipment.get("carrier_id"),
                    "dock_id": self.shipment.get("dock_id"),
                }
            )

        action_tool = {
            "scheduled": "schedule_shipment",
            "held": "hold_shipment",
            "escalated": "escalate_shipment",
        }[expected["action"]]
        expected_calls = [
            "get_pending_shipment",
            "get_shipping_options",
            action_tool,
            "notify_dispatch",
        ]
        call_names = [call["name"] for call in self.calls]
        tool_errors = [
            {"name": call["name"], "error": call["result"].get("error", "unknown")}
            for call in self.calls
            if not call["result"].get("ok", False)
        ]
        state_correct = all(actual.get(key) == value for key, value in expected.items())
        trace_correct = call_names == expected_calls
        passed = self.notified and state_correct and trace_correct and not tool_errors
        return {
            "passed": passed,
            "expected": copy.deepcopy(expected),
            "actual": actual,
            "dispatch_notified": self.notified,
            "tool_calls": len(self.calls),
            "call_names": call_names,
            "expected_calls": expected_calls,
            "state_correct": state_correct,
            "trace_correct": trace_correct,
            "tool_errors": tool_errors,
        }


def run_scripted(name: str, *, verbose: bool = True) -> ShippingRuntime:
    """Exercise every mock tool without an LLM."""

    runtime = ShippingRuntime.from_scenario(name, verbose=verbose)
    shipment_id = runtime.shipment["shipment_id"]
    runtime.call("get_pending_shipment", {}, runtime.get_pending_shipment)
    runtime.call(
        "get_shipping_options",
        {},
        lambda: runtime.get_shipping_options(shipment_id),
    )
    expected = runtime.state["expected"]
    if expected["action"] == "scheduled":
        runtime.call(
            "schedule_shipment",
            {
                "carrier_id": expected["carrier_id"],
                "dock_id": expected["dock_id"],
            },
            lambda: runtime.schedule_shipment(
                shipment_id, expected["carrier_id"], expected["dock_id"]
            ),
        )
    elif expected["action"] == "held":
        runtime.call(
            "hold_shipment",
            {"reason": "route temporarily closed"},
            lambda: runtime.hold_shipment(shipment_id, "route temporarily closed"),
        )
    else:
        runtime.call(
            "escalate_shipment",
            {"reason": "no compliant carrier"},
            lambda: runtime.escalate_shipment(shipment_id, "no compliant carrier"),
        )
    runtime.call(
        "notify_dispatch",
        {},
        lambda: runtime.notify_dispatch(shipment_id),
    )
    return runtime
