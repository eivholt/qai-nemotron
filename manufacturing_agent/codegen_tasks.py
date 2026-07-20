"""Task specifications and strict mock runtimes for manufacturing code generation."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ApiFunction:
    name: str
    parameters: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    title: str
    contract_version: str
    instructions: str
    functions: tuple[ApiFunction, ...]
    scenarios: dict[str, dict[str, Any]]

    @property
    def seed_scenario(self) -> str:
        return next(iter(self.scenarios))


BATCH_FUNCTIONS = (
    ApiFunction("get_production_context", (), "-> {batch_id: str, machine_ids: list[str]}"),
    ApiFunction(
        "get_policy",
        (),
        "-> {max_temperature_c: float, max_vibration_rms: float, max_defect_rate: float}",
    ),
    ApiFunction("get_quality_counts", (), "-> {inspected: int, defects: int}"),
    ApiFunction(
        "get_sensor_summary",
        ("machine_id",),
        "-> {machine_id: str, max_temperature_c: float, vibration_rms: float}",
    ),
    ApiFunction("schedule_inspection", ("machine_id", "reason"), "schedule one inspection"),
    ApiFunction("release_batch", ("batch_id", "evidence"), "release the active batch"),
    ApiFunction("hold_batch", ("batch_id", "reason"), "hold after scheduling inspection"),
    ApiFunction("quarantine_batch", ("batch_id", "reason"), "quarantine the active batch"),
    ApiFunction("notify_supervisor", ("message",), "send one final notification"),
)

MAINTENANCE_FUNCTIONS = (
    ApiFunction(
        "get_maintenance_policy",
        (),
        "-> {service_risk: float, critical_risk: float}",
    ),
    ApiFunction("get_machine_queue", (), "-> {machine_ids: list[str]} in priority tie order; iterate result['machine_ids']"),
    ApiFunction(
        "get_machine_health",
        ("machine_id",),
        "-> {machine_id: str, risk_score: float, production_critical: bool}",
    ),
    ApiFunction(
        "schedule_maintenance",
        ("machine_id", "priority", "reason"),
        "schedule one machine with priority planned or urgent",
    ),
    ApiFunction("record_monitoring", ("reason",), "record that no service is required"),
    ApiFunction("notify_maintenance", ("message",), "send one final notification"),
)

QUALITY_FUNCTIONS = (
    ApiFunction(
        "get_inspection_policy",
        (),
        "-> {max_total_defect_rate: float, max_station_rework_rate: float}",
    ),
    ApiFunction("get_active_lot", (), "-> {lot_id: str, station_ids: list[str]}; iterate result['station_ids']"),
    ApiFunction(
        "get_station_quality",
        ("station_id",),
        "-> {station_id: str, inspected: int, defects: int, rework: int}",
    ),
    ApiFunction("quarantine_lot", ("lot_id", "reason"), "quarantine the active lot"),
    ApiFunction(
        "increase_sampling",
        ("lot_id", "station_id", "reason"),
        "increase sampling for one station",
    ),
    ApiFunction("release_lot", ("lot_id", "evidence"), "release the active lot"),
    ApiFunction("notify_quality", ("message",), "send one final notification"),
)

ENERGY_FUNCTIONS = (
    ApiFunction(
        "get_energy_policy",
        (),
        "-> {max_price_per_kwh: float, max_projected_load_kw: float}",
    ),
    ApiFunction(
        "get_pending_energy_job",
        (),
        "-> {job_id: str, duration_slots: int, deadline_slot: int}",
    ),
    ApiFunction(
        "get_candidate_windows",
        (),
        "-> {windows: list[dict]}; iterate result['windows']; each window has "
        "window_id, start_slot, price_per_kwh, and projected_load_kw",
    ),
    ApiFunction("schedule_energy_job", ("job_id", "window_id"), "schedule one feasible window"),
    ApiFunction("defer_energy_job", ("job_id", "reason"), "defer when no window is feasible"),
    ApiFunction("notify_energy_desk", ("message",), "send one final notification"),
)

INVENTORY_FUNCTIONS = (
    ApiFunction(
        "get_required_part",
        (),
        "-> {part_id: str, on_hand: int, required_quantity: int, needed_in_days: int}",
    ),
    ApiFunction(
        "get_supplier_options",
        (),
        "-> {suppliers: list[dict]}; iterate result['suppliers']; each supplier has "
        "supplier_id, available_quantity, and lead_days",
    ),
    ApiFunction("record_inventory_ok", ("part_id", "evidence"), "record that stock is sufficient"),
    ApiFunction(
        "create_purchase_order",
        ("part_id", "supplier_id", "quantity"),
        "order the exact shortage from the first feasible supplier",
    ),
    ApiFunction("escalate_shortage", ("part_id", "reason"), "escalate when no supplier is feasible"),
    ApiFunction("notify_inventory", ("message",), "send one final notification"),
)


def _batch_scenarios() -> dict[str, dict[str, Any]]:
    base = {
        "policy": {
            "max_temperature_c": 80.0,
            "max_vibration_rms": 5.0,
            "max_defect_rate": 0.05,
        },
        "context": {"batch_id": "BATCH-100", "machine_ids": ["CELL-1", "CELL-2"]},
        "quality": {"inspected": 100, "defects": 2},
        "summaries": {
            "CELL-1": {"max_temperature_c": 65.0, "vibration_rms": 1.3},
            "CELL-2": {"max_temperature_c": 68.0, "vibration_rms": 1.8},
        },
    }
    normal = copy.deepcopy(base)
    normal.update(
        expected={"action": "released", "target": "BATCH-100"},
        expected_calls=[
            "get_production_context",
            "get_policy",
            "get_quality_counts",
            "get_sensor_summary",
            "get_sensor_summary",
            "release_batch",
            "notify_supervisor",
        ],
    )
    vibration = copy.deepcopy(base)
    vibration["context"]["batch_id"] = "BATCH-201"
    vibration["summaries"]["CELL-2"]["vibration_rms"] = 7.1
    vibration.update(
        expected={"action": "held", "target": "CELL-2"},
        expected_calls=[
            "get_production_context",
            "get_policy",
            "get_quality_counts",
            "get_sensor_summary",
            "get_sensor_summary",
            "schedule_inspection",
            "hold_batch",
            "notify_supervisor",
        ],
    )
    quality = copy.deepcopy(base)
    quality["context"]["batch_id"] = "BATCH-403"
    quality["quality"]["defects"] = 9
    quality.update(
        expected={"action": "quarantined", "target": "BATCH-403"},
        expected_calls=[
            "get_production_context",
            "get_policy",
            "get_quality_counts",
            "get_sensor_summary",
            "get_sensor_summary",
            "quarantine_batch",
            "notify_supervisor",
        ],
    )
    return {"normal": normal, "vibration": vibration, "quality_spike": quality}


def _maintenance_scenarios() -> dict[str, dict[str, Any]]:
    base = {
        "policy": {"service_risk": 0.55, "critical_risk": 0.85},
        "queue": {"machine_ids": ["M-10", "M-20", "M-30"]},
        "health": {
            "M-10": {"risk_score": 0.20, "production_critical": False},
            "M-20": {"risk_score": 0.32, "production_critical": True},
            "M-30": {"risk_score": 0.25, "production_critical": False},
        },
    }
    monitor = copy.deepcopy(base)
    monitor.update(
        expected={"action": "monitoring", "target": None, "priority": None},
        expected_calls=[
            "get_maintenance_policy",
            "get_machine_queue",
            "get_machine_health",
            "get_machine_health",
            "get_machine_health",
            "record_monitoring",
            "notify_maintenance",
        ],
    )
    planned = copy.deepcopy(base)
    planned["health"]["M-30"]["risk_score"] = 0.70
    planned.update(
        expected={"action": "maintenance", "target": "M-30", "priority": "planned"},
        expected_calls=[
            "get_maintenance_policy",
            "get_machine_queue",
            "get_machine_health",
            "get_machine_health",
            "get_machine_health",
            "schedule_maintenance",
            "notify_maintenance",
        ],
    )
    urgent = copy.deepcopy(base)
    urgent["health"]["M-20"]["risk_score"] = 0.90
    urgent.update(
        expected={"action": "maintenance", "target": "M-20", "priority": "urgent"},
        expected_calls=[
            "get_maintenance_policy",
            "get_machine_queue",
            "get_machine_health",
            "get_machine_health",
            "get_machine_health",
            "schedule_maintenance",
            "notify_maintenance",
        ],
    )
    return {"monitor": monitor, "planned": planned, "urgent": urgent}


def _quality_scenarios() -> dict[str, dict[str, Any]]:
    base = {
        "policy": {
            "max_total_defect_rate": 0.04,
            "max_station_rework_rate": 0.10,
        },
        "lot": {"lot_id": "LOT-70", "station_ids": ["S-1", "S-2"]},
        "stations": {
            "S-1": {"inspected": 100, "defects": 1, "rework": 3},
            "S-2": {"inspected": 100, "defects": 2, "rework": 4},
        },
    }
    release = copy.deepcopy(base)
    release.update(
        expected={"action": "released", "target": "LOT-70"},
        expected_calls=[
            "get_inspection_policy",
            "get_active_lot",
            "get_station_quality",
            "get_station_quality",
            "release_lot",
            "notify_quality",
        ],
    )
    sample = copy.deepcopy(base)
    sample["stations"]["S-2"]["rework"] = 18
    sample.update(
        expected={"action": "sampling", "target": "S-2"},
        expected_calls=[
            "get_inspection_policy",
            "get_active_lot",
            "get_station_quality",
            "get_station_quality",
            "increase_sampling",
            "notify_quality",
        ],
    )
    quarantine = copy.deepcopy(base)
    quarantine["stations"]["S-1"]["defects"] = 8
    quarantine["stations"]["S-2"]["defects"] = 6
    quarantine.update(
        expected={"action": "quarantined", "target": "LOT-70"},
        expected_calls=[
            "get_inspection_policy",
            "get_active_lot",
            "get_station_quality",
            "get_station_quality",
            "quarantine_lot",
            "notify_quality",
        ],
    )
    return {"release": release, "sampling": sample, "quarantine": quarantine}


def _energy_scenarios() -> dict[str, dict[str, Any]]:
    base = {
        "policy": {"max_price_per_kwh": 0.18, "max_projected_load_kw": 420},
        "job": {"job_id": "JOB-8", "duration_slots": 2, "deadline_slot": 8},
        "windows": [
            {
                "window_id": "W-1",
                "start_slot": 2,
                "price_per_kwh": 0.14,
                "projected_load_kw": 390,
            },
            {
                "window_id": "W-2",
                "start_slot": 4,
                "price_per_kwh": 0.16,
                "projected_load_kw": 400,
            },
            {
                "window_id": "W-3",
                "start_slot": 6,
                "price_per_kwh": 0.12,
                "projected_load_kw": 410,
            },
        ],
    }
    first = copy.deepcopy(base)
    first.update(
        expected={"action": "scheduled", "target": "W-1"},
        expected_calls=[
            "get_energy_policy",
            "get_pending_energy_job",
            "get_candidate_windows",
            "schedule_energy_job",
            "notify_energy_desk",
        ],
    )
    skip = copy.deepcopy(base)
    skip["windows"][0]["projected_load_kw"] = 500
    skip.update(
        expected={"action": "scheduled", "target": "W-2"},
        expected_calls=[
            "get_energy_policy",
            "get_pending_energy_job",
            "get_candidate_windows",
            "schedule_energy_job",
            "notify_energy_desk",
        ],
    )
    defer = copy.deepcopy(base)
    for window in defer["windows"]:
        window["price_per_kwh"] = 0.25
    defer.update(
        expected={"action": "deferred", "target": "JOB-8"},
        expected_calls=[
            "get_energy_policy",
            "get_pending_energy_job",
            "get_candidate_windows",
            "defer_energy_job",
            "notify_energy_desk",
        ],
    )
    return {"first_window": first, "skip_window": skip, "defer": defer}


def _inventory_scenarios() -> dict[str, dict[str, Any]]:
    base = {
        "part": {
            "part_id": "BEARING-42",
            "on_hand": 12,
            "required_quantity": 10,
            "needed_in_days": 6,
        },
        "suppliers": [
            {"supplier_id": "SUP-A", "available_quantity": 20, "lead_days": 4},
            {"supplier_id": "SUP-B", "available_quantity": 30, "lead_days": 5},
        ],
    }
    sufficient = copy.deepcopy(base)
    sufficient.update(
        expected={"action": "inventory_ok", "target": "BEARING-42", "quantity": 0},
        expected_calls=[
            "get_required_part",
            "get_supplier_options",
            "record_inventory_ok",
            "notify_inventory",
        ],
    )
    order = copy.deepcopy(base)
    order["part"]["on_hand"] = 3
    order.update(
        expected={"action": "ordered", "target": "SUP-A", "quantity": 7},
        expected_calls=[
            "get_required_part",
            "get_supplier_options",
            "create_purchase_order",
            "notify_inventory",
        ],
    )
    escalate = copy.deepcopy(base)
    escalate["part"]["on_hand"] = 2
    for supplier in escalate["suppliers"]:
        supplier["lead_days"] = 10
    escalate.update(
        expected={"action": "escalated", "target": "BEARING-42", "quantity": 8},
        expected_calls=[
            "get_required_part",
            "get_supplier_options",
            "escalate_shortage",
            "notify_inventory",
        ],
    )
    return {"sufficient": sufficient, "order": order, "escalate": escalate}


TASKS: dict[str, TaskSpec] = {
    "batch_disposition": TaskSpec(
        name="batch_disposition",
        title="Batch disposition from quality and machine summaries",
        contract_version="batch-v1",
        instructions="""Read context, policy, quality, and every machine summary before acting.
Compute defect_rate = defects / inspected. If it exceeds max_defect_rate,
quarantine. Otherwise inspect the first machine whose temperature or vibration
exceeds its matching limit, then hold. If neither condition applies, release.
Choose exactly one disposition and notify the supervisor exactly once afterward.""",
        functions=BATCH_FUNCTIONS,
        scenarios=_batch_scenarios(),
    ),
    "maintenance_priority": TaskSpec(
        name="maintenance_priority",
        title="Maintenance priority from machine risk",
        contract_version="maintenance-v1",
        instructions="""Read policy, queue, and every machine health record before acting.
Select the machine with the highest risk_score; queue order breaks ties. If its
risk is at least critical_risk, schedule it as urgent. Otherwise, if its risk is
at least service_risk, schedule it as planned. Otherwise record monitoring.
Choose exactly one action and notify maintenance exactly once afterward.""",
        functions=MAINTENANCE_FUNCTIONS,
        scenarios=_maintenance_scenarios(),
    ),
    "quality_sampling": TaskSpec(
        name="quality_sampling",
        title="Lot release, containment, or increased sampling",
        contract_version="quality-v1",
        instructions="""Read policy, lot, and every station quality record before acting.
Compute total_defect_rate from summed defects divided by summed inspected. If it
exceeds max_total_defect_rate, quarantine the lot. Otherwise find the first
station in lot order whose rework / inspected exceeds max_station_rework_rate
and increase sampling there. If neither applies, release the lot. Choose exactly
one action and notify quality exactly once afterward.""",
        functions=QUALITY_FUNCTIONS,
        scenarios=_quality_scenarios(),
    ),
    "energy_window": TaskSpec(
        name="energy_window",
        title="Energy-aware production-window scheduling",
        contract_version="energy-v2",
        instructions="""Read policy, pending job, and all candidate windows before acting.
The windows are already ordered by start_slot. A window is feasible exactly when
window["price_per_kwh"] <= policy["max_price_per_kwh"],
window["projected_load_kw"] <= policy["max_projected_load_kw"], and
window["start_slot"] + job["duration_slots"] <= job["deadline_slot"].
duration_slots belongs to the job and is not a window field. Schedule the first
feasible window. If none is feasible, defer the job. Choose exactly one action
and notify the energy desk exactly once afterward.""",
        functions=ENERGY_FUNCTIONS,
        scenarios=_energy_scenarios(),
    ),
    "spares_replenishment": TaskSpec(
        name="spares_replenishment",
        title="Spare-parts replenishment and shortage escalation",
        contract_version="inventory-v1",
        instructions="""Read the required part and every supplier option before acting.
shortage = max(required_quantity - on_hand, 0). If shortage is zero, record that
inventory is sufficient. Otherwise choose the first supplier whose available
quantity covers the shortage and whose lead_days is at most needed_in_days, and
order exactly the shortage. If no supplier is feasible, escalate the shortage.
Choose exactly one action and notify inventory exactly once afterward.""",
        functions=INVENTORY_FUNCTIONS,
        scenarios=_inventory_scenarios(),
    ),
}


@dataclass
class TaskRuntime:
    spec: TaskSpec
    scenario_name: str
    state: dict[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)
    actual: dict[str, Any] = field(
        default_factory=lambda: {
            "action": None,
            "target": None,
            "priority": None,
            "quantity": None,
        }
    )
    notified: bool = False
    trace: bool = False

    @classmethod
    def create(
        cls,
        task_name: str,
        scenario_name: str,
        *,
        trace: bool = False,
    ) -> "TaskRuntime":
        spec = TASKS[task_name]
        if scenario_name not in spec.scenarios:
            raise KeyError(f"Unknown scenario {scenario_name!r} for {task_name}")
        return cls(
            spec,
            scenario_name,
            copy.deepcopy(spec.scenarios[scenario_name]),
            trace=trace,
        )

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.trace:
            print(
                "MOCK API CALL   "
                + json.dumps({"name": name, "arguments": arguments}, sort_keys=True),
                flush=True,
            )
        try:
            result = self._invoke(name, arguments)
        except Exception as exc:
            result = {"ok": False, "error": f"runtime_exception: {type(exc).__name__}: {exc}"}
        self.calls.append(
            {
                "name": name,
                "arguments": copy.deepcopy(arguments),
                "result": copy.deepcopy(result),
            }
        )
        if self.trace:
            print("MOCK API RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return result

    def _read(self, key: str) -> dict[str, Any]:
        return {"ok": True, **copy.deepcopy(self.state[key])}

    def _action(self, action: str, target: Any, **extra: Any) -> dict[str, Any]:
        if self.actual["action"] is not None:
            return {"ok": False, "error": "action_already_recorded"}
        expected_reads = [
            name
            for name in self.state["expected_calls"]
            if name.startswith("get_")
        ]
        observed = Counter(item["name"] for item in self.calls)
        if any(observed[name] < count for name, count in Counter(expected_reads).items()):
            return {"ok": False, "error": "all_observations_required_before_action"}
        self.actual.update({"action": action, "target": target, **extra})
        return {"ok": True, "action": action, "target": target, **extra}

    def _notify(self, message: Any) -> dict[str, Any]:
        if self.actual["action"] is None:
            return {"ok": False, "error": "action_required_before_notification"}
        if self.notified:
            return {"ok": False, "error": "already_notified"}
        if not isinstance(message, str) or not message.strip():
            return {"ok": False, "error": "message_required"}
        self.notified = True
        return {"ok": True, "notified": True}

    @staticmethod
    def _reason(args: dict[str, Any], key: str = "reason") -> bool:
        value = args.get(key)
        return isinstance(value, str) and bool(value.strip())

    def _invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        task = self.spec.name
        if task == "batch_disposition":
            return self._batch(name, args)
        if task == "maintenance_priority":
            return self._maintenance(name, args)
        if task == "quality_sampling":
            return self._quality(name, args)
        if task == "energy_window":
            return self._energy(name, args)
        if task == "spares_replenishment":
            return self._inventory(name, args)
        return {"ok": False, "error": "unknown_task"}

    def _batch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        batch_id = self.state["context"]["batch_id"]
        if name == "get_production_context":
            return self._read("context")
        if name == "get_policy":
            return self._read("policy")
        if name == "get_quality_counts":
            return self._read("quality")
        if name == "get_sensor_summary":
            machine_id = args.get("machine_id")
            value = self.state["summaries"].get(machine_id)
            return (
                {"ok": True, "machine_id": machine_id, **copy.deepcopy(value)}
                if value
                else {"ok": False, "error": "unknown_machine"}
            )
        if name == "schedule_inspection":
            machine_id = args.get("machine_id")
            if machine_id not in self.state["summaries"]:
                return {"ok": False, "error": "unknown_machine"}
            if not self._reason(args):
                return {"ok": False, "error": "reason_required"}
            return self._action("inspection_pending", machine_id)
        if name == "hold_batch":
            if args.get("batch_id") != batch_id:
                return {"ok": False, "error": "unknown_batch"}
            if not self._reason(args):
                return {"ok": False, "error": "reason_required"}
            if self.actual["action"] != "inspection_pending":
                return {"ok": False, "error": "inspection_required_before_hold"}
            self.actual["action"] = "held"
            return {"ok": True, "action": "held"}
        if name in {"release_batch", "quarantine_batch"}:
            if args.get("batch_id") != batch_id:
                return {"ok": False, "error": "unknown_batch"}
            text_key = "evidence" if name == "release_batch" else "reason"
            if not self._reason(args, text_key):
                return {"ok": False, "error": f"{text_key}_required"}
            action = "released" if name == "release_batch" else "quarantined"
            return self._action(action, batch_id)
        if name == "notify_supervisor":
            return self._notify(args.get("message"))
        return {"ok": False, "error": "unknown_function"}

    def _maintenance(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "get_maintenance_policy":
            return self._read("policy")
        if name == "get_machine_queue":
            return self._read("queue")
        if name == "get_machine_health":
            machine_id = args.get("machine_id")
            value = self.state["health"].get(machine_id)
            return (
                {"ok": True, "machine_id": machine_id, **copy.deepcopy(value)}
                if value
                else {"ok": False, "error": "unknown_machine"}
            )
        if name == "schedule_maintenance":
            machine_id = args.get("machine_id")
            priority = args.get("priority")
            if machine_id not in self.state["health"]:
                return {"ok": False, "error": "unknown_machine"}
            if priority not in {"planned", "urgent"}:
                return {"ok": False, "error": "invalid_priority"}
            if not self._reason(args):
                return {"ok": False, "error": "reason_required"}
            return self._action("maintenance", machine_id, priority=priority)
        if name == "record_monitoring":
            if not self._reason(args):
                return {"ok": False, "error": "reason_required"}
            return self._action("monitoring", None)
        if name == "notify_maintenance":
            return self._notify(args.get("message"))
        return {"ok": False, "error": "unknown_function"}

    def _quality(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        lot_id = self.state["lot"]["lot_id"]
        if name == "get_inspection_policy":
            return self._read("policy")
        if name == "get_active_lot":
            return self._read("lot")
        if name == "get_station_quality":
            station_id = args.get("station_id")
            value = self.state["stations"].get(station_id)
            return (
                {"ok": True, "station_id": station_id, **copy.deepcopy(value)}
                if value
                else {"ok": False, "error": "unknown_station"}
            )
        if name in {"quarantine_lot", "release_lot"}:
            if args.get("lot_id") != lot_id:
                return {"ok": False, "error": "unknown_lot"}
            text_key = "reason" if name == "quarantine_lot" else "evidence"
            if not self._reason(args, text_key):
                return {"ok": False, "error": f"{text_key}_required"}
            action = "quarantined" if name == "quarantine_lot" else "released"
            return self._action(action, lot_id)
        if name == "increase_sampling":
            if args.get("lot_id") != lot_id:
                return {"ok": False, "error": "unknown_lot"}
            station_id = args.get("station_id")
            if station_id not in self.state["stations"]:
                return {"ok": False, "error": "unknown_station"}
            if not self._reason(args):
                return {"ok": False, "error": "reason_required"}
            return self._action("sampling", station_id)
        if name == "notify_quality":
            return self._notify(args.get("message"))
        return {"ok": False, "error": "unknown_function"}

    def _energy(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        job_id = self.state["job"]["job_id"]
        if name == "get_energy_policy":
            return self._read("policy")
        if name == "get_pending_energy_job":
            return self._read("job")
        if name == "get_candidate_windows":
            return {"ok": True, "windows": copy.deepcopy(self.state["windows"])}
        if name == "schedule_energy_job":
            if args.get("job_id") != job_id:
                return {"ok": False, "error": "unknown_job"}
            window_id = args.get("window_id")
            if window_id not in {item["window_id"] for item in self.state["windows"]}:
                return {"ok": False, "error": "unknown_window"}
            return self._action("scheduled", window_id)
        if name == "defer_energy_job":
            if args.get("job_id") != job_id:
                return {"ok": False, "error": "unknown_job"}
            if not self._reason(args):
                return {"ok": False, "error": "reason_required"}
            return self._action("deferred", job_id)
        if name == "notify_energy_desk":
            return self._notify(args.get("message"))
        return {"ok": False, "error": "unknown_function"}

    def _inventory(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        part = self.state["part"]
        part_id = part["part_id"]
        shortage = max(part["required_quantity"] - part["on_hand"], 0)
        if name == "get_required_part":
            return self._read("part")
        if name == "get_supplier_options":
            return {"ok": True, "suppliers": copy.deepcopy(self.state["suppliers"])}
        if name == "record_inventory_ok":
            if args.get("part_id") != part_id:
                return {"ok": False, "error": "unknown_part"}
            if not self._reason(args, "evidence"):
                return {"ok": False, "error": "evidence_required"}
            return self._action("inventory_ok", part_id, quantity=0)
        if name == "create_purchase_order":
            if args.get("part_id") != part_id:
                return {"ok": False, "error": "unknown_part"}
            supplier_id = args.get("supplier_id")
            supplier = next(
                (
                    item for item in self.state["suppliers"]
                    if item["supplier_id"] == supplier_id
                ),
                None,
            )
            if supplier is None:
                return {"ok": False, "error": "unknown_supplier"}
            quantity = args.get("quantity")
            if quantity != shortage:
                return {"ok": False, "error": "quantity_must_equal_shortage"}
            if supplier["available_quantity"] < shortage:
                return {"ok": False, "error": "supplier_capacity_insufficient"}
            if supplier["lead_days"] > part["needed_in_days"]:
                return {"ok": False, "error": "supplier_too_slow"}
            return self._action("ordered", supplier_id, quantity=quantity)
        if name == "escalate_shortage":
            if args.get("part_id") != part_id:
                return {"ok": False, "error": "unknown_part"}
            if not self._reason(args):
                return {"ok": False, "error": "reason_required"}
            return self._action("escalated", part_id, quantity=shortage)
        if name == "notify_inventory":
            return self._notify(args.get("message"))
        return {"ok": False, "error": "unknown_function"}

    def summary(self) -> dict[str, Any]:
        expected = self.state["expected"]
        actual = {
            key: self.actual.get(key)
            for key in expected
        }
        call_names = [item["name"] for item in self.calls]
        expected_calls = list(self.state["expected_calls"])
        errors = [
            {"name": item["name"], "error": item["result"].get("error")}
            for item in self.calls
            if not item["result"].get("ok")
        ]
        exact = Counter(call_names) == Counter(expected_calls)
        ordered = False
        if exact:
            read_names = {name for name in expected_calls if name.startswith("get_")}
            last_read = max(i for i, name in enumerate(call_names) if name in read_names)
            first_write = min(i for i, name in enumerate(call_names) if name not in read_names)
            ordered = last_read < first_write and call_names[-1].startswith("notify_")
        state_correct = actual == expected
        passed = exact and ordered and state_correct and self.notified and not errors
        return {
            "passed": passed,
            "task": self.spec.name,
            "scenario": self.scenario_name,
            "expected": copy.deepcopy(expected),
            "actual": actual,
            "call_names": call_names,
            "expected_calls": expected_calls,
            "trace_correct": exact and ordered,
            "state_correct": state_correct,
            "notified": self.notified,
            "tool_errors": errors,
        }


def build_api_module(spec: TaskSpec) -> str:
    lines = [
        "import json",
        "import os",
        "import urllib.request",
        "",
        '_BASE_URL = os.environ["PLANT_API_URL"]',
        "",
        "class APIError(RuntimeError):",
        "    pass",
        "",
        "def _call(name, arguments):",
        "    payload = json.dumps({'name': name, 'arguments': arguments}).encode('utf-8')",
        "    request = urllib.request.Request(",
        "        _BASE_URL + '/call',",
        "        data=payload,",
        "        method='POST',",
        "        headers={'Content-Type': 'application/json'},",
        "    )",
        "    with urllib.request.urlopen(request, timeout=2) as response:",
        "        result = json.loads(response.read().decode('utf-8'))",
        "    if not result.get('ok'):",
        "        raise APIError(result)",
        "    return {key: value for key, value in result.items() if key != 'ok'}",
        "",
    ]
    for function in spec.functions:
        params = ", ".join(function.parameters)
        args = ", ".join(f"'{name}': {name}" for name in function.parameters)
        lines.extend(
            [
                f"def {function.name}({params}):",
                f"    return _call('{function.name}', {{{args}}})",
                "",
            ]
        )
    return "\n".join(lines)
