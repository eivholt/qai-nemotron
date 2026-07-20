"""Tests for the five-task manufacturing code-generation suite."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from manufacturing_agent.codegen_suite import (
    contract_hash,
    run_case,
    run_program,
    save_cache,
    validate_for_promotion,
)
from manufacturing_agent.codegen_tasks import TASKS


REFERENCE_PROGRAMS = {
    "batch_disposition": """import plant_api

context = plant_api.get_production_context()
policy = plant_api.get_policy()
quality = plant_api.get_quality_counts()
summaries = [
    plant_api.get_sensor_summary(machine_id)
    for machine_id in context["machine_ids"]
]
defect_rate = quality["defects"] / quality["inspected"]
if defect_rate > policy["max_defect_rate"]:
    plant_api.quarantine_batch(context["batch_id"], "defect rate")
elif any(
    item["max_temperature_c"] > policy["max_temperature_c"]
    or item["vibration_rms"] > policy["max_vibration_rms"]
    for item in summaries
):
    target = next(
        item for item in summaries
        if item["max_temperature_c"] > policy["max_temperature_c"]
        or item["vibration_rms"] > policy["max_vibration_rms"]
    )
    plant_api.schedule_inspection(target["machine_id"], "sensor threshold")
    plant_api.hold_batch(context["batch_id"], "inspection required")
else:
    plant_api.release_batch(context["batch_id"], "within policy")
plant_api.notify_supervisor("batch disposition complete")
""",
    "maintenance_priority": """import plant_api

policy = plant_api.get_maintenance_policy()
queue = plant_api.get_machine_queue()
health = [
    plant_api.get_machine_health(machine_id)
    for machine_id in queue["machine_ids"]
]
highest = max(health, key=lambda item: item["risk_score"])
if highest["risk_score"] >= policy["critical_risk"]:
    plant_api.schedule_maintenance(highest["machine_id"], "urgent", "critical risk")
elif highest["risk_score"] >= policy["service_risk"]:
    plant_api.schedule_maintenance(highest["machine_id"], "planned", "service risk")
else:
    plant_api.record_monitoring("all machines below service threshold")
plant_api.notify_maintenance("maintenance triage complete")
""",
    "quality_sampling": """import plant_api

policy = plant_api.get_inspection_policy()
lot = plant_api.get_active_lot()
stations = [
    plant_api.get_station_quality(station_id)
    for station_id in lot["station_ids"]
]
total_inspected = sum(item["inspected"] for item in stations)
total_defects = sum(item["defects"] for item in stations)
total_rate = total_defects / total_inspected
if total_rate > policy["max_total_defect_rate"]:
    plant_api.quarantine_lot(lot["lot_id"], "total defect rate")
else:
    high_rework = next(
        (
            item for item in stations
            if item["rework"] / item["inspected"]
            > policy["max_station_rework_rate"]
        ),
        None,
    )
    if high_rework:
        plant_api.increase_sampling(
            lot["lot_id"], high_rework["station_id"], "station rework rate"
        )
    else:
        plant_api.release_lot(lot["lot_id"], "quality within policy")
plant_api.notify_quality("lot quality decision complete")
""",
    "energy_window": """import plant_api

policy = plant_api.get_energy_policy()
job = plant_api.get_pending_energy_job()
windows = plant_api.get_candidate_windows()["windows"]
selected = next(
    (
        window for window in windows
        if window["price_per_kwh"] <= policy["max_price_per_kwh"]
        and window["projected_load_kw"] <= policy["max_projected_load_kw"]
        and window["start_slot"] + job["duration_slots"] <= job["deadline_slot"]
    ),
    None,
)
if selected:
    plant_api.schedule_energy_job(job["job_id"], selected["window_id"])
else:
    plant_api.defer_energy_job(job["job_id"], "no feasible energy window")
plant_api.notify_energy_desk("energy scheduling complete")
""",
    "spares_replenishment": """import plant_api

part = plant_api.get_required_part()
suppliers = plant_api.get_supplier_options()["suppliers"]
shortage = max(part["required_quantity"] - part["on_hand"], 0)
if shortage == 0:
    plant_api.record_inventory_ok(part["part_id"], "stock sufficient")
else:
    supplier = next(
        (
            item for item in suppliers
            if item["available_quantity"] >= shortage
            and item["lead_days"] <= part["needed_in_days"]
        ),
        None,
    )
    if supplier:
        plant_api.create_purchase_order(
            part["part_id"], supplier["supplier_id"], shortage
        )
    else:
        plant_api.escalate_shortage(part["part_id"], "no feasible supplier")
plant_api.notify_inventory("inventory decision complete")
""",
}


class FakeClient:
    def __init__(self, outputs: list[str], model: str = "fake-model") -> None:
        self.outputs = outputs
        self.model = model
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        if self.calls >= len(self.outputs):
            raise AssertionError("Unexpected model generation")
        text = self.outputs[self.calls]
        self.calls += 1
        return {"text": text, "elapsed_s": 0.01}


class CodegenSuiteTests(unittest.TestCase):
    def test_reference_programs_pass_all_fifteen_cases(self) -> None:
        for task_name, spec in TASKS.items():
            for scenario_name in spec.scenarios:
                with self.subTest(task=task_name, scenario=scenario_name):
                    result = run_program(
                        REFERENCE_PROGRAMS[task_name],
                        task_name,
                        scenario_name,
                    )
                    self.assertTrue(result["passed"], result)

    def test_trace_api_prints_child_requests_and_parent_results(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = run_program(
                REFERENCE_PROGRAMS["energy_window"],
                "energy_window",
                "first_window",
                trace_api=True,
            )

        self.assertTrue(result["passed"])
        trace = output.getvalue()
        self.assertIn("MOCK API CALL", trace)
        self.assertIn("get_candidate_windows", trace)
        self.assertIn("MOCK API RESULT", trace)
    def test_source_preflight_rejects_non_api_import(self) -> None:
        result = run_program(
            "import os\nimport plant_api\n",
            "batch_disposition",
            "normal",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "source_rejected")
        self.assertIn("import not allowed: os", result["source_errors"])

    def test_extra_api_call_fails_exact_ledger(self) -> None:
        code = REFERENCE_PROGRAMS["energy_window"] + "\nplant_api.get_energy_policy()\n"
        result = run_program(code, "energy_window", "first_window")
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "wrong_state_or_trace")

    def test_wrong_job_id_is_rejected_by_mock_api(self) -> None:
        code = REFERENCE_PROGRAMS["energy_window"].replace(
            'plant_api.schedule_energy_job(job["job_id"], selected["window_id"])',
            'plant_api.schedule_energy_job("WRONG-JOB", selected["window_id"])',
        )
        result = run_program(code, "energy_window", "first_window")

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "runtime_error")
        self.assertIn("unknown_job", result["execution"]["stderr"])
    def test_promoted_cache_is_revalidated_without_model_call(self) -> None:
        spec = TASKS["maintenance_priority"]
        code = REFERENCE_PROGRAMS[spec.name]
        validations = validate_for_promotion(code, spec)
        self.assertTrue(all(item["passed"] for item in validations.values()))

        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "maintenance.json"
            save_cache(cache_path, spec, code, validations, "fixture")
            client = FakeClient([])
            result = run_case(client, spec, "urgent", cache_path, repair_retries=0)

        self.assertTrue(result["passed"])
        self.assertTrue(result["cache_hit"])
        self.assertFalse(result["model_called"])
        self.assertEqual(client.calls, 0)
        self.assertEqual(
            result["cache_check"]["fresh_validation"]["summary"]["actual"]["priority"],
            "urgent",
        )

    def test_contract_mismatch_forces_generation(self) -> None:
        spec = TASKS["energy_window"]
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "energy.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "contract_hash": "old-contract",
                        "code": REFERENCE_PROGRAMS[spec.name],
                    }
                )
            )
            client = FakeClient([REFERENCE_PROGRAMS[spec.name]])
            result = run_case(
                client,
                spec,
                "first_window",
                cache_path,
                repair_retries=0,
            )

        self.assertTrue(result["passed"])
        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["cache_check"]["status"], "contract_mismatch")
        self.assertEqual(client.calls, 1)

    def test_failed_fresh_cache_validation_requests_replacement(self) -> None:
        spec = TASKS["batch_disposition"]
        bad_code = """import plant_api
context = plant_api.get_production_context()
policy = plant_api.get_policy()
quality = plant_api.get_quality_counts()
for machine_id in context["machine_ids"]:
    plant_api.get_sensor_summary(machine_id)
plant_api.release_batch(context["batch_id"], "always release")
plant_api.notify_supervisor("done")
"""
        normal_only = validate_for_promotion(
            REFERENCE_PROGRAMS[spec.name],
            spec,
        )
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "batch.json"
            save_cache(cache_path, spec, bad_code, normal_only, "bad-fixture")
            client = FakeClient([REFERENCE_PROGRAMS[spec.name]])
            result = run_case(
                client,
                spec,
                "vibration",
                cache_path,
                repair_retries=0,
            )

        self.assertTrue(result["passed"])
        self.assertFalse(result["cache_hit"])
        self.assertEqual(
            result["cache_check"]["status"],
            "fresh_validation_failed",
        )
        self.assertEqual(client.calls, 1)
        self.assertIn("previous program failed", client.prompts[0])

    def test_promotion_rejects_code_that_only_passes_seed_case(self) -> None:
        spec = TASKS["batch_disposition"]
        seed_only = """import plant_api
context = plant_api.get_production_context()
plant_api.get_policy()
plant_api.get_quality_counts()
for machine_id in context["machine_ids"]:
    plant_api.get_sensor_summary(machine_id)
plant_api.release_batch(context["batch_id"], "always release")
plant_api.notify_supervisor("done")
"""
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "batch.json"
            client = FakeClient([seed_only])
            result = run_case(
                client,
                spec,
                "normal",
                cache_path,
                repair_retries=0,
            )
            cache_exists = cache_path.exists()

        self.assertFalse(result["passed"])
        self.assertTrue(result["promotion"]["normal"]["passed"])
        self.assertFalse(result["promotion"]["vibration"]["passed"])
        self.assertFalse(cache_exists)


if __name__ == "__main__":
    unittest.main()
