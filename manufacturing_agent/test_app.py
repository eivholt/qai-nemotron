"""Tests for the manufacturing code-generation application."""

import tempfile
import unittest
from pathlib import Path

from manufacturing_agent.app import run_case, run_program


REFERENCE_PROGRAM = """from plant_api import (
    get_production_context,
    get_policy,
    get_quality_counts,
    get_sensor_summary,
    schedule_inspection,
    release_batch,
    hold_batch,
    quarantine_batch,
    notify_supervisor,
)

context = get_production_context()
policy = get_policy()
quality = get_quality_counts()
anomalous = []
for machine_id in context["machine_ids"]:
    summary = get_sensor_summary(machine_id)
    if (
        summary["max_temperature_c"] > policy["max_temperature_c"]
        or summary["vibration_rms"] > policy["max_vibration_rms"]
    ):
        anomalous.append(machine_id)

defect_rate = quality["defects"] / quality["inspected"]
batch_id = context["batch_id"]
if defect_rate > policy["max_defect_rate"]:
    quarantine_batch(batch_id, "defect rate exceeds policy")
elif anomalous:
    schedule_inspection(anomalous[0], "sensor limit exceeded")
    hold_batch(batch_id, "inspection required")
else:
    release_batch(batch_id, "quality and sensor readings within policy")
notify_supervisor("batch analysis completed")
"""


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def generate(self, _prompt: str) -> dict[str, object]:
        if self.calls >= len(self.outputs):
            raise AssertionError("Unexpected model generation")
        text = self.outputs[self.calls]
        self.calls += 1
        return {"text": f"```python\n{text}\n```", "elapsed_s": 0.01}


class ManufacturingAgentTests(unittest.TestCase):
    def test_reference_program_passes_every_scenario(self) -> None:
        for name in (
            "normal",
            "bearing_vibration",
            "thermal_drift",
            "quality_spike",
        ):
            with self.subTest(name=name):
                result = run_program(REFERENCE_PROGRAM, name)
                self.assertTrue(result["passed"], result)

    def test_extra_call_fails_strict_ledger(self) -> None:
        result = run_program(
            REFERENCE_PROGRAM + "\nget_policy()\n",
            "normal",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "wrong_state_or_trace")

    def test_external_network_is_blocked(self) -> None:
        result = run_program(
            "import urllib.request\nurllib.request.urlopen('http://example.com')\n",
            "normal",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "sandbox_violation")

    def test_subprocess_is_blocked(self) -> None:
        result = run_program(
            "import subprocess\nsubprocess.run(['/bin/true'])\n",
            "normal",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "sandbox_violation")

    def test_execute_first_reuses_code_for_changed_sensor_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "policy.py"
            seed_client = FakeClient([REFERENCE_PROGRAM])
            seed = run_case(
                seed_client,
                "normal",
                cache,
                "execute_first",
                repair_retries=0,
            )
            self.assertTrue(seed["passed"])
            self.assertEqual(seed_client.calls, 1)

            reuse_client = FakeClient([])
            reused = run_case(
                reuse_client,
                "bearing_vibration",
                cache,
                "execute_first",
                repair_retries=0,
            )
            self.assertTrue(reused["passed"])
            self.assertTrue(reused["cache_hit"])
            self.assertEqual(reuse_client.calls, 0)

    def test_execution_feedback_can_repair_program(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient(["raise RuntimeError('broken')", REFERENCE_PROGRAM])
            result = run_case(
                client,
                "thermal_drift",
                Path(temp) / "policy.py",
                "none",
                repair_retries=1,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(client.calls, 2)
        self.assertEqual(result["attempts"][0]["failure"], "runtime_error")


if __name__ == "__main__":
    unittest.main()
