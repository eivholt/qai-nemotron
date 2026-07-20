"""Tests for generated shipping-code execution and reuse."""

import tempfile
import unittest
from pathlib import Path

from shipping_agent.codegen_demo import run_case, run_program


GENERIC_CODE = """from scenario_api import (
    get_pending_shipment,
    get_shipping_options,
    schedule_shipment,
    hold_shipment,
    escalate_shipment,
    notify_dispatch,
)

shipment = get_pending_shipment()
options = get_shipping_options()
if options["usable_carriers"] and options["usable_docks"]:
    carrier_id = options["usable_carriers"][0]["carrier_id"]
    dock_id = options["usable_docks"][0]["dock_id"]
    result = schedule_shipment(carrier_id, dock_id)
elif options["temporarily_blocked_carriers"]:
    result = hold_shipment("compatible route temporarily blocked")
else:
    result = escalate_shipment("no compatible carrier available")
if not result.get("ok"):
    raise RuntimeError(result)
notice = notify_dispatch()
if not notice.get("ok"):
    raise RuntimeError(notice)
"""


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def generate(self, _prompt: str) -> dict[str, object]:
        if self.calls >= len(self.outputs):
            raise AssertionError("Unexpected model call")
        text = self.outputs[self.calls]
        self.calls += 1
        return {"text": f"```python\n{text}\n```", "elapsed_s": 0.01}


class CodegenDemoTests(unittest.TestCase):
    def test_generic_program_passes_all_scenarios(self) -> None:
        for scenario in (
            "routine",
            "cold_chain",
            "weather_hold",
            "no_compliant_carrier",
        ):
            with self.subTest(scenario=scenario):
                result = run_program(GENERIC_CODE, scenario)
                self.assertTrue(result["passed"], result)

    def test_extra_api_call_fails_strict_trace(self) -> None:
        result = run_program(
            GENERIC_CODE + "\nget_pending_shipment()\n",
            "routine",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "wrong_state_or_trace")

    def test_external_network_is_blocked(self) -> None:
        result = run_program(
            "import urllib.request\nurllib.request.urlopen('http://example.com')\n",
            "routine",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "sandbox_violation")

    def test_subprocess_is_blocked(self) -> None:
        result = run_program(
            "import subprocess\nsubprocess.run(['/bin/true'])\n",
            "routine",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"], "sandbox_violation")

    def test_execute_first_reuses_validated_code_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "shipping.py"
            seed_client = FakeClient([GENERIC_CODE])
            seed = run_case(
                seed_client,
                "routine",
                cache,
                "execute_first",
                repair_retries=0,
            )
            self.assertTrue(seed["passed"])
            self.assertEqual(seed_client.calls, 1)

            reuse_client = FakeClient([])
            reused = run_case(
                reuse_client,
                "cold_chain",
                cache,
                "execute_first",
                repair_retries=0,
            )
            self.assertTrue(reused["passed"])
            self.assertTrue(reused["cache_hit"])
            self.assertEqual(reuse_client.calls, 0)

    def test_runtime_failure_can_be_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient(["raise RuntimeError('first attempt')", GENERIC_CODE])
            result = run_case(
                client,
                "weather_hold",
                Path(temp) / "shipping.py",
                "none",
                repair_retries=1,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["failure"], "runtime_error")


if __name__ == "__main__":
    unittest.main()
