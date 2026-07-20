"""Unit tests for the autonomous shipping world's tools and strict scoring."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from shipping_agent.app import build_direct_agent
from shipping_agent.mcp_server import ShippingMCPService
from shipping_agent.runtime import ShippingRuntime, run_scripted


class ShippingRuntimeTests(unittest.TestCase):
    def test_scripted_scenarios_reach_expected_state(self) -> None:
        for name in ("routine", "cold_chain", "weather_hold", "no_compliant_carrier"):
            with self.subTest(name=name):
                runtime = run_scripted(name, verbose=False)
                self.assertTrue(runtime.summary()["passed"])

    def test_cold_chain_rejects_standard_carrier(self) -> None:
        runtime = ShippingRuntime.from_scenario("cold_chain", verbose=False)
        shipment_id = runtime.shipment["shipment_id"]
        result = runtime.schedule_shipment(shipment_id, "TRUCK-7", "COLD-1")
        self.assertEqual(result["error"], "carrier_not_compliant")
        self.assertEqual(runtime.shipment["status"], "pending")

    def test_options_explain_shipment_specific_eligibility(self) -> None:
        runtime = ShippingRuntime.from_scenario("weather_hold", verbose=False)
        result = runtime.get_shipping_options(runtime.shipment["shipment_id"])
        self.assertEqual(result["usable_carriers"], [])
        blocked = result["temporarily_blocked_carriers"]
        self.assertEqual([item["carrier_id"] for item in blocked], ["TRUCK-9"])
        self.assertIn("route_temporarily_closed", blocked[0]["issues"])
        excluded = result["excluded_carriers"]
        self.assertEqual([item["carrier_id"] for item in excluded], ["VAN-3"])
        self.assertIn("insufficient_capacity", excluded[0]["issues"])

    def test_schedule_has_no_client_stage_gate(self) -> None:
        runtime = ShippingRuntime.from_scenario("routine", verbose=False)
        shipment_id = runtime.shipment["shipment_id"]
        result = runtime.schedule_shipment(shipment_id, "TRUCK-7", "D1")
        self.assertTrue(result["ok"])
        self.assertEqual(runtime.shipment["status"], "scheduled")

    def test_disposition_results_expose_notification_state(self) -> None:
        runtime = ShippingRuntime.from_scenario("routine", verbose=False)
        shipment_id = runtime.shipment["shipment_id"]
        planned = runtime.schedule_shipment(shipment_id, "TRUCK-7", "D1")
        self.assertFalse(planned["dispatch_notified"])

        notified = runtime.notify_dispatch(shipment_id)
        self.assertTrue(notified["dispatch_notified"])


    def test_dispatch_notification_requires_a_plan(self) -> None:
        runtime = ShippingRuntime.from_scenario("routine", verbose=False)
        shipment_id = runtime.shipment["shipment_id"]
        result = runtime.notify_dispatch(shipment_id)
        self.assertEqual(result["error"], "plan_shipment_before_notification")

    def test_hold_is_an_agent_decision_not_a_python_policy(self) -> None:
        runtime = ShippingRuntime.from_scenario("routine", verbose=False)
        shipment_id = runtime.shipment["shipment_id"]
        result = runtime.hold_shipment(shipment_id, "wait")
        self.assertTrue(result["ok"])
        self.assertEqual(runtime.shipment["status"], "held")

    def test_escalation_is_an_agent_decision_not_a_python_policy(self) -> None:
        runtime = ShippingRuntime.from_scenario("routine", verbose=False)
        shipment_id = runtime.shipment["shipment_id"]
        result = runtime.escalate_shipment(shipment_id, "operator review")
        self.assertTrue(result["ok"])
        self.assertEqual(runtime.shipment["status"], "escalated")

    @unittest.skipUnless(
        importlib.util.find_spec("pydantic_ai"),
        "Pydantic AI client dependency is not installed",
    )
    def test_direct_agent_registers_tools_without_annotation_errors(self) -> None:
        agent = build_direct_agent("http://127.0.0.1:1/v1", "local", "test")
        self.assertIsNotNone(agent)

    def test_mcp_service_persists_the_same_strict_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state_file = Path(temp) / "state.json"
            service = ShippingMCPService("routine", state_file)
            result = service.invoke(
                "get_pending_shipment",
                {},
                service.runtime.get_pending_shipment,
            )
            snapshot = json.loads(state_file.read_text())
        self.assertTrue(result["ok"])
        self.assertEqual(snapshot["calls"][0]["name"], "get_pending_shipment")
        self.assertEqual(snapshot["summary"]["tool_calls"], 1)

    def test_strict_score_rejects_a_safe_but_wrong_decision(self) -> None:
        runtime = ShippingRuntime.from_scenario("routine", verbose=False)
        shipment_id = runtime.shipment["shipment_id"]
        runtime.call("get_pending_shipment", {}, runtime.get_pending_shipment)
        runtime.call(
            "get_shipping_options",
            {"shipment_id": shipment_id},
            lambda: runtime.get_shipping_options(shipment_id),
        )
        runtime.call(
            "hold_shipment",
            {"shipment_id": shipment_id, "reason": "unnecessary hold"},
            lambda: runtime.hold_shipment(shipment_id, "unnecessary hold"),
        )
        runtime.call(
            "notify_dispatch",
            {"shipment_id": shipment_id},
            lambda: runtime.notify_dispatch(shipment_id),
        )
        summary = runtime.summary()
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["state_correct"])

    def test_strict_score_rejects_an_extra_failed_call(self) -> None:
        runtime = run_scripted("routine", verbose=False)
        runtime.calls.insert(
            2,
            {
                "name": "schedule_shipment",
                "arguments": {"carrier_id": "VAN-3"},
                "result": {"ok": False, "error": "insufficient_capacity"},
            },
        )
        summary = runtime.summary()
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["trace_correct"])
        self.assertEqual(len(summary["tool_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
