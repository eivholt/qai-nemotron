"""Tests for the experimental multi-tool shipping case."""

import copy
import unittest

from shipping_agent.multitool_demo import (
    EXPECTED_CALLS,
    MULTI_TOOL_SCENARIO,
    score_multitool,
)
from shipping_agent.runtime import ShippingRuntime


def completed_runtime(*, extra_call: bool = False) -> ShippingRuntime:
    runtime = ShippingRuntime(copy.deepcopy(MULTI_TOOL_SCENARIO), verbose=False)
    shipment_id = runtime.shipment["shipment_id"]
    runtime.call("get_pending_shipment", {}, runtime.get_pending_shipment)
    runtime.call("get_carrier_options", {}, runtime.get_carrier_options)
    runtime.call("get_dock_options", {}, runtime.get_dock_options)
    runtime.call(
        "schedule_shipment",
        {"carrier_id": "REEFER-2", "dock_id": "COLD-1"},
        lambda: runtime.schedule_shipment(
            shipment_id,
            "REEFER-2",
            "COLD-1",
        ),
    )
    runtime.call(
        "notify_dispatch",
        {},
        lambda: runtime.notify_dispatch(shipment_id),
    )
    if extra_call:
        runtime.call("get_pending_shipment", {}, runtime.get_pending_shipment)
    return runtime


class MultiToolRuntimeTests(unittest.TestCase):
    def test_split_options_match_combined_options(self) -> None:
        runtime = ShippingRuntime(copy.deepcopy(MULTI_TOOL_SCENARIO), verbose=False)
        combined = runtime.get_shipping_options(runtime.shipment["shipment_id"])
        carriers = runtime.get_carrier_options()
        docks = runtime.get_dock_options()

        self.assertEqual(combined["usable_carriers"], carriers["usable_carriers"])
        self.assertEqual(
            combined["temporarily_blocked_carriers"],
            carriers["temporarily_blocked_carriers"],
        )
        self.assertEqual(
            combined["excluded_carriers"],
            carriers["excluded_carriers"],
        )
        self.assertEqual(combined["usable_docks"], docks["usable_docks"])
        self.assertEqual(combined["excluded_docks"], docks["excluded_docks"])

    def test_task_and_batching_pass(self) -> None:
        summary = score_multitool(
            completed_runtime(),
            [
                EXPECTED_CALLS[:3],
                ["schedule_shipment"],
                ["notify_dispatch"],
            ],
        )

        self.assertTrue(summary["task_passed"])
        self.assertTrue(summary["batching_passed"])
        self.assertTrue(summary["passed"])

    def test_sequential_calls_pass_task_but_fail_batching(self) -> None:
        summary = score_multitool(
            completed_runtime(),
            [[name] for name in EXPECTED_CALLS],
        )

        self.assertTrue(summary["task_passed"])
        self.assertFalse(summary["batching_passed"])
        self.assertFalse(summary["passed"])

    def test_two_read_calls_in_one_batch_are_enough(self) -> None:
        summary = score_multitool(
            completed_runtime(),
            [
                ["get_pending_shipment"],
                ["get_carrier_options", "get_dock_options"],
                ["schedule_shipment"],
                ["notify_dispatch"],
            ],
        )

        self.assertTrue(summary["task_passed"])
        self.assertTrue(summary["batching_passed"])
        self.assertTrue(summary["passed"])

    def test_extra_call_fails_strict_task_score(self) -> None:
        summary = score_multitool(
            completed_runtime(extra_call=True),
            [
                EXPECTED_CALLS[:3],
                ["schedule_shipment"],
                ["notify_dispatch"],
                ["get_pending_shipment"],
            ],
        )
