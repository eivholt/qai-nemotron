#!/usr/bin/env python3
"""Expose the shipping simulator as a real MCP stdio server."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

from shipping_agent.runtime import SCENARIOS, ShippingRuntime


class ShippingMCPService:
    """Own one scenario and persist its exact tool ledger for strict scoring."""

    def __init__(self, scenario_name: str, state_file: Path) -> None:
        self.runtime = ShippingRuntime.from_scenario(scenario_name, verbose=False)
        self.state_file = state_file
        self.persist()

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        result = self.runtime.call(name, arguments, operation)
        self.persist()
        return result

    def persist(self) -> None:
        snapshot = {
            "summary": self.runtime.summary(),
            "calls": self.runtime.calls,
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(snapshot, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )


def build_server(service: ShippingMCPService) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MCP support is not installed. Run: "
            "python3 -m pip install -r shipping_agent/requirements.txt"
        ) from exc

    mcp = FastMCP(
        "shipping-agent-tools",
        instructions=(
            "Use shipping tools to inspect one pending shipment, choose a safe "
            "disposition, and notify dispatch."
        ),
    )

    @mcp.tool()
    def get_pending_shipment() -> dict[str, Any]:
        """Return the pending shipment, including ID, load, handling, and deadline."""
        runtime = service.runtime
        return service.invoke(
            "get_pending_shipment",
            {},
            runtime.get_pending_shipment,
        )

    @mcp.tool()
    def get_shipping_options() -> dict[str, Any]:
        """Group current-shipment options into usable, blocked, and excluded lists."""
        runtime = service.runtime
        shipment_id = runtime.shipment["shipment_id"]
        return service.invoke(
            "get_shipping_options",
            {},
            lambda: runtime.get_shipping_options(shipment_id),
        )

    @mcp.tool()
    def schedule_shipment(
        carrier_id: str,
        dock_id: str,
    ) -> dict[str, Any]:
        """Schedule IDs from usable_carriers and usable_docks; never use other lists."""
        runtime = service.runtime
        shipment_id = runtime.shipment["shipment_id"]
        arguments = {"carrier_id": carrier_id, "dock_id": dock_id}
        return service.invoke(
            "schedule_shipment",
            arguments,
            lambda: runtime.schedule_shipment(shipment_id, carrier_id, dock_id),
        )

    @mcp.tool()
    def hold_shipment(reason: str) -> dict[str, Any]:
        """Temporarily hold only when an otherwise usable carrier is expected to recover soon."""
        runtime = service.runtime
        shipment_id = runtime.shipment["shipment_id"]
        arguments = {"reason": reason}
        return service.invoke(
            "hold_shipment",
            arguments,
            lambda: runtime.hold_shipment(shipment_id, reason),
        )

    @mcp.tool()
    def escalate_shipment(reason: str) -> dict[str, Any]:
        """Escalate when no compatible carrier is available now or expected to recover."""
        runtime = service.runtime
        shipment_id = runtime.shipment["shipment_id"]
        arguments = {"reason": reason}
        return service.invoke(
            "escalate_shipment",
            arguments,
            lambda: runtime.escalate_shipment(shipment_id, reason),
        )

    @mcp.tool()
    def notify_dispatch() -> dict[str, Any]:
        """Notify dispatch after the current shipment is scheduled, held, or escalated."""
        runtime = service.runtime
        shipment_id = runtime.shipment["shipment_id"]
        return service.invoke(
            "notify_dispatch",
            {},
            lambda: runtime.notify_dispatch(shipment_id),
        )

    return mcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
    service = ShippingMCPService(args.scenario, args.state_file)
    build_server(service).run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
