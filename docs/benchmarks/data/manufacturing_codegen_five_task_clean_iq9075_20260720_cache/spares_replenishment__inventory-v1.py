import plant_api

def run_spare_parts_pipeline():
    part_data = plant_api.get_required_part()
    suppliers = plant_api.get_supplier_options()

    shortage = max(part_data["required_quantity"] - part_data["on_hand"], 0)
    if shortage == 0:
        plant_api.record_inventory_ok(part_data["part_id"], "sufficient stock")
        plant_api.notify_inventory("Inventory is sufficient.")
    else:
        feasible_supplier = next(
            (supplier for supplier in suppliers["suppliers"]
             if supplier["available_quantity"] >= shortage and supplier["lead_days"] <= part_data["needed_in_days"]),
            None
        )
        if feasible_supplier:
            plant_api.create_purchase_order(
                part_data["part_id"], feasible_supplier["supplier_id"], shortage
            )
            plant_api.notify_inventory(f"Ordered {shortage} units from {feasible_supplier['supplier_id']}.")
        else:
            plant_api.escalate_shortage(part_data["part_id"], "no feasible supplier found")
            plant_api.notify_inventory("Shortage escalated due to no feasible supplier.")

run_spare_parts_pipeline()