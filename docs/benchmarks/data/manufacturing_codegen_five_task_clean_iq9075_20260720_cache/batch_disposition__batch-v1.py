import plant_api

def batch_disposition():
    context = plant_api.get_production_context()
    policy = plant_api.get_policy()
    quality = plant_api.get_quality_counts()
    machine_data = {m: plant_api.get_sensor_summary(m) for m in context["machine_ids"]}

    defect_rate = quality["defects"] / quality["inspected"] if quality["inspected"] > 0 else 0
    if defect_rate > policy["max_defect_rate"]:
        plant_api.quarantine_batch(context["batch_id"], "high_defect_rate")
        plant_api.notify_supervisor("Batch quarantined due to defect rate exceeding policy threshold.")
    else:
        inspected_machine = None
        for machine in context["machine_ids"]:
            sensor = machine_data[machine]
            if sensor["max_temperature_c"] > policy["max_temperature_c"] or sensor["vibration_rms"] > policy["max_vibration_rms"]:
                inspected_machine = machine
                plant_api.schedule_inspection(machine, "machine_violation")
                break
        if inspected_machine:
            plant_api.hold_batch(context["batch_id"], f"machine_{inspected_machine}_violation")
            plant_api.notify_supervisor(f"Batch held due to machine {inspected_machine} exceeding limits.")
        else:
            plant_api.release_batch(context["batch_id"], "no_violations")
            plant_api.notify_supervisor("Batch released as all conditions met.")

batch_disposition()