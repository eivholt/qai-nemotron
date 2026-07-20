import plant_api

def run_maintenance_priority():
    policy = plant_api.get_maintenance_policy()
    queue = plant_api.get_machine_queue()
    highest_risk_machine = None
    highest_risk = -1.0

    for machine_id in queue['machine_ids']:
        health = plant_api.get_machine_health(machine_id)
        if health['risk_score'] > highest_risk:
            highest_risk = health['risk_score']
            highest_risk_machine = health

    if highest_risk_machine:
        if highest_risk >= policy['critical_risk']:
            plant_api.schedule_maintenance(highest_risk_machine['machine_id'], 'urgent', 'high risk detected')
        elif highest_risk >= policy['service_risk']:
            plant_api.schedule_maintenance(highest_risk_machine['machine_id'], 'planned', 'service required')
        else:
            plant_api.record_monitoring('no service needed')
        plant_api.notify_maintenance(f"Action taken for {highest_risk_machine['machine_id']}")

run_maintenance_priority()