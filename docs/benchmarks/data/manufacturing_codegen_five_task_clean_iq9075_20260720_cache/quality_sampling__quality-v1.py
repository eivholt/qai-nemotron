import plant_api

def decide_lot_action():
    policy = plant_api.get_inspection_policy()
    lot = plant_api.get_active_lot()
    stations = [plant_api.get_station_quality(st) for st in lot['station_ids']]

    total_inspected = sum(station['inspected'] for station in stations)
    total_defects = sum(station['defects'] for station in stations)
    total_defect_rate = total_defects / total_inspected if total_inspected > 0 else 0

    if total_defect_rate > policy['max_total_defect_rate']:
        plant_api.quarantine_lot(lot['lot_id'], "High defect rate")
    else:
        max_rework = max(station['rework'] / station['inspected'] for station in stations if station['inspected'] > 0)
        if max_rework > policy['max_station_rework_rate']:
            for station in stations:
                if station['rework'] / station['inspected'] > policy['max_station_rework_rate']:
                    plant_api.increase_sampling(lot['lot_id'], station['station_id'], "High rework rate")
                    break
        else:
            plant_api.release_lot(lot['lot_id'], "No issues detected")

    plant_api.notify_quality("Lot action completed")
decide_lot_action()