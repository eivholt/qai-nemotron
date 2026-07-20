import plant_api

def schedule_energy_job():
    policy = plant_api.get_energy_policy()
    job = plant_api.get_pending_energy_job()
    windows = plant_api.get_candidate_windows()

    for window in windows['windows']:
        if (window['price_per_kwh'] <= policy['max_price_per_kwh'] and
            window['projected_load_kw'] <= policy['max_projected_load_kw'] and
            window['start_slot'] + job['duration_slots'] <= job['deadline_slot']):
            plant_api.schedule_energy_job(job['job_id'], window['window_id'])
            plant_api.notify_energy_desk(f"Scheduled job {job['job_id']} in window {window['window_id']}")
            return

    plant_api.defer_energy_job(job['job_id'], "No feasible window found")
    plant_api.notify_energy_desk(f"Deferred job {job['job_id']} due to no feasible window")

schedule_energy_job()