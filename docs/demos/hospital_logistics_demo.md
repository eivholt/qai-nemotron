# Hospital Logistics Coordinator Demo

This demo models a hospital edge agent that coordinates non-emergency internal logistics. It plans jobs, checks constraints, assigns porters or robots, reserves elevators, updates status, notifies wards, and escalates conflicts. It does not directly drive robots, diagnose, or prescribe.

Why this fits edge AI on the EVK:

- Hospital logistics can contain sensitive operational context.
- Local execution helps during cloud or network outages.
- Latency and continuity matter when samples, medication totes, blood products, and sterile instruments have time constraints.
- Logistics failures are expensive enough to justify a local edge device better than a toy robot demo.

Implemented files:

- `agent_arena/hospital_logistics_runtime.py`: deterministic mock hospital state, tools, scenarios, and scorer.
- `agent_arena/pydantic_hospital_logistics_arena.py`: Pydantic AI function-tool runner.
- `agent_arena/run_host_pydantic_hospital_probe.sh`: host-side runner for EVK OpenAI-compatible endpoints.

Tool surface:

- `get_pending_jobs`
- `get_asset_location`
- `check_elevator_status`
- `reserve_elevator`
- `assign_porter`
- `assign_robot`
- `check_cold_chain_window`
- `notify_ward`
- `escalate_to_human`
- `update_job_status`
- `query_policy`

Initial scenarios:

| Scenario | Focus |
|---|---|
| `hospital_L1_sample_elevator_out` | Blood sample deadline, one elevator out, use alternate elevator and porter. |
| `hospital_L2_cold_chain_robot_low` | Medication tote cold-chain limit, robot low battery, assign porter. |
| `hospital_L3_replan_priority_medication` | Higher-priority medication job appears mid-plan; preempt/delay lower-priority linen. |
| `hospital_L4_blood_product_conflict_escalate` | Blood product with tight cold-chain window and no safe route; escalate to human. |

Run examples:

```bash
BASE_URL=http://192.168.1.158:8020/v1 MODEL_NAME=nemotron MODEL_LABEL=nemotron_hospital MODE=thinking_off \
bash agent_arena/run_host_pydantic_hospital_probe.sh

BASE_URL=http://192.168.1.158:8012/v1 MODEL_NAME=stock-llama MODEL_LABEL=stock_llama_hospital MODE=stock \
bash agent_arena/run_host_pydantic_hospital_probe.sh

BASE_URL=http://192.168.1.158:8013/v1 MODEL_NAME=ministral-q4 MODEL_LABEL=ministral_hospital MODE=stock \
bash agent_arena/run_host_pydantic_hospital_probe.sh
```

First smoke-test observation:

- Nemotron with the guarded BFCL parser initially emitted a bare tool plan such as `<TOOLCALL>[get_pending_jobs, ...]</TOOLCALL>`. The shim now recovers the first no-argument tool from that pattern so the agent loop can begin.
- On the first sample-delivery scenario, Nemotron executed `get_pending_jobs` and stopped early. Ministral gathered several facts but also stopped before assignment/status actions. This makes the demo useful as a harder multi-step workflow benchmark rather than only a happy-path showcase.
