#!/usr/bin/env python3
"""Run and compare the BFCL holdout used to check adapter overfitting."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BFCL_ROOT = PROJECT_ROOT / "agent_arena_results" / "bfcl_v4_100"
SELECTION = BFCL_ROOT / "holdout_signal90_20260627_selection" / "holdout_case_ids.json"
EVK_HOST = os.environ.get("EVK_HOST", "192.168.1.158")


@dataclass(frozen=True)
class ModelRun:
    key: str
    label: str
    final_summary: Path
    holdout_run_name: str
    holdout_model_id: str
    endpoint: str
    endpoint_model: str
    native_bfcl_functions: bool = False


MODELS = [
    ModelRun(
        key="nemotron",
        label="Nemotron Nano 8B W4A16 guarded v7",
        final_summary=BFCL_ROOT
        / "exact_rerun_20260626_current_best_nemotron_guarded_v7_signal100"
        / "summary_exact-current-nemotron-guarded-v7-signal100.json",
        holdout_run_name="holdout_signal80_nonweb_20260627_nemotron_guarded_v7",
        holdout_model_id="holdout-nemotron-guarded-v7-nonweb",
        endpoint=f"http://{EVK_HOST}:8020/v1",
        endpoint_model="nemotron",
        native_bfcl_functions=True,
    ),
    ModelRun(
        key="stock_llama",
        label="Stock Llama 3.1 8B W4A16 qcom_tool",
        final_summary=BFCL_ROOT
        / "exact_rerun_20260626_current_best_stock_llama_qcom_signal100"
        / "summary_exact-current-stock-llama-qcom-signal100.json",
        holdout_run_name="holdout_signal80_nonweb_20260627_stock_llama_qcom",
        holdout_model_id="holdout-stock-llama-qcom-nonweb",
        endpoint=f"http://{EVK_HOST}:8012/v1",
        endpoint_model="stock-llama",
    ),
    ModelRun(
        key="ministral",
        label="Ministral 3.3B Q4 mistral_tool",
        final_summary=BFCL_ROOT
        / "exact_rerun_20260626_current_best_ministral_mistral_tool_signal100"
        / "summary_exact-current-ministral-mistral-tool-signal100.json",
        holdout_run_name="holdout_signal80_nonweb_20260627_ministral_mistral_tool",
        holdout_model_id="holdout-ministral-mistral-tool-nonweb",
        endpoint=f"http://{EVK_HOST}:8013/v1",
        endpoint_model="ministral-q4",
    ),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def holdout_ids() -> list[str]:
    """Return the context-safe holdout set used for EVK comparison.

    Web-search BFCL cases are intentionally excluded from execution, not just
    from score aggregation. They trigger long search/fetch loops in this local
    EVK harness and are not useful for comparing the edge-hosted tool adapters.
    """
    return [
        case_id
        for case_id in load_json(SELECTION)["case_ids"]
        if not str(case_id).startswith("web_search")
    ]


def summary_path(model: ModelRun) -> Path:
    return (
        BFCL_ROOT
        / model.holdout_run_name
        / f"summary_{model.holdout_model_id.replace('/', '_')}.json"
    )


def run_model(model: ModelRun) -> int:
    cmd = [
        sys.executable,
        "agent_arena/bfcl_v4_subset_runner.py",
        "run",
        "--run-name",
        model.holdout_run_name,
        "--model-id",
        model.holdout_model_id,
        "--display-name",
        f"EVK {model.label} holdout",
        "--endpoint",
        model.endpoint,
        "--endpoint-model",
        model.endpoint_model,
        "--temperature",
        "0.001",
        "--num-threads",
        "1",
        "--overwrite",
        "--case-id",
        ",".join(holdout_ids()),
    ]
    if model.native_bfcl_functions:
        cmd.insert(-2, "--native-bfcl-functions")
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def category_rows(summary: dict) -> dict[str, dict]:
    return {row["category"]: row for row in summary.get("categories", [])}


def non_web_score(summary: dict) -> tuple[float, int, float]:
    rows = [
        row
        for row in summary.get("categories", [])
        if not row["category"].startswith("web_search")
    ]
    correct = sum(float(row["correct"]) for row in rows)
    total = sum(int(row["total"]) for row in rows)
    return correct, total, correct / total if total else 0.0


def count_inference_errors(run_name: str, model_id: str) -> int:
    result_root = BFCL_ROOT / run_name / "result" / model_id.replace("/", "_")
    failure_markers = (
        "Error during inference: Connection error",
        "Failure to initialize model",
        "Create From Binary FAILED",
        "Failed to create the dialog",
    )
    count = 0
    for path in result_root.rglob("BFCL_v4_*_result.json"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if any(marker in line for marker in failure_markers):
                count += 1
    return count


def compare(min_nonweb_accuracy: float, max_abs_drop: float) -> tuple[dict, bool]:
    rows = []
    valid = True
    for model in MODELS:
        final = load_json(model.final_summary)
        holdout_file = summary_path(model)
        if not holdout_file.exists():
            rows.append({"model": model.label, "valid": False, "reason": "missing_holdout"})
            valid = False
            continue
        holdout = load_json(holdout_file)
        errors = count_inference_errors(model.holdout_run_name, model.holdout_model_id)
        final_correct, final_total, final_acc = non_web_score(final)
        holdout_correct, holdout_total, holdout_acc = non_web_score(holdout)
        drop = final_acc - holdout_acc
        row = {
            "model": model.label,
            "valid": errors == 0,
            "inference_connection_errors": errors,
            "final_nonweb": [final_correct, final_total, final_acc],
            "holdout_nonweb": [holdout_correct, holdout_total, holdout_acc],
            "absolute_drop": drop,
            "similar": errors == 0
            and holdout_acc >= min_nonweb_accuracy
            and drop <= max_abs_drop,
        }
        rows.append(row)
        valid = valid and row["valid"] and row["similar"]
    return {"rows": rows}, valid


def write_report(report: dict, passes: bool, path: Path) -> None:
    lines = [
        "# BFCL Holdout Validation",
        "",
        f"Status: {'passed similarity check' if passes else 'not yet valid/similar'}",
        "",
        "| Model | Final non-web | Holdout 80 non-web | Drop | Connection errors | Similar |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        if not row.get("valid") and "final_nonweb" not in row:
            lines.append(f"| {row['model']} | n/a | n/a | n/a | n/a | no: {row['reason']} |")
            continue
        fc, ft, fa = row["final_nonweb"]
        hc, ht, ha = row["holdout_nonweb"]
        lines.append(
            f"| {row['model']} | {fc:.0f}/{ft} ({fa * 100:.1f}%) | "
            f"{hc:.0f}/{ht} ({ha * 100:.1f}%) | "
            f"{row['absolute_drop'] * 100:.1f} pp | "
            f"{row['inference_connection_errors']} | "
            f"{'yes' if row['similar'] else 'no'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "compare"])
    parser.add_argument("--model", choices=[m.key for m in MODELS], action="append")
    parser.add_argument("--min-nonweb-accuracy", type=float, default=0.60)
    parser.add_argument("--max-abs-drop", type=float, default=0.15)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "resources" / "bfcl_holdout_validation.md",
    )
    args = parser.parse_args()

    selected = [m for m in MODELS if not args.model or m.key in args.model]
    if args.command == "run":
        for model in selected:
            rc = run_model(model)
            if rc != 0:
                raise SystemExit(rc)
    else:
        report, passes = compare(args.min_nonweb_accuracy, args.max_abs_drop)
        write_report(report, passes, args.report)
        print(json.dumps({"passes": passes, **report}, indent=2))
        print(f"report: {args.report}")
        raise SystemExit(0 if passes else 2)


if __name__ == "__main__":
    main()
