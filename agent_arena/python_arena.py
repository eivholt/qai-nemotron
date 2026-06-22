#!/usr/bin/env python3
"""Agent arena where generated Python programs are executed in a small sandbox."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_arena.model_client import GenieClient, extract_python_code, write_summary


def big_metrics() -> str:
    rows = ["request_id,latency_ms,status"]
    rows.extend(f"R-{i},12,ok" for i in range(700))
    rows.append("R-8842,93,timeout")
    rows.extend(f"R-X{i},14,ok" for i in range(700))
    return "\n".join(rows) + "\n"


PYTHON_CASES: list[dict[str, Any]] = [
    {
        "id": "py_00_add_params_seed",
        "difficulty": 1,
        "cache_key": "add_params_json",
        "goal": (
            "Read params.json. It contains numbers a and b. Write result.json as "
            '{"sum": number}. Do not hard-code the numbers.'
        ),
        "files": {"params.json": json.dumps({"a": 7, "b": 5})},
        "expected": {"sum": 12},
    },
    {
        "id": "py_00_add_params_reuse",
        "difficulty": 1,
        "cache_key": "add_params_json",
        "goal": (
            "This is the same program shape as the earlier params task, but the "
            "numbers changed. Read params.json and write result.json as "
            '{"sum": number}. Reuse the previous approach if available.'
        ),
        "files": {"params.json": json.dumps({"a": 13, "b": 29})},
        "expected": {"sum": 42},
    },
    {
        "id": "py_01_sum_csv_west",
        "difficulty": 1,
        "cache_key": "sum_csv_by_region",
        "goal": (
            "Read sales.csv and params.json. Sum the amount column for rows whose "
            "region equals params.target_region. Write result.json as "
            '{"region": "...", "total": number}. Do not hard-code the region.'
        ),
        "files": {
            "sales.csv": "region,amount\nwest,10\neast,7\nwest,32\nnorth,5\n",
            "params.json": json.dumps({"target_region": "west"}),
        },
        "expected": {"region": "west", "total": 42},
        "hints": [
            "params.json uses key target_region.",
            "CSV values are strings; convert amount with int(row['amount']) or float(row['amount']).",
        ],
    },
    {
        "id": "py_02_sum_csv_east_reuse",
        "difficulty": 2,
        "cache_key": "sum_csv_by_region",
        "goal": (
            "This is the same program shape as the earlier sales task, but the "
            "parameters changed. Read sales.csv and params.json. Write result.json "
            'as {"region": "...", "total": number}. Reuse the previous approach if available.'
        ),
        "files": {
            "sales.csv": "region,amount\nwest,10\neast,7\neast,11\nnorth,5\n",
            "params.json": json.dumps({"target_region": "east"}),
        },
        "expected": {"region": "east", "total": 18},
        "hints": [
            "params.json uses key target_region.",
            "CSV values are strings; convert amount with int(row['amount']) or float(row['amount']).",
        ],
    },
    {
        "id": "py_03_log_count",
        "difficulty": 2,
        "goal": (
            "Read events.log. Count lines containing level=ERROR and write "
            'result.json as {"error_count": number}.'
        ),
        "files": {
            "events.log": (
                "ts=1 level=INFO msg=start\n"
                "ts=2 level=ERROR msg=bad-input\n"
                "ts=3 level=WARN msg=retry\n"
                "ts=4 level=ERROR msg=failed\n"
            )
        },
        "expected": {"error_count": 2},
        "hints": ["Count matching lines once; do not count matching words."],
    },
    {
        "id": "py_04_context_breaking_large_csv",
        "difficulty": 4,
        "goal": (
            "metrics.csv is large. Find the row with request_id R-8842 and write "
            'result.json as {"request_id": "R-8842", "latency_ms": number, "status": "..."} .'
        ),
        "files": {"metrics.csv": big_metrics()},
        "expected": {"request_id": "R-8842", "latency_ms": 93, "status": "timeout"},
        "hints": ["Use csv.DictReader and compare row['request_id'] to R-8842."],
    },
    {
        "id": "py_05_fixture_http_transform",
        "difficulty": 3,
        "goal": (
            "The sandbox has no network. Treat http_response.json as the result of "
            "a fixture HTTP GET. Read it and write result.json as "
            '{"health": "...", "temp_f": number}, converting temp_c to Fahrenheit.'
        ),
        "files": {"http_response.json": json.dumps({"health": "degraded", "temp_c": 74})},
        "expected": {"health": "degraded", "temp_f": 165.2},
        "hints": [
            "http_response.json has keys health and temp_c.",
            "Use health = data['health']; temp_f = data['temp_c'] * 9 / 5 + 32.",
            "Do not assume weather API keys such as main or temp.",
        ],
    },
]


SITE_CUSTOMIZE = r'''
import os
import socket
import subprocess

def _blocked(*args, **kwargs):
    raise RuntimeError("disabled inside agent arena sandbox")

socket.socket = _blocked
subprocess.Popen = _blocked
os.system = _blocked
'''


def limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))


def build_prompt(
    case: dict[str, Any],
    cached_code: str | None,
    feedback: dict[str, Any] | None = None,
) -> str:
    cache_block = ""
    if cached_code:
        cache_block = (
            "\nA previous successful program for the same task family is available below. "
            "Reuse the approach when appropriate, but adapt to the current files and params.\n"
            "```python\n"
            f"{cached_code}\n"
            "```\n"
        )
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\nThe previous attempt failed. Fix the program and return a full replacement.\n"
            f"Failure kind: {feedback.get('failure_kind')}\n"
            f"Validator missing: {json.dumps(feedback.get('missing', []))}\n"
            f"Expected result.json shape: {json.dumps(feedback.get('expected', {}))}\n"
            f"Actual result.json from previous attempt: {json.dumps(feedback.get('actual', None))}\n"
            f"Runtime error: {feedback.get('runtime_error', '')[-800:]}\n"
        )
    file_list = sorted(case.get("files", {}).keys())
    hint_lines = "\n".join(f"- {hint}" for hint in case.get("hints", []))
    hint_block = f"Extra hints:\n{hint_lines}\n\n" if hint_lines else ""
    return (
        "Write exactly one complete Python program in one ```python code block.\n"
        "Return no explanation outside the code block. Keep the code short, direct, and under 80 lines.\n"
        "The program will be executed as-is in a temporary working directory.\n"
        "Available input files: "
        f"{', '.join(file_list)}.\n"
        "Requirements:\n"
        "- Use only the Python standard library.\n"
        "- Read input files from the current working directory.\n"
        "- Write the final answer to result.json in the current working directory.\n"
        "- Do not use network access, subprocesses, shell commands, or absolute paths.\n\n"
        "File-format hints:\n"
        "- For .json files, use json.load.\n"
        "- For .csv files, use csv.DictReader. Do not use json.loads on CSV rows.\n"
        "- For .log files, iterate over text lines.\n\n"
        f"{hint_block}"
        f"TASK:\n{case['goal']}\n"
        f"{cache_block}"
        f"{feedback_block}"
    )


def run_program(code: str, case: dict[str, Any], work_root: Path) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    for rel, text in case.get("files", {}).items():
        path = work_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    site_dir = work_root / "_site"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(SITE_CUSTOMIZE)
    script = work_root / "candidate.py"
    script.write_text(code)
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(site_dir),
        "PYTHONNOUSERSITE": "1",
        "HOME": str(work_root),
        "TMPDIR": str(work_root),
    }
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=work_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=6,
            preexec_fn=limit_child if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"timeout after {exc.timeout}s"}
    result_path = work_root / "result.json"
    result: Any = None
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"invalid result.json: {exc}"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-1000:],
        "stderr": proc.stderr[-1000:],
        "result": result,
    }


def score_result(case: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    actual = execution.get("result")
    if not execution.get("ok"):
        return {"passed": False, "score": 0.0, "missing": [execution.get("error") or execution.get("stderr", "")]}
    if not isinstance(actual, dict):
        return {"passed": False, "score": 0.0, "missing": ["result.json object"]}
    passed = 0
    missing: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            ok = isinstance(actual_value, (int, float)) and abs(actual_value - expected_value) < 0.001
        else:
            ok = actual_value == expected_value
        if ok:
            passed += 1
        else:
            missing.append(f"{key}={expected_value!r}")
    score = passed / len(expected)
    return {"passed": score == 1.0, "score": round(score, 3), "missing": missing}


def classify_python_failure(
    score: dict[str, Any],
    execution: dict[str, Any],
    code: str,
    model_timed_out: bool = False,
) -> str:
    if score["passed"]:
        return "passed"
    if model_timed_out:
        return "model_timeout"
    if not code.strip():
        return "no_code"
    runtime_text = f"{execution.get('error', '')}\n{execution.get('stderr', '')}"
    if "disabled inside agent arena sandbox" in runtime_text:
        return "sandbox_violation"
    if "SyntaxError" in runtime_text:
        return "code_syntax_error"
    if execution.get("returncode") not in (0, None):
        return "runtime_error"
    if execution.get("result") is None:
        return "missing_result"
    return "wrong_result"


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    compact_a = re.sub(r"\s+", " ", a).strip()
    compact_b = re.sub(r"\s+", " ", b).strip()
    return round(difflib.SequenceMatcher(None, compact_a, compact_b).ratio(), 3)


def run_case(
    client: GenieClient,
    result_dir: Path,
    cache_dir: Path,
    case: dict[str, Any],
    reuse_policy: str,
    repair_retries: int,
) -> dict[str, Any]:
    cache_key = case.get("cache_key")
    cached_path = cache_dir / f"{cache_key}.py" if cache_key else None
    cached_code = cached_path.read_text() if cached_path and cached_path.exists() else None

    if reuse_policy == "execute_first" and cached_code:
        with tempfile.TemporaryDirectory(prefix=f"{case['id']}__cache__") as temp:
            execution = run_program(cached_code, case, Path(temp))
        score = score_result(case, execution)
        if score["passed"]:
            record = {
                "model": client.model,
                "mode": client.mode,
                "case_id": case["id"],
                "difficulty": case["difficulty"],
                "cache_hit": True,
                "reuse_similarity": 1.0,
                "score": score,
                "execution": execution,
                "failure_kind": "passed",
                "notes": "cache_hit",
            }
            (result_dir / f"{client.model}__{client.mode}__{case['id']}.json").write_text(json.dumps(record, indent=2))
            return record

    prompt_cache = cached_code if reuse_policy in {"prompt", "execute_first"} else None
    attempts: list[dict[str, Any]] = []
    feedback: dict[str, Any] | None = None
    gen = None
    code = ""
    execution: dict[str, Any] = {}
    score = {"passed": False, "score": 0.0, "missing": ["not run"]}
    failure_kind = "not_run"
    model_timed_out = False

    for attempt in range(1, repair_retries + 2):
        try:
            gen = client.generate(
                build_prompt(case, prompt_cache, feedback),
                f"{case['id']}__try{attempt}",
            )
            model_timed_out = False
        except subprocess.TimeoutExpired as exc:
            code = ""
            execution = {"ok": False, "error": f"model generation timed out after {exc.timeout}s"}
            score = {
                "passed": False,
                "score": 0.0,
                "missing": [execution["error"]],
            }
            failure_kind = classify_python_failure(score, execution, code, model_timed_out=True)
            attempts.append(
                {
                    "attempt": attempt,
                    "failure_kind": failure_kind,
                    "score": score,
                    "execution": execution,
                }
            )
            feedback = {
                "failure_kind": failure_kind,
                "missing": score["missing"],
                "runtime_error": execution["error"],
            }
            continue

        code = extract_python_code(gen.answer or gen.raw_text)
        with tempfile.TemporaryDirectory(prefix=f"{case['id']}__try{attempt}__") as temp:
            execution = run_program(code, case, Path(temp))
        score = score_result(case, execution)
        failure_kind = classify_python_failure(score, execution, code, model_timed_out)
        attempts.append(
            {
                "attempt": attempt,
                "failure_kind": failure_kind,
                "score": score,
                "execution": execution,
                "code_chars": len(code),
            }
        )
        if score["passed"]:
            break
        feedback = {
            "failure_kind": failure_kind,
            "missing": score.get("missing", []),
            "expected": case.get("expected", {}),
            "actual": execution.get("result"),
            "runtime_error": execution.get("error") or execution.get("stderr", ""),
        }

    reuse_score = similarity(cached_code or "", code)
    if score["passed"] and cached_path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path.write_text(code)
    record = {
        "model": client.model,
        "mode": client.mode,
        "case_id": case["id"],
        "difficulty": case["difficulty"],
        "cache_hit": False,
        "reuse_policy": reuse_policy,
        "reuse_similarity": reuse_score,
        "raw_answer": gen.raw_text if gen else "",
        "code": code,
        "score": score,
        "failure_kind": failure_kind,
        "execution": execution,
        "attempts": attempts,
        "paths": gen.paths if gen else {},
        "notes": (
            f"attempts={len(attempts)}"
            + (f",reuse_similarity={reuse_score:.3f}" if cached_code else "")
        ),
    }
    (result_dir / f"{client.model}__{client.mode}__{case['id']}.json").write_text(json.dumps(record, indent=2))
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--model", default="nemotron")
    parser.add_argument("--mode", choices=["stock", "thinking_off", "thinking_on"], default="thinking_off")
    parser.add_argument("--case-ids")
    parser.add_argument("--timeout-s", type=int, default=150)
    parser.add_argument("--repair-retries", type=int, default=1)
    parser.add_argument("--reuse-policy", choices=["none", "prompt", "execute_first"], default="prompt")
    parser.add_argument("--out-root", type=Path, default=Path.home() / "agent_arena_results")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Persistent directory for successful generated programs. Defaults below --out-root.",
    )
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for case in PYTHON_CASES:
            cache = f"\tcache_key={case['cache_key']}" if case.get("cache_key") else ""
            print(f"{case['id']}\tdifficulty={case['difficulty']}{cache}")
        return 0
    if args.bundle is None:
        raise SystemExit("--bundle is required unless --list-cases is used")
    wanted = {item.strip() for item in args.case_ids.split(",")} if args.case_ids else None
    cases = [case for case in PYTHON_CASES if wanted is None or case["id"] in wanted]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.out_root.expanduser() / f"{timestamp}__python_arena__{args.model}__{args.mode}"
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        args.cache_dir.expanduser()
        if args.cache_dir
        else args.out_root.expanduser() / "python_code_cache" / f"{args.model}__{args.mode}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = GenieClient(args.bundle, args.model, args.mode, result_dir, timeout_s=args.timeout_s)
    results = [
        run_case(client, result_dir, cache_dir, case, args.reuse_policy, args.repair_retries)
        for case in cases
    ]
    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(result_dir / "summary.md", "Python Agent Arena", results)
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
