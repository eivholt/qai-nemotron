"""Generate, validate, sandbox, promote, and reuse manufacturing programs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator

from manufacturing_agent.app import SITE_CUSTOMIZE, extract_python_code
from manufacturing_agent.codegen_tasks import TASKS, TaskRuntime, TaskSpec, build_api_module


SYSTEM_PROMPT = """You write concise, reusable Python for a sandboxed manufacturing runtime.
Return exactly one complete program in one python code block and no other text.
Use plant_api as the only way to read or change plant state. Derive every decision
from returned data and policy. Never hard-code identifiers, thresholds, or outcomes.
"""

SANDBOX_CONTRACT_VERSION = "manufacturing-sandbox-v2"
VALIDATOR_CONTRACT_VERSION = "strict-mock-validator-v2"
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "input"}


def task_prompt(spec: TaskSpec) -> str:
    api_lines = "\n".join(
        f"- {item.name}({', '.join(item.parameters)}): {item.description}"
        for item in spec.functions
    )
    return f"""Write one reusable Python program for this task:

{spec.title}

Available plant_api functions:
{api_lines}
- APIError is raised automatically when an API operation is rejected.

Policy:
{spec.instructions}

Use every return value as the exact dict shape documented above, including its
named list key. Read every required observation before the first state-changing
API call. Do not catch APIError, repeat calls, retry rejected operations, or
perform extra actions. Put the one final notification after the complete
if/elif/else decision so every successful outcome reaches it exactly once.
Keep the program under 120 lines. Import only plant_api. Do not access files,
sockets, subprocesses, environment variables, or absolute paths directly.
Call the program's entry function so the task actually runs.
"""


def contract_hash(spec: TaskSpec) -> str:
    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "task_prompt": task_prompt(spec),
        "api_module": build_api_module(spec),
        "task_version": spec.contract_version,
        "sandbox_version": SANDBOX_CONTRACT_VERSION,
        "validator_version": VALIDATOR_CONTRACT_VERSION,
        "scenarios": spec.scenarios,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_source(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} at line {exc.lineno}"]

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "plant_api":
                    errors.append(f"import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module != "plant_api":
                errors.append(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                errors.append(f"call not allowed: {node.func.id}")
    return sorted(set(errors))


def limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))


class MockApiHandler(BaseHTTPRequestHandler):
    runtime: TaskRuntime

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/call":
            self._send({"ok": False, "error": "unknown_endpoint"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send({"ok": False, "error": "invalid_json"}, 400)
            return
        name = payload.get("name", "")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            self._send({"ok": False, "error": "invalid_call"}, 400)
            return
        self._send(self.runtime.invoke(name, arguments))

    def _send(self, value: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def mock_api_server(runtime: TaskRuntime) -> Iterator[tuple[str, int]]:
    handler = type("BoundMockApiHandler", (MockApiHandler,), {"runtime": runtime})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_program(
    code: str,
    task_name: str,
    scenario_name: str,
    *,
    trace_api: bool = False,
) -> dict[str, Any]:
    spec = TASKS[task_name]
    source_errors = validate_source(code)
    if source_errors:
        return {
            "passed": False,
            "failure": "source_rejected",
            "source_errors": source_errors,
            "execution": {"ok": False, "stderr": "\n".join(source_errors)},
            "summary": {},
        }

    runtime = TaskRuntime.create(task_name, scenario_name, trace=trace_api)
    with tempfile.TemporaryDirectory(
        prefix=f"manufacturing_suite__{task_name}__{scenario_name}__"
    ) as temp:
        root = Path(temp)
        (root / "plant_api.py").write_text(build_api_module(spec))
        (root / "candidate.py").write_text(code)
        site_dir = root / "_site"
        site_dir.mkdir()
        (site_dir / "sitecustomize.py").write_text(SITE_CUSTOMIZE)

        with mock_api_server(runtime) as (base_url, port):
            env = {
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(site_dir),
                "PYTHONNOUSERSITE": "1",
                "HOME": str(root),
                "TMPDIR": str(root),
                "PLANT_API_URL": base_url,
                "PLANT_API_PORT": str(port),
            }
            try:
                proc = subprocess.run(
                    [sys.executable, str(root / "candidate.py")],
                    cwd=root,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=8,
                    preexec_fn=limit_child if os.name == "posix" else None,
                )
                execution = {
                    "ok": proc.returncode == 0,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-1600:],
                    "stderr": proc.stderr[-1600:],
                }
            except subprocess.TimeoutExpired as exc:
                execution = {
                    "ok": False,
                    "error": f"timeout after {exc.timeout}s",
                    "stdout": "",
                    "stderr": "",
                }

    summary = runtime.summary()
    passed = bool(execution["ok"] and summary["passed"])
    diagnostic = f"{execution.get('error', '')}\n{execution.get('stderr', '')}"
    if passed:
        failure = "passed"
    elif "disabled inside manufacturing sandbox" in diagnostic:
        failure = "sandbox_violation"
    elif not execution["ok"]:
        failure = "runtime_error"
    else:
        failure = "wrong_state_or_trace"
    return {
        "passed": passed,
        "failure": failure,
        "source_errors": [],
        "execution": execution,
        "summary": summary,
    }


@dataclass
class ModelClient:
    base_url: str
    api_key: str
    model: str
    timeout_s: float

    def generate(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1024,
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"model HTTP {exc.code}: {detail[-1000:]}") from exc
        content = data["choices"][0]["message"].get("content")
        if not isinstance(content, str):
            raise RuntimeError("model returned non-text content")
        return {"text": content, "elapsed_s": round(time.monotonic() - started, 3)}


def describe_validation(validations: dict[str, dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    for scenario_name, result in validations.items():
        if result["passed"]:
            continue
        summary = result.get("summary", {})
        messages.append(
            f"{scenario_name}: {result['failure']}; "
            f"calls={summary.get('call_names', [])}; "
            f"expected_calls={summary.get('expected_calls', [])}; "
            f"actual={summary.get('actual')}; "
            f"expected={summary.get('expected')}; "
            f"errors={summary.get('tool_errors', [])}; "
            f"stderr={result.get('execution', {}).get('stderr', '')[-600:]}"
        )
    return messages


def build_generation_prompt(
    spec: TaskSpec,
    previous_code: str | None,
    validations: dict[str, dict[str, Any]] | None,
) -> str:
    prompt = task_prompt(spec)
    if previous_code:
        prompt += (
            "\nA previous program failed fresh validation. Return a complete corrected "
            "replacement, not a patch.\n<previous_program>\n"
            + previous_code
            + "\n</previous_program>\n"
        )
    if validations:
        prompt += "\nValidation failures:\n- " + "\n- ".join(
            describe_validation(validations)
        )
    return prompt


def validate_for_promotion(
    code: str,
    spec: TaskSpec,
    *,
    trace_api: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        name: run_program(code, spec.name, name, trace_api=trace_api)
        for name in spec.scenarios
    }


def load_cache(
    cache_path: Path,
    expected_hash: str,
) -> tuple[dict[str, Any] | None, str]:
    if not cache_path.exists():
        return None, "not_found"
    try:
        record = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, "invalid_json"
    if record.get("contract_hash") != expected_hash:
        return None, "contract_mismatch"
    code = record.get("code")
    if not isinstance(code, str) or not code.strip():
        return None, "missing_code"
    return record, "contract_match"


def save_cache(
    cache_path: Path,
    spec: TaskSpec,
    code: str,
    validations: dict[str, dict[str, Any]],
    model: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "task": spec.name,
        "contract_version": spec.contract_version,
        "contract_hash": contract_hash(spec),
        "sandbox_version": SANDBOX_CONTRACT_VERSION,
        "validator_version": VALIDATOR_CONTRACT_VERSION,
        "model": model,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "promotion_scenarios": list(spec.scenarios),
        "promotion_results": {
            name: {
                "passed": result["passed"],
                "summary": result["summary"],
            }
            for name, result in validations.items()
        },
        "code": code,
    }
    cache_path.write_text(json.dumps(record, indent=2))
    cache_path.with_suffix(".py").write_text(code)


def run_case(
    client: ModelClient,
    spec: TaskSpec,
    scenario_name: str,
    cache_path: Path,
    repair_retries: int,
    *,
    trace_api: bool = False,
) -> dict[str, Any]:
    expected_hash = contract_hash(spec)
    cached, cache_status = load_cache(cache_path, expected_hash)
    cache_check: dict[str, Any] = {
        "path": str(cache_path),
        "status": cache_status,
        "contract_hash": expected_hash,
    }
    previous_code = None
    previous_validations = None

    if cached:
        cached_result = run_program(
            cached["code"],
            spec.name,
            scenario_name,
            trace_api=trace_api,
        )
        cache_check["fresh_validation"] = cached_result
        if cached_result["passed"]:
            return {
                "task": spec.name,
                "scenario": scenario_name,
                "passed": True,
                "cache_hit": True,
                "model_called": False,
                "cache_check": cache_check,
                "code": cached["code"],
                "result": cached_result,
                "attempts": [],
            }
        cache_check["status"] = "fresh_validation_failed"
        previous_code = cached["code"]
        previous_validations = {scenario_name: cached_result}

    attempts: list[dict[str, Any]] = []
    final_validations: dict[str, dict[str, Any]] = {}
    final_code = previous_code or ""
    for attempt in range(1, repair_retries + 2):
        generation = client.generate(
            build_generation_prompt(spec, previous_code, previous_validations)
        )
        code = extract_python_code(generation["text"])
        validations = validate_for_promotion(code, spec, trace_api=trace_api)
        promotion_passed = all(item["passed"] for item in validations.values())
        attempts.append(
            {
                "attempt": attempt,
                "elapsed_s": generation["elapsed_s"],
                "raw_text": generation["text"],
                "code": code,
                "promotion_passed": promotion_passed,
                "validations": validations,
            }
        )
        final_code = code
        final_validations = validations
        if promotion_passed:
            save_cache(cache_path, spec, code, validations, client.model)
            break
        previous_code = code
        previous_validations = validations

    scenario_result = final_validations.get(
        scenario_name,
        {
            "passed": False,
            "failure": "not_validated",
            "execution": {},
            "summary": {},
        },
    )
    promotion_passed = bool(
        final_validations
        and all(item["passed"] for item in final_validations.values())
    )
    return {
        "task": spec.name,
        "scenario": scenario_name,
        "passed": bool(promotion_passed and scenario_result["passed"]),
        "cache_hit": False,
        "model_called": True,
        "cache_check": cache_check,
        "code": final_code,
        "result": scenario_result,
        "promotion": final_validations,
        "attempts": attempts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--scenario", default="all")
    parser.add_argument(
        "--base-url",
        default=os.getenv("MANUFACTURING_MODEL_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("MANUFACTURING_API_KEY", "local"))
    parser.add_argument("--model", default=os.getenv("MANUFACTURING_MODEL", "ministral3-3b-q4"))
    parser.add_argument("--label", default="manufacturing-codegen-suite")
    parser.add_argument("--repair-retries", type=int, default=2)
    parser.add_argument("--model-timeout-s", type=float, default=300.0)
    parser.add_argument("--trace-api", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path.home() / "manufacturing_codegen_results",
    )
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_names = list(TASKS) if args.task == "all" else [args.task]
    if args.task == "all" and args.scenario != "all":
        raise SystemExit("--scenario can be specific only when --task is specific")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.out_root.expanduser() / f"{timestamp}__{args.label}"
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (
        args.cache_dir.expanduser()
        if args.cache_dir
        else args.out_root.expanduser() / "cache" / args.label
    )
    client = ModelClient(args.base_url, args.api_key, args.model, args.model_timeout_s)
    results: list[dict[str, Any]] = []

    for task_name in task_names:
        spec = TASKS[task_name]
        if args.scenario == "all":
            scenario_names = list(spec.scenarios)
        elif args.scenario in spec.scenarios:
            scenario_names = [args.scenario]
        else:
            raise SystemExit(f"Unknown scenario {args.scenario!r} for {task_name}")
        cache_path = cache_root / f"{spec.name}__{spec.contract_version}.json"

        print(f"\n##### {spec.title} #####", flush=True)
        for scenario_name in scenario_names:
            print(f"\n=== {task_name} / {scenario_name} ===", flush=True)
            try:
                result = run_case(
                    client,
                    spec,
                    scenario_name,
                    cache_path,
                    args.repair_retries,
                    trace_api=args.trace_api,
                )
            except Exception as exc:
                result = {
                    "task": task_name,
                    "scenario": scenario_name,
                    "passed": False,
                    "cache_hit": False,
                    "model_called": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            summary = result.get("result", {}).get("summary", {})
            print(
                json.dumps(
                    {
                        "task": task_name,
                        "scenario": scenario_name,
                        "passed": result["passed"],
                        "cache_hit": result.get("cache_hit", False),
                        "model_called": result.get("model_called", False),
                        "cache_status": result.get("cache_check", {}).get("status"),
                        "calls": summary.get("call_names", []),
                        "attempts": len(result.get("attempts", [])),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if result.get("model_called") and result.get("code"):
                print("PROMOTED PROGRAM" if result["passed"] else "LAST PROGRAM")
                print(result["code"], flush=True)

    output = {
        "label": args.label,
        "model": args.model,
        "system_prompt": SYSTEM_PROMPT,
        "sandbox_contract_version": SANDBOX_CONTRACT_VERSION,
        "validator_contract_version": VALIDATOR_CONTRACT_VERSION,
        "trace_api": args.trace_api,
        "passed": sum(bool(item["passed"]) for item in results),
        "total": len(results),
        "tasks": {
            name: {
                "title": TASKS[name].title,
                "contract_version": TASKS[name].contract_version,
                "contract_hash": contract_hash(TASKS[name]),
                "prompt": task_prompt(TASKS[name]),
            }
            for name in task_names
        },
        "results": results,
    }
    result_path = result_dir / "results.json"
    result_path.write_text(json.dumps(output, indent=2))
    print(f"\nRESULT_PATH={result_path}")
    print("SUMMARY " + json.dumps({"passed": output["passed"], "total": output["total"]}))
    return 0 if output["passed"] == output["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
