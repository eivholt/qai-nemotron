"""Generate, sandbox, validate, and reuse a manufacturing analysis program."""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator

from manufacturing_agent.runtime import ManufacturingRuntime, SCENARIOS


SYSTEM_PROMPT = """You write concise, reliable Python for a sandboxed factory analysis runtime.
Return exactly one complete program in one python code block and no other text.
Use plant_api as the only way to read or update plant state. Compute decisions from
returned data and policy. Never hard-code batch IDs, machine IDs, limits, or outcomes.
"""

PROGRAM_TASK = """Write one reusable Python program that evaluates the active production batch.

Available plant_api functions:
- get_production_context() -> dict with batch_id, product, machine_ids, and status
- get_policy() -> dict with max_temperature_c, max_vibration_rms, max_defect_rate
- get_quality_counts() -> dict with inspected and defects
- get_sensor_summary(machine_id) -> dict with machine_id, max_temperature_c,
  and vibration_rms
- schedule_inspection(machine_id, reason)
- release_batch(batch_id, evidence)
- hold_batch(batch_id, reason); an inspection must be scheduled first
- quarantine_batch(batch_id, reason)
- notify_supervisor(message)
- APIError is raised automatically when an API operation is rejected

Required policy:
1. Read context, policy, quality counts, and every machine's sensor summary before
   taking an action.
2. defect_rate = defects / inspected.
3. Read max_temperature_c and vibration_rms from each machine summary.
4. If defect_rate is greater than max_defect_rate, quarantine the batch.
5. Otherwise, if any machine summary exceeds either matching policy limit,
   schedule one inspection for the first anomalous machine and hold the batch.
6. Otherwise release the batch.
7. Notify the supervisor exactly once after the disposition succeeds.

Program structure matters: first collect every machine summary without taking
any action. Store each complete summary dict. After the read loop, choose
exactly one disposition for the batch. When holding, inspect the first stored
summary whose max_temperature_c or vibration_rms exceeds its matching policy
limit. Notify the supervisor after every successful disposition, including
hold and quarantine.

Keep the program under 100 lines. Import only plant_api. Do not access
files, sockets, subprocesses, environment variables, or absolute paths directly.
"""

PLANT_API_MODULE = """import json
import os
import urllib.parse
import urllib.request

_BASE_URL = os.environ["PLANT_API_URL"]


class APIError(RuntimeError):
    pass


def _request(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise APIError(result)
    return result


def get_production_context():
    return _request("GET", "/context")["context"]


def get_policy():
    return _request("GET", "/policy")["policy"]


def get_quality_counts():
    return _request("GET", "/quality")


def get_sensor_summary(machine_id):
    query = urllib.parse.urlencode({"machine_id": machine_id})
    return _request("GET", "/sensor-summary?" + query)


def schedule_inspection(machine_id, reason):
    return _request(
        "POST",
        "/inspection",
        {"machine_id": machine_id, "reason": reason},
    )


def release_batch(batch_id, evidence):
    return _request(
        "POST",
        "/release",
        {"batch_id": batch_id, "evidence": evidence},
    )


def hold_batch(batch_id, reason):
    return _request(
        "POST",
        "/hold",
        {"batch_id": batch_id, "reason": reason},
    )


def quarantine_batch(batch_id, reason):
    return _request(
        "POST",
        "/quarantine",
        {"batch_id": batch_id, "reason": reason},
    )


def notify_supervisor(message):
    return _request("POST", "/notify", {"message": message})
"""

SITE_CUSTOMIZE = """import os
import socket
import subprocess

_ALLOWED_PORT = int(os.environ["PLANT_API_PORT"])
_original_connect = socket.socket.connect


def _guarded_connect(sock, address):
    if (
        sock.family == socket.AF_INET
        and isinstance(address, tuple)
        and address[0] == "127.0.0.1"
        and address[1] == _ALLOWED_PORT
    ):
        return _original_connect(sock, address)
    raise RuntimeError("socket disabled inside manufacturing sandbox")


def _blocked(*args, **kwargs):
    raise RuntimeError("subprocess disabled inside manufacturing sandbox")


socket.socket.connect = _guarded_connect
subprocess.Popen = _blocked
os.system = _blocked
"""


def extract_python_code(text: str) -> str:
    match = re.search(r"```python\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    generic = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    return generic.group(1).strip() if generic else text.strip()


def limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))


class PlantApiHandler(BaseHTTPRequestHandler):
    runtime: ManufacturingRuntime

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _send(self, value: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _unknown(self) -> None:
        result = {"ok": False, "error": "unknown_endpoint", "path": self.path}
        self.runtime.calls.append(
            {"name": "unknown_endpoint", "arguments": {"path": self.path}, "result": result}
        )
        self._send(result, 404)

    def do_GET(self) -> None:
        runtime = self.runtime
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/context":
            result = runtime.call(
                "get_production_context",
                {},
                runtime.get_production_context,
            )
        elif parsed.path == "/policy":
            result = runtime.call("get_policy", {}, runtime.get_policy)
        elif parsed.path == "/quality":
            result = runtime.call(
                "get_quality_counts",
                {},
                runtime.get_quality_counts,
            )
        elif parsed.path == "/sensor-summary":
            query = urllib.parse.parse_qs(parsed.query)
            machine_id = query.get("machine_id", [""])[0]
            result = runtime.call(
                "get_sensor_summary",
                {"machine_id": machine_id},
                lambda: runtime.get_sensor_summary(machine_id),
            )
        else:
            self._unknown()
            return
        self._send(result)

    def do_POST(self) -> None:
        runtime = self.runtime
        body = self._body()
        if self.path == "/inspection":
            machine_id = body.get("machine_id", "")
            reason = body.get("reason", "")
            result = runtime.call(
                "schedule_inspection",
                {"machine_id": machine_id, "reason": reason},
                lambda: runtime.schedule_inspection(machine_id, reason),
            )
        elif self.path == "/release":
            batch_id = body.get("batch_id", "")
            evidence = body.get("evidence", "")
            result = runtime.call(
                "release_batch",
                {"batch_id": batch_id, "evidence": evidence},
                lambda: runtime.release_batch(batch_id, evidence),
            )
        elif self.path == "/hold":
            batch_id = body.get("batch_id", "")
            reason = body.get("reason", "")
            result = runtime.call(
                "hold_batch",
                {"batch_id": batch_id, "reason": reason},
                lambda: runtime.hold_batch(batch_id, reason),
            )
        elif self.path == "/quarantine":
            batch_id = body.get("batch_id", "")
            reason = body.get("reason", "")
            result = runtime.call(
                "quarantine_batch",
                {"batch_id": batch_id, "reason": reason},
                lambda: runtime.quarantine_batch(batch_id, reason),
            )
        elif self.path == "/notify":
            message = body.get("message", "")
            result = runtime.call(
                "notify_supervisor",
                {"message": message},
                lambda: runtime.notify_supervisor(message),
            )
        else:
            self._unknown()
            return
        self._send(result)


@contextmanager
def api_server(runtime: ManufacturingRuntime) -> Iterator[tuple[str, int]]:
    handler = type("BoundPlantApiHandler", (PlantApiHandler,), {"runtime": runtime})
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


def run_program(code: str, scenario_name: str) -> dict[str, Any]:
    runtime = ManufacturingRuntime.from_scenario(scenario_name, verbose=False)
    with tempfile.TemporaryDirectory(
        prefix=f"manufacturing_codegen__{scenario_name}__"
    ) as temp:
        root = Path(temp)
        (root / "plant_api.py").write_text(PLANT_API_MODULE)
        (root / "candidate.py").write_text(code)
        site_dir = root / "_site"
        site_dir.mkdir()
        (site_dir / "sitecustomize.py").write_text(SITE_CUSTOMIZE)
        with api_server(runtime) as (base_url, port):
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
                    "stdout": proc.stdout[-1200:],
                    "stderr": proc.stderr[-1200:],
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
    runtime_text = f"{execution.get('error', '')}\n{execution.get('stderr', '')}"
    if passed:
        failure = "passed"
    elif "disabled inside manufacturing sandbox" in runtime_text:
        failure = "sandbox_violation"
    elif "SyntaxError" in runtime_text:
        failure = "code_syntax_error"
    elif not execution["ok"]:
        failure = "runtime_error"
    else:
        failure = "wrong_state_or_trace"
    return {
        "passed": passed,
        "failure": failure,
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


def build_prompt(
    cached_code: str | None,
    feedback: dict[str, Any] | None,
) -> str:
    prompt = PROGRAM_TASK
    if cached_code:
        prompt += (
            "\nA previously validated program for this policy follows. Reuse its "
            "general algorithm, but do not assume scenario values.\n"
            "```python\n"
            f"{cached_code}\n"
            "```\n"
        )
    if feedback:
        violations = feedback.get("violations", [])
        prompt += (
            "\nThe previous program failed strict validation. Return a complete "
            "corrected replacement.\n"
            f"Failure: {feedback['failure']}\n"
            f"Violated invariants: {json.dumps(violations)}\n"
            f"Ledger and state: {json.dumps(feedback['summary'], sort_keys=True)[:3000]}\n"
            f"Runtime stderr: {feedback.get('stderr', '')[-1000:]}\n"
        )
    return prompt


def describe_violations(summary: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    call_names = summary.get("call_names", [])
    expected_calls = summary.get("expected_calls", [])
    if sorted(call_names) != sorted(expected_calls):
        violations.append(
            "The API call multiset was not exact; make every required read, one "
            "disposition, and one final notification, with no extra calls."
        )
    if not summary.get("trace_correct", False):
        violations.append(
            "Finish all reads before the first write; inspection precedes hold; "
            "notification is the final call."
        )
    if not summary.get("state_correct", False):
        violations.append("The selected disposition or inspection machine was wrong.")
    if not summary.get("notified", False):
        violations.append("notify_supervisor must run after every disposition path.")
    if summary.get("tool_errors"):
        violations.append("One or more API operations were rejected; do not retry them.")
    return violations


def run_case(
    client: ModelClient,
    scenario_name: str,
    cache_path: Path,
    reuse_policy: str,
    repair_retries: int,
) -> dict[str, Any]:
    cached_code = cache_path.read_text() if cache_path.exists() else None
    if reuse_policy == "execute_first" and cached_code:
        cached_result = run_program(cached_code, scenario_name)
        if cached_result["passed"]:
            return {
                "scenario": scenario_name,
                "passed": True,
                "cache_hit": True,
                "code": cached_code,
                "attempts": [],
                **cached_result,
            }

    prompt_cache = cached_code if reuse_policy in {"prompt", "execute_first"} else None
    feedback = None
    attempts: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    code = ""
    raw_text = ""
    result: dict[str, Any] = {
        "passed": False,
        "failure": "not_run",
        "execution": {},
        "summary": {},
    }
    for attempt in range(1, repair_retries + 2):
        generation = client.generate(build_prompt(prompt_cache, feedback))
        raw_text = generation["text"]
        code = extract_python_code(raw_text)
        result = run_program(code, scenario_name)
        generations.append(
            {
                "attempt": attempt,
                "elapsed_s": generation["elapsed_s"],
                "output_chars": len(raw_text),
                "raw_text": raw_text,
                "code": code,
            }
        )
        attempts.append(
            {
                "attempt": attempt,
                "passed": result["passed"],
                "failure": result["failure"],
                "execution": result["execution"],
                "summary": result["summary"],
            }
        )
        if result["passed"]:
            break
        feedback = {
            "failure": result["failure"],
            "summary": result["summary"],
            "stderr": result["execution"].get("stderr", ""),
            "violations": describe_violations(result["summary"]),
        }

    if result["passed"] and reuse_policy != "none":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(code)
    return {
        "scenario": scenario_name,
        "passed": result["passed"],
        "cache_hit": False,
        "failure": result["failure"],
        "attempts": attempts,
        "generations": generations,
        "raw_text": raw_text,
        "code": code,
        "execution": result["execution"],
        "summary": result["summary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("MANUFACTURING_MODEL_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("MANUFACTURING_API_KEY", "local"))
    parser.add_argument("--model", default=os.getenv("MANUFACTURING_MODEL", "ministral3-3b-q4"))
    parser.add_argument("--label", default="ministral3-3b-q4")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument(
        "--reuse-policy",
        choices=["none", "prompt", "execute_first"],
        default="execute_first",
    )
    parser.add_argument("--repair-retries", type=int, default=1)
    parser.add_argument("--model-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path.home() / "manufacturing_agent_results",
    )
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.out_root.expanduser() / (
        f"{timestamp}__{args.label}__{args.reuse_policy}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        args.cache_dir.expanduser()
        if args.cache_dir
        else args.out_root.expanduser() / "cache" / args.label
    )
    cache_path = cache_dir / "manufacturing_policy_v5.py"
    client = ModelClient(args.base_url, args.api_key, args.model, args.model_timeout_s)

    results = []
    for name in names:
        print(f"\n=== {name}: {SCENARIOS[name]['title']} ===", flush=True)
        try:
            result = run_case(
                client,
                name,
                cache_path,
                args.reuse_policy,
                args.repair_retries,
            )
        except Exception as exc:
            result = {
                "scenario": name,
                "passed": False,
                "cache_hit": False,
                "failure": "model_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(
            json.dumps(
                {
                    "scenario": name,
                    "passed": result["passed"],
                    "cache_hit": result.get("cache_hit", False),
                    "failure": result.get("failure", "passed"),
                    "calls": result.get("summary", {}).get("call_names", []),
                    "attempts": len(result.get("attempts", [])),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if result.get("code") and not result.get("cache_hit"):
            print("VALIDATED PROGRAM" if result["passed"] else "LAST PROGRAM")
            print(result["code"])

    output = {
        "label": args.label,
        "model": args.model,
        "reuse_policy": args.reuse_policy,
        "passed": sum(bool(item["passed"]) for item in results),
        "total": len(results),
        "cache_path": str(cache_path),
        "system_prompt": SYSTEM_PROMPT,
        "program_task": PROGRAM_TASK,
        "model_timeout_s": args.model_timeout_s,
        "repair_retries": args.repair_retries,
        "results": results,
    }
    (result_dir / "results.json").write_text(json.dumps(output, indent=2))
    print(f"\nRESULT_DIR={result_dir}")
    print(
        "SUMMARY "
        + json.dumps(
            {
                "passed": output["passed"],
                "total": output["total"],
                "reuse_policy": output["reuse_policy"],
            },
            sort_keys=True,
        )
    )
    return 0 if output["passed"] == output["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
