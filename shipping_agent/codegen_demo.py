"""Generate and execute Python that coordinates the mock shipping API."""

from __future__ import annotations

import argparse
import copy
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
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator

from shipping_agent.runtime import SCENARIOS, ShippingRuntime


SYSTEM_PROMPT = """You write short, reliable Python programs for a sandboxed logistics runtime.
Return exactly one complete Python program in one python code block and no other text.
Use the documented scenario_api functions as the only way to inspect or change logistics
state. Base decisions on returned data. Never hard-code shipment, carrier, or dock IDs.
"""

TASK_PROMPT = """Write a complete Python program that safely plans the pending shipment.

Available module:
- get_pending_shipment() -> shipment record with status, load, handling, and deadline
- get_shipping_options() -> a dict containing usable_carriers,
  temporarily_blocked_carriers, excluded_carriers, usable_docks, and excluded_docks
- each carrier is a dict; pass carrier["carrier_id"] to an action
- each dock is a dict; pass dock["dock_id"] to an action
- schedule_shipment(carrier_id, dock_id) -> schedule a usable combination
- hold_shipment(reason) -> hold only for a temporarily blocked compatible carrier
- escalate_shipment(reason) -> escalate when no compatible carrier is usable or recovering
- notify_dispatch() -> call only after a disposition succeeds
- APIError -> raised automatically when any API operation is rejected

Import these functions from scenario_api. First inspect the shipment and options.
Choose exactly one safe disposition from the returned facts, then notify dispatch
exactly once. Do not catch APIError unless you can safely recover without repeating
successful work. Do not access files, sockets, subprocesses, environment variables,
or absolute paths directly.
"""

SCENARIO_API_MODULE = """import json
import os
import urllib.request

_BASE_URL = os.environ["SHIPPING_API_URL"]


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


def get_pending_shipment():
    return _request("GET", "/shipment")["shipment"]


def get_shipping_options():
    return _request("GET", "/options")


def schedule_shipment(carrier_id, dock_id):
    return _request(
        "POST",
        "/schedule",
        {"carrier_id": carrier_id, "dock_id": dock_id},
    )


def hold_shipment(reason):
    return _request("POST", "/hold", {"reason": reason})


def escalate_shipment(reason):
    return _request("POST", "/escalate", {"reason": reason})


def notify_dispatch():
    return _request("POST", "/notify", {})
"""

SITE_CUSTOMIZE = """import os
import socket
import subprocess

_ALLOWED_HOST = "127.0.0.1"
_ALLOWED_PORT = int(os.environ["SHIPPING_API_PORT"])
_original_connect = socket.socket.connect


def _guarded_connect(sock, address):
    if (
        sock.family == socket.AF_INET
        and isinstance(address, tuple)
        and address[0] == _ALLOWED_HOST
        and address[1] == _ALLOWED_PORT
    ):
        return _original_connect(sock, address)
    raise RuntimeError("socket disabled inside shipping code sandbox")


def _blocked(*args, **kwargs):
    raise RuntimeError("subprocess disabled inside shipping code sandbox")


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


class ShippingApiHandler(BaseHTTPRequestHandler):
    runtime: ShippingRuntime

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def _json_body(self) -> dict[str, Any]:
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
        if self.path == "/shipment":
            result = runtime.call(
                "get_pending_shipment",
                {},
                runtime.get_pending_shipment,
            )
        elif self.path == "/options":
            shipment_id = runtime.shipment["shipment_id"]
            result = runtime.call(
                "get_shipping_options",
                {},
                lambda: runtime.get_shipping_options(shipment_id),
            )
        else:
            self._unknown()
            return
        self._send(result)

    def do_POST(self) -> None:
        runtime = self.runtime
        body = self._json_body()
        shipment_id = runtime.shipment["shipment_id"]
        if self.path == "/schedule":
            arguments = {
                "carrier_id": body.get("carrier_id"),
                "dock_id": body.get("dock_id"),
            }
            result = runtime.call(
                "schedule_shipment",
                arguments,
                lambda: runtime.schedule_shipment(
                    shipment_id,
                    body.get("carrier_id"),
                    body.get("dock_id"),
                ),
            )
        elif self.path == "/hold":
            arguments = {"reason": body.get("reason", "")}
            result = runtime.call(
                "hold_shipment",
                arguments,
                lambda: runtime.hold_shipment(shipment_id, body.get("reason", "")),
            )
        elif self.path == "/escalate":
            arguments = {"reason": body.get("reason", "")}
            result = runtime.call(
                "escalate_shipment",
                arguments,
                lambda: runtime.escalate_shipment(
                    shipment_id,
                    body.get("reason", ""),
                ),
            )
        elif self.path == "/notify":
            result = runtime.call(
                "notify_dispatch",
                {},
                lambda: runtime.notify_dispatch(shipment_id),
            )
        else:
            self._unknown()
            return
        self._send(result)


@contextmanager
def api_server(runtime: ShippingRuntime) -> Iterator[tuple[str, int]]:
    handler = type(
        "BoundShippingApiHandler",
        (ShippingApiHandler,),
        {"runtime": runtime},
    )
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
    runtime = ShippingRuntime.from_scenario(scenario_name, verbose=False)
    with tempfile.TemporaryDirectory(prefix=f"shipping_codegen__{scenario_name}__") as temp:
        root = Path(temp)
        (root / "scenario_api.py").write_text(SCENARIO_API_MODULE)
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
                "SHIPPING_API_URL": base_url,
                "SHIPPING_API_PORT": str(port),
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
    failure = "passed"
    runtime_text = f"{execution.get('error', '')}\n{execution.get('stderr', '')}"
    if not passed:
        if "disabled inside shipping code sandbox" in runtime_text:
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
class OpenAITextClient:
    base_url: str
    api_key: str
    model: str
    timeout_s: float

    def generate(self, prompt: str) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + "/chat/completions"
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
            url,
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
        message = data["choices"][0]["message"]
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"model returned non-text content: {message!r}")
        return {
            "text": content,
            "elapsed_s": round(time.monotonic() - started, 3),
        }


def build_prompt(
    cached_code: str | None,
    feedback: dict[str, Any] | None,
) -> str:
    prompt = TASK_PROMPT
    if cached_code:
        prompt += (
            "\nA previously validated program for this task family follows. "
            "Reuse its general approach, but do not assume scenario values.\n"
            "```python\n"
            f"{cached_code}\n"
            "```\n"
        )
    if feedback:
        prompt += (
            "\nThe previous attempt failed. Return a complete corrected program.\n"
            f"Failure: {feedback['failure']}\n"
            "Strict result:\n"
            f"{json.dumps(feedback['summary'], sort_keys=True)[:2400]}\n"
            f"Runtime stderr:\n{feedback.get('stderr', '')[-800:]}\n"
        )
    return prompt


def run_case(
    client: OpenAITextClient,
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
                "attempts": [],
                "code": cached_code,
                **cached_result,
            }

    prompt_cache = cached_code if reuse_policy in {"prompt", "execute_first"} else None
    attempts: list[dict[str, Any]] = []
    code = ""
    raw_text = ""
    result: dict[str, Any] = {
        "passed": False,
        "failure": "not_run",
        "execution": {},
        "summary": {},
    }
    feedback = None
    generations: list[dict[str, Any]] = []
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
            }
        )
        attempts.append(
            {
                "attempt": attempt,
                "failure": result["failure"],
                "passed": result["passed"],
                "summary": result["summary"],
                "execution": result["execution"],
            }
        )
        if result["passed"]:
            break
        feedback = {
            "failure": result["failure"],
            "summary": result["summary"],
            "stderr": result["execution"].get("stderr", ""),
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
        default=os.getenv("SHIPPING_AGENT_BASE_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("SHIPPING_AGENT_API_KEY", "local"))
    parser.add_argument("--model", default=os.getenv("SHIPPING_AGENT_MODEL", "ministral3-3b-q4"))
    parser.add_argument("--label", default="ministral3-3b-q4")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--reuse-policy", choices=["none", "prompt", "execute_first"], default="none")
    parser.add_argument("--repair-retries", type=int, default=1)
    parser.add_argument("--model-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path.home() / "shipping_agent_results" / "codegen",
    )
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.out_root.expanduser() / f"{timestamp}__{args.label}__{args.reuse_policy}"
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        args.cache_dir.expanduser()
        if args.cache_dir
        else args.out_root.expanduser() / "cache" / args.label
    )
    cache_path = cache_dir / "shipping_coordination_v1.py"
    client = OpenAITextClient(
        args.base_url,
        args.api_key,
        args.model,
        args.model_timeout_s,
    )

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
        print(json.dumps(
            {
                "scenario": name,
                "passed": result["passed"],
                "cache_hit": result.get("cache_hit", False),
                "failure": result.get("failure", "passed"),
                "call_names": result.get("summary", {}).get("call_names", []),
                "attempts": len(result.get("attempts", [])),
            },
            sort_keys=True,
        ))

    output = {
        "label": args.label,
        "model": args.model,
        "reuse_policy": args.reuse_policy,
        "passed": sum(bool(item["passed"]) for item in results),
        "total": len(results),
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
