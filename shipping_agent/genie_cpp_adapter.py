"""Adapt a persistent C++ Genie service to the model's native tool protocol."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agent_arena.openai_genie_server import (
    mistral_tool_parse,
    openai_message,
    render_mistral_tool_prompt,
)


def build_upstream_payload(
    payload: dict[str, Any],
    upstream_model: str,
) -> tuple[dict[str, Any], str]:
    """Render once and bypass the C++ service's generic chat/tool template."""
    prompt = render_mistral_tool_prompt(payload, "stock")
    upstream: dict[str, Any] = {
        "model": upstream_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    for key in ("max_tokens", "temperature", "top_p", "seed"):
        if key in payload:
            upstream[key] = payload[key]
    return upstream, prompt


def adapt_upstream_response(
    upstream: dict[str, Any],
    public_model: str,
) -> tuple[dict[str, Any], str]:
    choices = upstream.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("C++ Genie response has no choices")
    message = choices[0].get("message", {})
    raw = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(raw, str):
        raise ValueError("C++ Genie response content is not text")

    parsed, parse_status = mistral_tool_parse(raw)
    adapted_message, finish_reason = openai_message(parsed, raw)
    result = {
        "id": upstream.get("id", f"chatcmpl-cpp-{time.time_ns()}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": public_model,
        "choices": [
            {
                "index": 0,
                "message": adapted_message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": upstream.get(
            "usage",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ),
    }
    return result, parse_status


class AdapterServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], args: argparse.Namespace) -> None:
        super().__init__(address, Handler)
        self.args = args
        self.inference_lock = threading.Lock()
        self.counter = 0
        self.work_dir = args.work_dir.expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def post_upstream(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.args.upstream_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"C++ Genie returned HTTP {exc.code}: {detail}") from exc


class Handler(BaseHTTPRequestHandler):
    server: AdapterServer

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.args.verbose:
            super().log_message(format, *args)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") in {"", "/health"}:
            self.send_json(200, {"status": "ok", "backend": "cpp-genie"})
            return
        if self.path.rstrip("/") == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.args.model_name,
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") not in {
            "/chat/completions",
            "/v1/chat/completions",
        }:
            self.send_json(404, {"error": "not_found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if payload.get("stream"):
                raise ValueError("Streaming is not supported by this tutorial adapter")
            upstream_payload, prompt = build_upstream_payload(
                payload,
                self.server.args.upstream_model,
            )
            started = time.monotonic()
            with self.server.inference_lock:
                upstream = self.server.post_upstream(upstream_payload)
            result, parse_status = adapt_upstream_response(
                upstream,
                self.server.args.model_name,
            )
            elapsed_s = time.monotonic() - started
            self.server.counter += 1
            record = {
                "request": self.server.counter,
                "elapsed_s": round(elapsed_s, 3),
                "parse_status": parse_status,
                "prompt": prompt,
                "raw_response": upstream,
                "adapted_response": result,
            }
            log_path = self.server.work_dir / f"request_{self.server.counter:04d}.json"
            log_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            self.send_json(200, result)
        except Exception as exc:
            self.send_json(
                502,
                {"error": {"type": type(exc).__name__, "message": str(exc)}},
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8911/v1")
    parser.add_argument("--upstream-model", default="ministral_q4_genie_export")
    parser.add_argument("--model-name", default="ministral3-3b-q4")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.home() / "shipping_agent_results" / "cpp_adapter",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = AdapterServer((args.host, args.port), args)
    print(
        f"Shipping Genie adapter: http://{args.host}:{args.port}/v1 -> "
        f"{args.upstream_url}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
