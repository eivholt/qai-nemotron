#!/usr/bin/env python3
"""OpenAI-compatible host shim backed by Hugging Face Transformers.

This is a host-side reference server for comparing the agent arena against
source models on the RTX machine. It reuses the prompt renderers and tolerant
tool parsers from ``openai_genie_server.py`` so benchmark differences are less
likely to come from client-side parsing drift.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_arena.model_client import build_chat_prompt, split_reasoning
from agent_arena.openai_genie_server import (
    NEMOTRON_BFCL_OFFICIAL_CLEAN_ARGS_GUIDANCE,
    NEMOTRON_BFCL_OFFICIAL_EXACT_GUIDANCE,
    NEMOTRON_BFCL_OFFICIAL_SELECTIVE_GUIDANCE,
    NEMOTRON_BFCL_OFFICIAL_VALUE_GUIDANCE,
    enhance_tool_arguments_for_payload,
    keep_only_available_tool_calls,
    llama3_json_parse,
    merge_tool_calls,
    mistral_tool_parse,
    mode_system_text,
    nemotron_native_parse,
    normalize_tool_call_names_for_payload,
    openai_message,
    prune_or_rewrite_action_mismatch_tool_calls,
    qcom_tool_parse,
    qwen3_native_parse,
    recover_final_tool_call_for_payload,
    reject_relevance_unsupported_tool_calls,
    reject_semantic_mismatch_tool_calls,
    render_llama3_json_prompt,
    render_mistral_tool_prompt,
    render_qwen3_native_prompt,
    render_nemotron_bfcl_modelcard_prompt,
    render_nemotron_bfcl_official_prompt,
    render_nemotron_bfcl_official_user_prompt,
    render_nemotron_bfcl_schema_guided_prompt,
    render_nemotron_bfcl_schema_prompt,
    render_nemotron_bfcl_user_prompt,
    render_nemotron_native_prompt,
    render_qcom_tool_prompt,
    render_request_prompt,
    repair_tool_arguments_for_payload,
    strict_parse,
    tolerant_parse,
)


REQUEST_LOG: list[dict[str, Any]] = []


def assistant_prefill(mode: str) -> str:
    return "<think>\n</think>\n" if mode == "thinking_off" else ""


def eos_token_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    for token in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted(set(ids))


def clean_generation(text: str) -> str:
    for marker in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
        text = text.replace(marker, "")
    return text.strip()


def render_prompt(payload: dict[str, Any], args: argparse.Namespace) -> str:
    parser = args.parser
    if parser == "llama3_json":
        return render_llama3_json_prompt(payload, args.mode, args.tool_output_mode)
    if parser == "mistral_tool":
        return render_mistral_tool_prompt(payload, args.mode)
    if parser == "qwen3_native":
        return render_qwen3_native_prompt(payload, args.mode)
    if parser == "nemotron_native":
        return render_nemotron_native_prompt(payload, args.mode)
    if parser == "nemotron_bfcl_schema":
        return render_nemotron_bfcl_schema_prompt(payload, args.mode)
    if parser == "nemotron_bfcl_schema_guided":
        return render_nemotron_bfcl_schema_guided_prompt(payload, args.mode)
    if parser == "nemotron_bfcl_official":
        return render_nemotron_bfcl_official_prompt(payload, args.mode)
    if parser == "nemotron_bfcl_official_exact":
        return render_nemotron_bfcl_official_prompt(
            payload, args.mode, extra_guidance=NEMOTRON_BFCL_OFFICIAL_EXACT_GUIDANCE
        )
    if parser == "nemotron_bfcl_official_clean_args":
        return render_nemotron_bfcl_official_prompt(
            payload, args.mode, extra_guidance=NEMOTRON_BFCL_OFFICIAL_CLEAN_ARGS_GUIDANCE
        )
    if parser in {
        "nemotron_bfcl_official_strict_names",
        "nemotron_bfcl_official_strict_schema",
        "nemotron_bfcl_official_strict_schema_enhanced",
        "nemotron_bfcl_official_strict_schema_enhanced_guarded",
        "nemotron_bfcl_official_strict_schema_guarded",
    }:
        return render_nemotron_bfcl_official_prompt(payload, args.mode)
    if parser == "nemotron_bfcl_official_user_strict_schema":
        return render_nemotron_bfcl_official_user_prompt(payload, args.mode)
    if parser in {
        "nemotron_bfcl_official_user_selective_schema",
        "nemotron_bfcl_official_user_selective_schema_guarded",
        "nemotron_bfcl_official_user_selective_schema_supported",
    }:
        return render_nemotron_bfcl_official_user_prompt(
            payload, args.mode, extra_guidance=NEMOTRON_BFCL_OFFICIAL_SELECTIVE_GUIDANCE
        )
    if parser == "nemotron_bfcl_modelcard_supported":
        return render_nemotron_bfcl_modelcard_prompt(payload, args.mode)
    if parser == "nemotron_bfcl_official_strict_schema_exact":
        return render_nemotron_bfcl_official_prompt(
            payload, args.mode, extra_guidance=NEMOTRON_BFCL_OFFICIAL_EXACT_GUIDANCE
        )
    if parser == "nemotron_bfcl_official_strict_schema_values":
        return render_nemotron_bfcl_official_prompt(
            payload, args.mode, extra_guidance=NEMOTRON_BFCL_OFFICIAL_VALUE_GUIDANCE
        )
    if parser == "nemotron_bfcl_user":
        return render_nemotron_bfcl_user_prompt(payload, args.mode)
    if parser == "qcom_tool":
        return render_qcom_tool_prompt(payload, args.mode)

    user_text = render_request_prompt(payload, parser)
    return build_chat_prompt(mode_system_text(args.mode), user_text, assistant_prefill(args.mode))


def parse_answer(raw: str, payload: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    split = split_reasoning(raw)
    parse_input = split["final_answer"] or raw
    parser = args.parser
    if parser == "strict":
        parsed, status = strict_parse(parse_input)
    elif parser == "llama3_json":
        parsed, status = llama3_json_parse(parse_input)
    elif parser == "mistral_tool":
        parsed, status = mistral_tool_parse(parse_input)
    elif parser == "qwen3_native":
        parsed, status = qwen3_native_parse(parse_input)
    elif parser in {
        "nemotron_bfcl_official",
        "nemotron_bfcl_official_exact",
        "nemotron_bfcl_official_clean_args",
        "nemotron_bfcl_official_strict_names",
        "nemotron_bfcl_official_strict_schema",
        "nemotron_bfcl_official_strict_schema_enhanced",
        "nemotron_bfcl_official_strict_schema_enhanced_guarded",
        "nemotron_bfcl_official_strict_schema_guarded",
        "nemotron_bfcl_official_user_strict_schema",
        "nemotron_bfcl_official_user_selective_schema",
        "nemotron_bfcl_official_user_selective_schema_guarded",
        "nemotron_bfcl_official_user_selective_schema_supported",
        "nemotron_bfcl_official_strict_schema_exact",
        "nemotron_bfcl_official_strict_schema_values",
        "nemotron_bfcl_modelcard_supported",
    }:
        parsed, status = nemotron_native_parse(parse_input, outside_tag_fallback=False)
    elif parser in {"nemotron_native", "nemotron_bfcl_schema", "nemotron_bfcl_schema_guided", "nemotron_bfcl_user"}:
        parsed, status = nemotron_native_parse(parse_input)
    elif parser == "qcom_tool":
        parsed, status = qcom_tool_parse(parse_input)
    else:
        parsed, status = tolerant_parse(parse_input)

    parsed = normalize_tool_call_names_for_payload(parsed, payload)
    if parser in {
        "nemotron_bfcl_official_strict_schema_enhanced",
        "nemotron_bfcl_official_strict_schema_enhanced_guarded",
        "nemotron_bfcl_official_user_selective_schema_supported",
        "nemotron_bfcl_modelcard_supported",
        "qcom_tool",
    }:
        parsed, recovered = recover_final_tool_call_for_payload(parsed, payload)
        if recovered:
            status = f"{status}_final_recovered"
    if parser in {
        "nemotron_bfcl_official_strict_schema",
        "nemotron_bfcl_official_strict_schema_enhanced",
        "nemotron_bfcl_official_strict_schema_enhanced_guarded",
        "nemotron_bfcl_official_strict_schema_guarded",
        "nemotron_bfcl_official_user_strict_schema",
        "nemotron_bfcl_official_user_selective_schema",
        "nemotron_bfcl_official_user_selective_schema_guarded",
        "nemotron_bfcl_official_user_selective_schema_supported",
        "nemotron_bfcl_official_strict_schema_exact",
        "nemotron_bfcl_official_strict_schema_values",
        "nemotron_bfcl_modelcard_supported",
        "qcom_tool",
    }:
        parsed, repaired = repair_tool_arguments_for_payload(parsed, payload)
        if repaired:
            status = f"{status}_schema_repaired"
    if parser in {
        "nemotron_bfcl_official_strict_names",
        "nemotron_bfcl_official_strict_schema",
        "nemotron_bfcl_official_strict_schema_enhanced",
        "nemotron_bfcl_official_strict_schema_enhanced_guarded",
        "nemotron_bfcl_official_strict_schema_guarded",
        "nemotron_bfcl_official_user_strict_schema",
        "nemotron_bfcl_official_user_selective_schema",
        "nemotron_bfcl_official_user_selective_schema_guarded",
        "nemotron_bfcl_official_user_selective_schema_supported",
        "nemotron_bfcl_official_strict_schema_exact",
        "nemotron_bfcl_official_strict_schema_values",
        "nemotron_bfcl_modelcard_supported",
        "qcom_tool",
    }:
        before = parsed
        parsed = keep_only_available_tool_calls(parsed, payload)
        if before and not parsed:
            status = f"{status}_unknown_tool_rejected"
    if parser in {"nemotron_bfcl_official_strict_schema_enhanced_guarded", "nemotron_bfcl_modelcard_supported", "qcom_tool"}:
        parsed, action_guard = prune_or_rewrite_action_mismatch_tool_calls(parsed, payload)
        if action_guard and action_guard.get("decision") == "rewrite_or_prune":
            status = f"{status}_action_pruned"
        elif action_guard and action_guard.get("decision") == "reject":
            status = f"{status}_action_rejected"
    if parser in {
        "nemotron_bfcl_official_strict_schema_enhanced",
        "nemotron_bfcl_official_strict_schema_enhanced_guarded",
        "nemotron_bfcl_official_user_selective_schema_supported",
        "nemotron_bfcl_modelcard_supported",
        "qcom_tool",
    }:
        parsed, enhanced = enhance_tool_arguments_for_payload(parsed, payload)
        if enhanced:
            status = f"{status}_supported_enhanced"
    if parser in {"nemotron_bfcl_official_strict_schema_enhanced_guarded", "nemotron_bfcl_modelcard_supported", "qcom_tool"}:
        parsed, semantic_guard = reject_semantic_mismatch_tool_calls(parsed, payload)
        if semantic_guard and semantic_guard.get("decision") == "prune":
            status = f"{status}_semantic_pruned"
        elif semantic_guard and semantic_guard.get("decision") == "reject":
            status = f"{status}_semantic_rejected"
        parsed, relevance_guard = reject_relevance_unsupported_tool_calls(parsed, payload)
        if relevance_guard and relevance_guard.get("decision") == "reject":
            status = f"{status}_relevance_rejected"
    return parsed, status


def apply_multi_tool_policy(parsed: dict[str, Any] | None, policy: str) -> dict[str, Any] | None:
    if policy == "all" or not parsed or parsed.get("type") != "tool":
        return parsed
    calls = parsed.get("tool_calls")
    if not isinstance(calls, list) or len(calls) <= 1:
        return parsed
    first = calls[0]
    return {"type": "tool", "id": first.get("id", "call_1"), "name": first["name"], "arguments": first["arguments"]}


def parsed_summary(parsed: dict[str, Any] | None) -> dict[str, Any]:
    if not parsed:
        return {}
    summary: dict[str, Any] = {"type": parsed.get("type")}
    calls = parsed.get("tool_calls")
    if isinstance(calls, list):
        summary["tool_calls"] = [{"name": c.get("name"), "arguments": c.get("arguments", {})} for c in calls]
    if parsed.get("type") == "tool":
        summary["name"] = parsed.get("name")
        summary["arguments"] = parsed.get("arguments", {})
    return summary


class TransformersOpenAIServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], args: argparse.Namespace) -> None:
        super().__init__(server_address, Handler)
        self.args = args
        self.counter = 0
        self.work_dir = args.work_dir.expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading tokenizer {args.model_id}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"Loading model {args.model_id}", flush=True)
        started = time.monotonic()
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=getattr(torch, args.torch_dtype),
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
            device_map=args.device_map if args.device_map != "none" else None,
        )
        if args.device_map == "none":
            self.model.to(args.device)
        self.model.eval()
        self.load_s = time.monotonic() - started
        self.eos_ids = eos_token_ids(self.tokenizer)
        print(f"Loaded in {self.load_s:.1f}s", flush=True)

    def completion_payload(
        self,
        request_id: str,
        message: dict[str, Any],
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.args.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": max(1, prompt_tokens),
                "completion_tokens": max(1, completion_tokens),
                "total_tokens": max(2, prompt_tokens + completion_tokens),
            },
            "arena": {
                "parser": self.args.parser,
                "parse_status": record["parse_status"],
                "parsed_ok": record["parsed_ok"],
                "elapsed_s": record["elapsed_s"],
                "tokens_per_s": record["tokens_per_s"],
            },
        }

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.counter += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        stem = f"{self.args.model_name}__req{self.counter:05d}__{uuid.uuid4().hex[:8]}"
        prompt = render_prompt(payload, self.args)
        prompt_path = self.work_dir / f"{stem}.prompt.txt"
        answer_path = self.work_dir / f"{stem}.answer.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.args.device_map == "none":
            inputs = {key: value.to(self.args.device) for key, value in inputs.items()}
        max_tokens = int(payload.get("max_tokens") or self.args.max_new_tokens)
        temperature = float(payload.get("temperature", self.args.temperature))
        do_sample = temperature > 0.01
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "eos_token_id": self.eos_ids,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 0.01)
            gen_kwargs["top_p"] = float(payload.get("top_p") or self.args.top_p)

        if self.args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        if self.args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_s = time.monotonic() - started

        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = int(output.shape[1] - prompt_tokens)
        raw = self.tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=False)
        raw = clean_generation(raw)
        answer_path.write_text(raw, encoding="utf-8")
        parsed, status = parse_answer(raw, payload, self.args)
        parsed = apply_multi_tool_policy(parsed, self.args.multi_tool_policy)
        message, finish_reason = openai_message(parsed, raw)
        tokens_per_s = completion_tokens / elapsed_s if elapsed_s else 0.0
        record = {
            "request_index": self.counter,
            "request_id": request_id,
            "model": self.args.model_name,
            "model_id": self.args.model_id,
            "mode": self.args.mode,
            "parser": self.args.parser,
            "elapsed_s": round(elapsed_s, 3),
            "tokens_per_s": round(tokens_per_s, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "parse_status": status,
            "parsed_ok": bool(parsed),
            "parsed_action": parsed_summary(parsed),
            "finish_reason": finish_reason,
            "raw_answer": raw,
            "paths": {"prompt": str(prompt_path), "answer": str(answer_path)},
        }
        REQUEST_LOG.append(record)
        return self.completion_payload(request_id, message, finish_reason, prompt_tokens, completion_tokens, record)


class Handler(BaseHTTPRequestHandler):
    server: TransformersOpenAIServer

    def log_message(self, fmt: str, *args: Any) -> None:
        if not self.server.args.quiet:
            super().log_message(fmt, *args)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def send_json(self, payload: Any, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=True, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json({"ok": True, "model": self.server.args.model_name, "load_s": self.server.load_s})
            return
        if self.path == "/v1/models":
            self.send_json(
                {
                    "object": "list",
                    "data": [{"id": self.server.args.model_name, "object": "model", "created": 0, "owned_by": "agent_arena"}],
                }
            )
            return
        if self.path.startswith("/debug/requests"):
            self.send_json({"requests": REQUEST_LOG})
            return
        self.send_json({"error": f"not found: {self.path}"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/debug/reset":
            REQUEST_LOG.clear()
            self.send_json({"ok": True})
            return
        if self.path != "/v1/chat/completions":
            self.send_json({"error": f"not found: {self.path}"}, HTTPStatus.NOT_FOUND)
            return
        payload = self.read_json()
        if payload.get("stream"):
            self.send_json({"error": "streaming is not implemented by agent_arena transformers shim"}, 400)
            return
        self.send_json(self.server.generate(payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", default="hf-local")
    parser.add_argument("--mode", choices=["stock", "thinking_off", "thinking_on"], default="stock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--parser",
        choices=[
            "strict",
            "tolerant",
            "llama3_json",
            "mistral_tool",
            "qwen3_native",
            "qcom_tool",
            "nemotron_native",
            "nemotron_bfcl_schema",
            "nemotron_bfcl_schema_guided",
            "nemotron_bfcl_official",
            "nemotron_bfcl_official_exact",
            "nemotron_bfcl_official_clean_args",
            "nemotron_bfcl_official_strict_names",
            "nemotron_bfcl_official_strict_schema",
            "nemotron_bfcl_official_strict_schema_enhanced",
            "nemotron_bfcl_official_strict_schema_enhanced_guarded",
            "nemotron_bfcl_official_strict_schema_guarded",
            "nemotron_bfcl_official_user_strict_schema",
            "nemotron_bfcl_official_user_selective_schema",
            "nemotron_bfcl_official_user_selective_schema_guarded",
            "nemotron_bfcl_official_user_selective_schema_supported",
            "nemotron_bfcl_official_strict_schema_exact",
            "nemotron_bfcl_official_strict_schema_values",
            "nemotron_bfcl_modelcard_supported",
            "nemotron_bfcl_user",
        ],
        default="tolerant",
    )
    parser.add_argument("--multi-tool-policy", choices=["all", "first"], default="all")
    parser.add_argument("--tool-output-mode", choices=["llama", "json"], default="llama")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path.home() / "agent_arena_results" / "openai_transformers_server")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = TransformersOpenAIServer((args.host, args.port), args)
    print(
        f"OPENAI_TRANSFORMERS_SERVER http://{args.host}:{args.port}/v1 "
        f"model={args.model_name} parser={args.parser}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
