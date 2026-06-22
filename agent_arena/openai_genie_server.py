#!/usr/bin/env python3
"""Small OpenAI-compatible HTTP shim for Genie CLI models.

This server is intentionally modest: it exposes enough of
POST /v1/chat/completions for real agent clients such as Pydantic AI to
exercise tool-calling loops against a Genie bundle on the EVK.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agent_arena.model_client import (
    build_chat_prompt,
    extract_first_json,
    extract_genie_answer,
    qairt_env,
    split_reasoning,
)
from agent_arena.tool_arena import extract_action


REQUEST_LOG: list[dict[str, Any]] = []


def mode_system_text(mode: str) -> str:
    if mode == "thinking_on":
        return (
            "detailed thinking on\n"
            "Use a short reasoning budget for agent tool-calling tasks. Close any "
            "<think> block quickly, then provide the requested JSON object."
        )
    if mode == "thinking_off":
        return (
            "detailed thinking off\n"
            "Provide only the requested JSON object."
        )
    return "You are a concise OpenAI-compatible chat model."


def assistant_prefill(mode: str) -> str:
    if mode == "thinking_off":
        return "<think>\n</think>\n"
    return ""


def compact_json(value: Any, max_chars: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return compact_json(content, max_chars=4000)


def structured_tool_content(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    if isinstance(content, list):
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        ]
        if len(texts) == 1:
            return structured_tool_content(texts[0])
    return content


def render_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    lines: list[str] = []
    last_user = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "unknown")).upper()
        content = message_content_text(message.get("content", ""))
        if role == "USER":
            last_user = content
        if role == "ASSISTANT" and message.get("tool_calls"):
            lines.append(f"{role}_TOOL_CALLS: {compact_json(message.get('tool_calls'))}")
            if content:
                lines.append(f"{role}: {content}")
            continue
        if role == "TOOL":
            tool_call_id = message.get("tool_call_id", "")
            lines.append(f"TOOL_RESULT tool_call_id={tool_call_id}: {content}")
            continue
        lines.append(f"{role}: {content}")
    return last_user, "\n".join(lines)


def render_request_prompt(payload: dict[str, Any], parser: str) -> str:
    messages = payload.get("messages", [])
    tools = payload.get("tools", [])
    last_user, rendered_messages = render_messages(messages if isinstance(messages, list) else [])
    tool_names = []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                function = tool.get("function", {})
                if isinstance(function, dict) and function.get("name"):
                    tool_names.append(str(function["name"]))
    strict_note = (
        "The server parser is STRICT: only the exact JSON shapes below will become "
        "OpenAI tool calls or final messages."
        if parser == "strict"
        else "The server parser is TOLERANT, but malformed outputs are still logged."
    )
    return (
        "You are serving an OpenAI-compatible Chat Completions API for a real agent client.\n"
        f"{strict_note}\n\n"
        "Return exactly one JSON object and no prose.\n"
        "If the answer is available from the current messages, finish with this shape:\n"
        '{"content":"final answer"}\n\n'
        "Only when a tool is needed, call one tool with this shape:\n"
        '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"TOOL_NAME_FROM_TOOLS_JSON","arguments":{"PARAMETER_NAME":"value"}}}]}\n'
        "TOOL_NAME_FROM_TOOLS_JSON and PARAMETER_NAME are placeholders; never output them literally.\n\n"
        "Rules:\n"
        "- If no tool is needed, do not call a tool.\n"
        "- Use only tool names from TOOLS_JSON.\n"
        "- Do not guess values that should come from a tool.\n"
        "- Do not answer with placeholders such as <value>, owner_of_id, or T-104.owner.\n"
        "- If the task asks to find, fetch, search, read, calculate, or inspect data, call a tool first.\n"
        "- If a tool result is already present in MESSAGES_JSON, use it to answer.\n"
        "- Use at most one tool call in each response.\n"
        "- Function arguments must be a JSON object.\n\n"
        f"LAST_USER_TASK:\n{last_user}\n\n"
        f"AVAILABLE_TOOL_NAMES:\n{', '.join(tool_names) if tool_names else 'none'}\n\n"
        f"TOOLS_JSON:\n{compact_json(tools)}\n\n"
        f"CONVERSATION:\n{rendered_messages}\n"
    )


def render_qcom_tool_prompt(payload: dict[str, Any], mode: str) -> str:
    """Render the generic GenieAPIService-style <tool_call> tool prompt."""
    messages = payload.get("messages", [])
    messages = list(messages) if isinstance(messages, list) else []
    tools = payload.get("tools", [])
    tools = tools if isinstance(tools, list) else []

    system_messages: list[str] = []
    conversation: list[str] = []
    tool_names_by_id: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        if role == "system":
            system_messages.append(message_content_text(message.get("content", "")).strip())
            continue
        if role == "assistant" and message.get("tool_calls"):
            rendered_calls: list[str] = []
            for call in message.get("tool_calls", []):
                if not isinstance(call, dict):
                    continue
                function = call.get("function", {})
                if not isinstance(function, dict):
                    continue
                call_id = str(call.get("id", ""))
                name = function.get("name", "")
                if call_id and isinstance(name, str):
                    tool_names_by_id[call_id] = name
                arguments = parse_arguments(function.get("arguments", {})) or {}
                rendered_calls.append(
                    "<tool_call>\n"
                    + json.dumps(
                        {"name": name, "arguments": arguments},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n</tool_call>"
                )
            if rendered_calls:
                conversation.append("assistant:\n" + "\n".join(rendered_calls))
            continue
        if role == "tool":
            tool_name = message.get("name") or tool_names_by_id.get(str(message.get("tool_call_id", "")))
            content = message_content_text(message.get("content", "")).strip()
            if tool_name:
                content = f"{tool_name}: {content}"
            conversation.append(f"tool:\n<tool_response>\n{content}\n</tool_response>")
            continue
        conversation.append(f"{role}:\n{message_content_text(message.get('content', '')).strip()}")

    tool_descs = "\n".join(compact_json(tool) for tool in tools)
    tool_prompt = (
        "\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{tool_descs}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>\n"
    )
    if mode == "thinking_off":
        mode_text = "detailed thinking off\nReturn only the final answer or the requested <tool_call> block."
    elif mode == "thinking_on":
        mode_text = (
            "detailed thinking on\n"
            "Use a short reasoning budget, close any <think> block, then return the final answer or <tool_call> block."
        )
    else:
        mode_text = "You are a concise OpenAI-compatible chat model."
    system_text = "\n".join([mode_text, *system_messages, tool_prompt]).strip()
    user_text = "# Conversation\n\n" + "\n\n".join(conversation)
    return build_chat_prompt(system_text, user_text, assistant_prefill(mode))


def native_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    return function if isinstance(function, dict) else tool


def render_nemotron_native_prompt(payload: dict[str, Any], mode: str) -> str:
    """Render Nemotron Nano's tokenizer-native tool template."""
    messages = payload.get("messages", [])
    messages = list(messages) if isinstance(messages, list) else []
    tools = payload.get("tools", [])
    tools = tools if isinstance(tools, list) else []

    system_messages: list[str] = []
    conversation = [message for message in messages if isinstance(message, dict)]
    conversation_without_system: list[dict[str, Any]] = []
    for message in conversation:
        if message.get("role") == "system":
            system_messages.append(message_content_text(message.get("content", "")).strip())
        else:
            conversation_without_system.append(message)

    if mode == "thinking_off":
        mode_text = (
            "detailed thinking off\n"
            "Use tools directly. Do not write chain-of-thought. If a tool is needed, "
            "return only a <TOOLCALL>[...]</TOOLCALL> block."
        )
    elif mode == "thinking_on":
        mode_text = (
            "detailed thinking on\n"
            "Use a short reasoning budget, then return a <TOOLCALL>[...]</TOOLCALL> block "
            "or a concise final answer."
        )
    else:
        mode_text = "You are a concise OpenAI-compatible chat model."

    tool_schemas = [native_tool_schema(tool) for tool in tools if isinstance(tool, dict)]
    system_text = "\n".join([mode_text, *system_messages]).strip()
    parts: list[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        system_text,
    ]
    if tool_schemas:
        if system_text:
            parts.append("\n\n")
        parts.append("<AVAILABLE_TOOLS>[")
        parts.append(", ".join(json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tool_schemas))
        parts.append("]</AVAILABLE_TOOLS>")
    parts.append("<|eot_id|>")

    tool_names_by_id: dict[str, str] = {}
    idx = 0
    while idx < len(conversation_without_system):
        message = conversation_without_system[idx]
        role = str(message.get("role", "user"))
        if role == "assistant" and message.get("tool_calls"):
            rendered_calls: list[str] = []
            for call in message.get("tool_calls", []):
                if not isinstance(call, dict):
                    continue
                function = call.get("function", {})
                if not isinstance(function, dict):
                    continue
                call_id = str(call.get("id", ""))
                name = function.get("name", "")
                if call_id and isinstance(name, str):
                    tool_names_by_id[call_id] = name
                arguments = parse_arguments(function.get("arguments", {})) or {}
                rendered_calls.append(
                    json.dumps(
                        {"name": name, "arguments": arguments},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            if rendered_calls:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                parts.append("<TOOLCALL>[")
                parts.append(", ".join(rendered_calls))
                parts.append("]</TOOLCALL><|eot_id|>")
            idx += 1
            continue
        if role == "tool":
            response_items: list[str] = []
            while idx < len(conversation_without_system):
                tool_message = conversation_without_system[idx]
                if not isinstance(tool_message, dict) or tool_message.get("role") != "tool":
                    break
                tool_name = tool_message.get("name") or tool_names_by_id.get(str(tool_message.get("tool_call_id", "")))
                response_items.append(render_nemotron_tool_response_item(tool_message, str(tool_name or "")))
                idx += 1
            parts.append("<|start_header_id|>user<|end_header_id|>\n\n")
            parts.append("<TOOL_RESPONSE>[")
            parts.append(", ".join(response_items))
            parts.append("]</TOOL_RESPONSE><|eot_id|>")
            continue
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n")
        parts.append(message_content_text(message.get("content", "")).strip())
        parts.append("<|eot_id|>")
        idx += 1

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    if mode == "thinking_off":
        parts.append(assistant_prefill(mode))
    return "".join(parts)


def render_nemotron_tool_response_item(message: dict[str, Any], tool_name: str) -> str:
    content = structured_tool_content(message.get("content", ""))
    if isinstance(content, str):
        try:
            item = json.loads(content)
        except json.JSONDecodeError:
            item = {"content": content}
    else:
        item = content
    if tool_name and isinstance(item, dict) and "name" not in item and "tool" not in item:
        item = {"name": tool_name, "content": item}
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=4)


def render_llama3_json_prompt(payload: dict[str, Any], mode: str, tool_output_mode: str = "llama") -> str:
    """Render the vLLM Llama 3.1 JSON tool-calling chat template in plain Python."""
    messages = payload.get("messages", [])
    messages = list(messages) if isinstance(messages, list) else []
    tools = payload.get("tools", [])
    tools = tools if isinstance(tools, list) and tools else None

    system_message = ""
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        system_message = message_content_text(messages[0].get("content", "")).strip()
        messages = messages[1:]
    elif tools is not None:
        system_message = (
            "You are a helpful assistant with tool calling capabilities. Only reply "
            "with a tool call if the function exists in the library provided by the "
            "user. If it doesn't exist, just reply directly in natural language. "
            "When you receive a tool call response, use the output to format an "
            "answer to the original user question."
        )
    if mode == "thinking_off":
        system_message = ("detailed thinking off\n" + system_message).strip()
    elif mode == "thinking_on":
        system_message = (
            "detailed thinking on\n"
            "Use a short reasoning budget for tool calls, then output the Llama JSON tool call or final answer.\n"
            + system_message
        ).strip()

    parts: list[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    ]
    if tools is not None:
        parts.append("Environment: ipython\n")
    parts.append("Cutting Knowledge Date: December 2023\n")
    parts.append("Today Date: 26 Jul 2024\n\n")
    parts.append(system_message)
    parts.append("<|eot_id|>")

    if tools is not None:
        first_user_message = ""
        if messages:
            first = messages[0]
            if isinstance(first, dict) and first.get("role") == "user":
                first_user_message = message_content_text(first.get("content", "")).strip()
                messages = messages[1:]
        parts.append("<|start_header_id|>user<|end_header_id|>\n\n")
        parts.append(
            "Given the following functions, please respond with a JSON for a function call "
            "with its proper arguments that best answers the given prompt.\n\n"
        )
        parts.append(
            'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. '
            "Do not use variables.\n\n"
        )
        parts.append(
            "When calling functions, output JSON only: no markdown fences, prose, or final-answer text. Prefer one "
            "function call per assistant message. If you output multiple function calls, they must be independent "
            "observation/read-only calls. Never batch state-changing action calls before the needed tool results have "
            "been returned.\n\n"
        )
        for tool in tools:
            parts.append(pretty_json(tool))
            parts.append("\n\n")
        parts.append(first_user_message)
        parts.append("<|eot_id|>")

    tool_names_by_id: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        if message.get("tool_calls"):
            tool_calls = message.get("tool_calls")
            rendered_calls: list[str] = []
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function", {})
                    if not isinstance(function, dict):
                        continue
                    call_id = str(call.get("id", ""))
                    if call_id and isinstance(function.get("name"), str):
                        tool_names_by_id[call_id] = function["name"]
                    arguments = function.get("arguments", {})
                    parsed_arguments = parse_arguments(arguments) or {}
                    rendered_calls.append(
                        json.dumps(
                            {"name": function.get("name", ""), "parameters": parsed_arguments},
                            ensure_ascii=False,
                            separators=(",", ": "),
                        )
                    )
            if rendered_calls:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                parts.append("\n".join(rendered_calls))
                parts.append("<|eot_id|>")
            continue
        if role == "tool":
            parts.append("<|start_header_id|>ipython<|end_header_id|>\n\n")
            if tool_output_mode == "json":
                output: dict[str, Any] = {"output": structured_tool_content(message.get("content", ""))}
                tool_name = message.get("name") or tool_names_by_id.get(str(message.get("tool_call_id", "")))
                if isinstance(tool_name, str) and tool_name:
                    output = {"tool": tool_name, **output}
                parts.append(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
            else:
                parts.append(json.dumps({"output": message_content_text(message.get("content", ""))}, ensure_ascii=False))
            parts.append("<|eot_id|>")
            continue
        if role == "ipython":
            parts.append("<|start_header_id|>ipython<|end_header_id|>\n\n")
            parts.append(message_content_text(message.get("content", "")).strip())
            parts.append("<|eot_id|>")
            continue
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n")
        parts.append(message_content_text(message.get("content", "")).strip())
        parts.append("<|eot_id|>")

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def render_mistral_tool_prompt(payload: dict[str, Any], mode: str) -> str:
    """Render Ministral/Mistral native tool-calling prompt tokens."""
    messages = payload.get("messages", [])
    messages = list(messages) if isinstance(messages, list) else []
    tools = payload.get("tools", [])
    tools = tools if isinstance(tools, list) else []

    system_messages: list[str] = []
    conversation: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            system_messages.append(message_content_text(message.get("content", "")).strip())
        else:
            conversation.append(message)

    mode_text = ""
    if mode == "thinking_off":
        mode_text = (
            "Use tools directly when useful. Do not write chain-of-thought. "
            "If a tool is needed, output only the native [TOOL_CALLS] block."
        )
    elif mode == "thinking_on":
        mode_text = (
            "Use brief private reasoning, then output either a concise final answer "
            "or the native [TOOL_CALLS] block."
        )

    system_text = "\n".join(part for part in [mode_text, *system_messages] if part).strip()
    parts: list[str] = ["<s>"]
    if system_text:
        parts.append("[SYSTEM_PROMPT]")
        parts.append(system_text)
        parts.append("[/SYSTEM_PROMPT]")
    if tools:
        parts.append("[AVAILABLE_TOOLS]")
        parts.append(json.dumps(tools, ensure_ascii=False, separators=(",", ":")))
        parts.append("[/AVAILABLE_TOOLS]")

    tool_names_by_id: dict[str, str] = {}
    for message in conversation:
        role = str(message.get("role", "user"))
        if role == "user":
            parts.append("[INST]")
            parts.append(message_content_text(message.get("content", "")).strip())
            parts.append("[/INST]")
            continue
        if role == "assistant":
            content = message_content_text(message.get("content", "")).strip()
            if content:
                parts.append(content)
            tool_calls = message.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function", {})
                    if not isinstance(function, dict):
                        continue
                    name = str(function.get("name", ""))
                    call_id = str(call.get("id", ""))
                    if call_id and name:
                        tool_names_by_id[call_id] = name
                    arguments = parse_arguments(function.get("arguments", {})) or {}
                    parts.append("[TOOL_CALLS]")
                    parts.append(name)
                    parts.append("[ARGS]")
                    parts.append(json.dumps(arguments, ensure_ascii=False, separators=(",", ":")))
            parts.append("</s>")
            continue
        if role == "tool":
            content = structured_tool_content(message.get("content", ""))
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            tool_name = message.get("name") or tool_names_by_id.get(str(message.get("tool_call_id", "")))
            if tool_name:
                content = f"{tool_name}: {content}"
            parts.append("[TOOL_RESULTS]")
            parts.append(content)
            parts.append("[/TOOL_RESULTS]")

    return "".join(parts)


def parse_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


SCHEMA_DESCRIPTOR_KEYS = {
    "type",
    "description",
    "enum",
    "items",
    "properties",
    "required",
    "default",
    "minimum",
    "maximum",
    "oneOf",
    "anyOf",
    "allOf",
    "format",
}


def schema_echo_value(name: str, value: Any) -> tuple[Any, bool]:
    if not isinstance(value, dict):
        return value, False
    if "value" in value:
        return value["value"], True
    nested_arguments = value.get("arguments")
    if isinstance(nested_arguments, dict):
        if name in nested_arguments:
            return nested_arguments[name], True
        repaired, did_repair = repair_schema_echo_arguments(nested_arguments)
        if did_repair:
            return repaired, True
        if len(nested_arguments) == 1:
            return next(iter(nested_arguments.values())), True
    nested_parameters = value.get("parameters")
    if isinstance(nested_parameters, dict):
        repaired, did_repair = repair_schema_echo_arguments(nested_parameters)
        if did_repair:
            return repaired, True
    properties = value.get("properties")
    if isinstance(properties, dict):
        repaired, did_repair = repair_schema_echo_arguments(value)
        if did_repair:
            return repaired, True
    return value, False


def repair_schema_echo_arguments(arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    wrapped = arguments.get("arguments")
    if set(arguments) == {"arguments"} and isinstance(wrapped, dict):
        return repair_schema_echo_arguments(wrapped)
    wrapped = arguments.get("parameters")
    if set(arguments) == {"parameters"} and isinstance(wrapped, dict):
        return repair_schema_echo_arguments(wrapped)

    properties = arguments.get("properties")
    if isinstance(properties, dict) and properties:
        repaired: dict[str, Any] = {}
        did_repair = False
        for name, value in properties.items():
            repaired_value, value_repaired = schema_echo_value(str(name), value)
            repaired[str(name)] = repaired_value
            did_repair = did_repair or value_repaired
        if did_repair:
            return repaired, True

    repaired = {}
    did_repair = False
    for name, value in arguments.items():
        if name in SCHEMA_DESCRIPTOR_KEYS:
            repaired[name] = value
            continue
        repaired_value, value_repaired = schema_echo_value(str(name), value)
        repaired[str(name)] = repaired_value
        did_repair = did_repair or value_repaired
    return (repaired, True) if did_repair else (arguments, False)


def normalize_tool_call(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    name = function.get("name") or call.get("name") or call.get("tool")
    if not isinstance(name, str) or not name:
        return None
    arguments = (
        function.get("arguments")
        if "arguments" in function
        else function.get("parameters", call.get("arguments", call.get("parameters", call.get("args", {}))))
    )
    parsed_arguments = parse_arguments(arguments)
    if parsed_arguments is None:
        return None
    return {
        "type": "tool",
        "id": str(call.get("id", "call_1")),
        "name": name,
        "arguments": parsed_arguments,
    }


def normalize_openai_object(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, list) and obj:
        return normalize_openai_object(obj[0])
    if not isinstance(obj, dict):
        return None
    tool_calls = obj.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        parsed_calls = [parsed for call in tool_calls if (parsed := normalize_tool_call(call))]
        if parsed_calls:
            return {
                "type": "tool",
                "tool_calls": parsed_calls,
                "id": parsed_calls[0].get("id", "call_1"),
                "name": parsed_calls[0]["name"],
                "arguments": parsed_calls[0]["arguments"],
            }
    content = obj.get("content")
    if isinstance(content, str) and content.strip():
        return {"type": "final", "content": content}
    parsed_call = normalize_tool_call(obj)
    if parsed_call:
        return parsed_call
    name = obj.get("name")
    parameters = obj.get("parameters")
    if isinstance(name, str):
        parsed_arguments = parse_arguments(parameters if "parameters" in obj else obj.get("arguments", {}))
        if parsed_arguments is not None:
            return {
                "type": "tool",
                "id": str(obj.get("id", "call_1")),
                "name": name,
                "arguments": parsed_arguments,
            }
    if "final" in obj or "answer" in obj:
        return {"type": "final", "content": str(obj.get("final", obj.get("answer", "")))}
    return None


def normalize_nemotron_native_call(call: Any, idx: int) -> tuple[dict[str, Any] | None, bool]:
    parsed = normalize_tool_call(call)
    if not parsed:
        return None, False
    repaired_arguments, repaired = repair_schema_echo_arguments(parsed.get("arguments", {}))
    if repaired:
        parsed["arguments"] = repaired_arguments
        parsed["argument_repair"] = "schema_echo_properties"
    parsed["id"] = str(parsed.get("id") or f"call_{idx}")
    return parsed, repaired


def normalize_nemotron_native_calls(call: Any, start_idx: int) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(call, dict):
        return [], False
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    name = function.get("name") or call.get("name") or call.get("tool")
    if not isinstance(name, str) or not name:
        return [], False
    raw_arguments = (
        function.get("arguments")
        if "arguments" in function
        else function.get("parameters", call.get("arguments", call.get("parameters", call.get("args", {}))))
    )
    if isinstance(raw_arguments, list):
        calls: list[dict[str, Any]] = []
        repaired = False
        for offset, item in enumerate(raw_arguments):
            if not isinstance(item, dict):
                continue
            item_arguments, item_repaired = repair_schema_echo_arguments(item)
            repaired = repaired or item_repaired
            calls.append(
                {
                    "type": "tool",
                    "id": f"call_{start_idx + len(calls)}",
                    "name": name,
                    "arguments": item_arguments,
                }
            )
        return calls, repaired
    parsed, repaired = normalize_nemotron_native_call(call, start_idx)
    return ([parsed] if parsed else []), repaired


def json_candidates(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        candidates.append(obj)
    return candidates


def leading_json(text: str) -> Any | None:
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(stripped)
        return obj
    except json.JSONDecodeError:
        return None


def merge_tool_calls(calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for call in calls:
        key = (
            str(call.get("name", "")),
            json.dumps(call.get("arguments", {}), sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(call)
    if not merged:
        return None
    return {
        "type": "tool",
        "tool_calls": merged,
        "id": merged[0].get("id", "call_1"),
        "name": merged[0]["name"],
        "arguments": merged[0]["arguments"],
    }


def csv_names(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def extract_regex_tool_call(raw_text: str) -> dict[str, Any] | None:
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw_text)
    if not name_match:
        return None
    args: dict[str, Any] = {}
    args_match = re.search(r'"(?:arguments|parameters)"\s*:\s*', raw_text[name_match.end() :])
    if args_match:
        decoder = json.JSONDecoder()
        start = name_match.end() + args_match.end()
        starts = [idx for idx in (raw_text.find("{", start), raw_text.find("[", start)) if idx != -1]
        arg_start = min(starts) if starts else -1
        if arg_start != -1:
            try:
                parsed, _ = decoder.raw_decode(raw_text[arg_start:])
                if isinstance(parsed, list):
                    calls = []
                    for idx, item in enumerate(parsed, start=1):
                        if not isinstance(item, dict):
                            continue
                        repaired, _ = repair_schema_echo_arguments(item)
                        calls.append(
                            {
                                "type": "tool",
                                "id": f"call_{idx}",
                                "name": name_match.group(1),
                                "arguments": repaired,
                            }
                        )
                    if calls:
                        return {
                            "type": "tool",
                            "tool_calls": calls,
                            "id": calls[0]["id"],
                            "name": calls[0]["name"],
                            "arguments": calls[0]["arguments"],
                        }
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
    args, _ = repair_schema_echo_arguments(args)
    return {
        "type": "tool",
        "id": "call_1",
        "name": name_match.group(1),
        "arguments": args,
    }


def strict_parse(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    obj = leading_json(raw_text)
    if not isinstance(obj, dict):
        return None, "no_json_object"
    parsed = normalize_openai_object(obj)
    if not parsed:
        return None, "missing_tool_calls_or_content"
    return parsed, parsed["type"]


def tolerant_parse(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    strict, status = strict_parse(raw_text)
    if strict:
        return strict, status
    candidate_calls: list[dict[str, Any]] = []
    final_candidate: dict[str, Any] | None = None
    for candidate in json_candidates(raw_text):
        parsed = normalize_openai_object(candidate)
        if parsed:
            if parsed["type"] == "tool":
                parsed_calls = parsed.get("tool_calls")
                if isinstance(parsed_calls, list) and parsed_calls:
                    candidate_calls.extend(parsed_calls)
                else:
                    candidate_calls.append(parsed)
            elif final_candidate is None:
                final_candidate = parsed
    merged_calls = merge_tool_calls(candidate_calls)
    if merged_calls:
        return merged_calls, "tool_json_candidates"
    if final_candidate:
        return final_candidate, "final_json_candidate"
    obj = extract_first_json(raw_text)
    if isinstance(obj, dict) and isinstance(obj.get("content"), str):
        nested_text = obj["content"]
        if nested_text != raw_text:
            nested, nested_status = tolerant_parse(nested_text)
            if nested:
                return nested, f"{nested_status}_from_content_string"
    regex_tool = extract_regex_tool_call(raw_text)
    if regex_tool:
        return regex_tool, "tool_regex"
    action = extract_action(raw_text, "openai")
    if not action:
        return None, status
    if action["action"] == "final":
        return {"type": "final", "content": action["answer"]}, "final_tolerant"
    arguments = action.get("arguments", {})
    if isinstance(arguments, dict) and set(arguments) == {"arguments"} and isinstance(arguments["arguments"], dict):
        arguments = arguments["arguments"]
    return {
        "type": "tool",
        "id": action.get("call_id", "call_1"),
        "name": action.get("tool", ""),
        "arguments": arguments,
    }, "tool_tolerant"


def qcom_tool_parse(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    calls: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw_text, flags=re.DOTALL | re.IGNORECASE):
        for candidate in json_candidates(block):
            parsed = normalize_openai_object(candidate)
            if not parsed or parsed.get("type") != "tool":
                continue
            parsed_calls = parsed.get("tool_calls")
            if isinstance(parsed_calls, list) and parsed_calls:
                calls.extend(parsed_calls)
            else:
                calls.append(parsed)
    if not calls:
        for line in raw_text.splitlines():
            line = line.strip()
            if not line.startswith('{"name"'):
                continue
            parsed = normalize_openai_object(leading_json(line))
            if parsed and parsed.get("type") == "tool":
                calls.append(parsed)
    merged_calls = merge_tool_calls(calls)
    if merged_calls:
        return merged_calls, "tool_qcom_tool_call"
    without_tool_calls = re.sub(
        r"<tool_call>[\s\S]*?</tool_call>\s*",
        "",
        raw_text,
        flags=re.IGNORECASE,
    ).strip()
    tolerant, status = tolerant_parse(without_tool_calls or raw_text)
    if tolerant:
        return tolerant, f"{status}_qcom_fallback"
    return {"type": "final", "content": without_tool_calls or raw_text.strip()}, "final_qcom_tool"


def llama3_json_parse(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    text = raw_text.strip()
    if text.startswith("<|python_tag|>"):
        text = text[len("<|python_tag|>") :].strip()
    calls: list[dict[str, Any]] = []
    for candidate in json_candidates(text):
        if not isinstance(candidate, dict):
            continue
        name = candidate.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = candidate.get("parameters") if "parameters" in candidate else candidate.get("arguments", {})
        parsed_arguments = parse_arguments(arguments)
        if parsed_arguments is None:
            continue
        calls.append(
            {
                "type": "tool",
                "id": "call_1" if not calls else f"call_{len(calls) + 1}",
                "name": name,
                "arguments": parsed_arguments,
            }
        )
    merged_calls = merge_tool_calls(calls)
    if merged_calls:
        return merged_calls, "tool_llama3_json"
    return {"type": "final", "content": raw_text.strip()}, "final_llama3_json"


def mistral_tool_parse(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    calls: list[dict[str, Any]] = []
    pattern = re.compile(
        r"\[TOOL_CALLS\]\s*([A-Za-z_][\w.-]*)\s*\[ARGS\]\s*(\{[\s\S]*?})(?=\s*(?:\[TOOL_CALLS\]|</s>|\[END\]|$))",
        flags=re.IGNORECASE,
    )
    for idx, match in enumerate(pattern.finditer(raw_text), start=1):
        arguments = parse_arguments(match.group(2).strip())
        if arguments is None:
            continue
        calls.append(
            {
                "type": "tool",
                "id": f"call_{idx}",
                "name": match.group(1).strip(),
                "arguments": arguments,
            }
        )
    if not calls:
        for candidate in json_candidates(raw_text):
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("name") or candidate.get("function")
            arguments = candidate.get("arguments") if "arguments" in candidate else candidate.get("parameters", {})
            parsed_arguments = parse_arguments(arguments)
            if isinstance(name, str) and parsed_arguments is not None:
                calls.append(
                    {
                        "type": "tool",
                        "id": "call_1" if not calls else f"call_{len(calls) + 1}",
                        "name": name,
                        "arguments": parsed_arguments,
                    }
                )
    merged_calls = merge_tool_calls(calls)
    if merged_calls:
        return merged_calls, "tool_mistral_native"
    without_calls = re.sub(
        r"\[TOOL_CALLS\][\s\S]*?(?=(?:\[TOOL_CALLS\]|</s>|\[END\]|$))",
        "",
        raw_text,
        flags=re.IGNORECASE,
    ).strip()
    return {"type": "final", "content": without_calls or raw_text.strip()}, "final_mistral_native"


def nemotron_native_parse(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    calls: list[dict[str, Any]] = []
    repaired = False
    for block in re.findall(r"<TOOLCALL>\s*(.*?)\s*</TOOLCALL>", raw_text, flags=re.DOTALL | re.IGNORECASE):
        for candidate in json_candidates(block):
            candidate_calls = candidate if isinstance(candidate, list) else [candidate]
            for call in candidate_calls:
                parsed_calls, call_repaired = normalize_nemotron_native_calls(call, len(calls) + 1)
                calls.extend(parsed_calls)
                repaired = repaired or call_repaired
    if not calls:
        for candidate in json_candidates(raw_text):
            candidate_calls = candidate if isinstance(candidate, list) else [candidate]
            for call in candidate_calls:
                parsed_calls, call_repaired = normalize_nemotron_native_calls(call, len(calls) + 1)
                calls.extend(parsed_calls)
                repaired = repaired or call_repaired
    merged_calls = merge_tool_calls(calls)
    if merged_calls:
        status = "tool_nemotron_native_repaired" if repaired else "tool_nemotron_native"
        return merged_calls, status
    without_tool_calls = re.sub(
        r"<TOOLCALL>[\s\S]*?</TOOLCALL>\s*",
        "",
        raw_text,
        flags=re.IGNORECASE,
    ).strip()
    tolerant, status = tolerant_parse(without_tool_calls or raw_text)
    if tolerant:
        return tolerant, f"{status}_nemotron_native_fallback"
    return {"type": "final", "content": without_tool_calls or raw_text.strip()}, "final_nemotron_native"


def openai_message(parsed: dict[str, Any] | None, raw_answer: str) -> tuple[dict[str, Any], str]:
    if not parsed:
        return {"role": "assistant", "content": raw_answer}, "stop"
    if parsed["type"] == "final":
        return {"role": "assistant", "content": parsed["content"]}, "stop"
    parsed_calls = parsed.get("tool_calls")
    tool_calls = parsed_calls if isinstance(parsed_calls, list) and parsed_calls else [parsed]
    return (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.get("id", f"call_{idx}"),
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=True),
                    },
                }
                for idx, call in enumerate(tool_calls, start=1)
            ],
        },
        "tool_calls",
    )


class GenieOpenAIServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], args: argparse.Namespace) -> None:
        super().__init__(server_address, Handler)
        self.args = args
        self.bundle = args.bundle.expanduser().resolve()
        self.work_dir = args.work_dir.expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.pending_tool_calls: list[dict[str, Any]] = []
        if not (self.bundle / "genie_config.json").exists():
            raise FileNotFoundError(f"Missing genie_config.json in {self.bundle}")
        if shutil.which("genie-t2t-run", path=qairt_env()["PATH"]) is None:
            raise FileNotFoundError("genie-t2t-run not found in QAIRT PATH")

    def payload_has_tool_result(self, payload: dict[str, Any]) -> bool:
        messages = payload.get("messages", [])
        return any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in messages
            if isinstance(messages, list)
        )

    def completed_tool_names(self, payload: dict[str, Any]) -> set[str]:
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            return set()
        calls_by_id: dict[str, str] = {}
        completed: set[str] = set()
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                tool_calls = message.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
                        call_id = str(call.get("id", ""))
                        function = call.get("function", {})
                        if call_id and isinstance(function, dict) and isinstance(function.get("name"), str):
                            calls_by_id[call_id] = function["name"]
            if message.get("role") == "tool":
                name = calls_by_id.get(str(message.get("tool_call_id", "")))
                if name:
                    completed.add(name)
        return completed

    def parsed_action_summary(self, parsed: dict[str, Any] | None) -> dict[str, Any]:
        if not parsed:
            return {}
        parsed_action: dict[str, Any] = {"type": parsed["type"]}
        if parsed["type"] == "tool":
            parsed_calls = parsed.get("tool_calls")
            if isinstance(parsed_calls, list) and parsed_calls:
                parsed_action["tool_calls"] = [
                    {
                        "name": call.get("name"),
                        "arguments": call.get("arguments", {}),
                    }
                    for call in parsed_calls
                ]
            parsed_action.update(
                {
                    "name": parsed.get("name"),
                    "arguments": parsed.get("arguments", {}),
                }
            )
        return parsed_action

    def apply_multi_tool_policy(self, parsed: dict[str, Any] | None, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not parsed or parsed.get("type") != "tool":
            return parsed
        calls = parsed.get("tool_calls")
        if not isinstance(calls, list) or len(calls) <= 1:
            return parsed
        if self.args.multi_tool_policy == "all":
            return parsed
        if self.args.multi_tool_policy == "safe_batch":
            safe_tools = csv_names(self.args.parallel_safe_tools)
            ready_tools = csv_names(self.args.action_ready_tools)
            call_names = {str(call.get("name", "")) for call in calls}
            if safe_tools and call_names <= safe_tools:
                return parsed
            if ready_tools and ready_tools <= self.completed_tool_names(payload):
                return parsed
            leading_safe: list[dict[str, Any]] = []
            for call in calls:
                if str(call.get("name", "")) not in safe_tools:
                    break
                leading_safe.append(call)
            if leading_safe:
                return merge_tool_calls(leading_safe)
        if self.args.multi_tool_policy == "queue":
            self.pending_tool_calls.extend(calls[1:])
        first = calls[0]
        return {
            "type": "tool",
            "id": first.get("id", "call_1"),
            "name": first["name"],
            "arguments": first["arguments"],
        }

    def queued_response(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self.args.multi_tool_policy != "queue" or not self.pending_tool_calls:
            return None
        if not self.payload_has_tool_result(payload):
            self.pending_tool_calls.clear()
            return None
        call = self.pending_tool_calls.pop(0)
        parsed = {
            "type": "tool",
            "id": call.get("id", "call_1"),
            "name": call["name"],
            "arguments": call["arguments"],
        }
        message, finish_reason = openai_message(parsed, "")
        record = {
            "request_index": self.counter,
            "request_id": request_id,
            "model": self.args.model_name,
            "mode": self.args.mode,
            "parser": self.args.parser,
            "elapsed_s": 0.0,
            "returncode": 0,
            "timed_out": False,
            "parse_status": "tool_queued",
            "parsed_ok": True,
            "parsed_action": self.parsed_action_summary(parsed),
            "finish_reason": finish_reason,
            "raw_answer": "",
            "final_answer": "",
            "paths": {},
        }
        REQUEST_LOG.append(record)
        return self.completion_payload(request_id, message, finish_reason, 1, 1, record)

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
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
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
                "timed_out": record["timed_out"],
            },
        }

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.counter += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        queued = self.queued_response(request_id, payload)
        if queued:
            return queued
        stem = f"{self.args.model_name}__{self.args.mode}__req{self.counter:05d}__{uuid.uuid4().hex[:8]}"
        prompt_path = self.work_dir / f"{stem}.prompt.txt"
        profile_path = self.work_dir / f"{stem}.profile.json"
        log_path = self.work_dir / f"{stem}.log"
        if self.args.parser == "llama3_json":
            prompt_text = render_llama3_json_prompt(payload, self.args.mode, self.args.tool_output_mode)
        elif self.args.parser == "mistral_tool":
            prompt_text = render_mistral_tool_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_native":
            prompt_text = render_nemotron_native_prompt(payload, self.args.mode)
        elif self.args.parser == "qcom_tool":
            prompt_text = render_qcom_tool_prompt(payload, self.args.mode)
        else:
            user_text = render_request_prompt(payload, self.args.parser)
            prompt_text = build_chat_prompt(
                mode_system_text(self.args.mode), user_text, assistant_prefill(self.args.mode)
            )
        prompt_path.write_text(prompt_text)
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [
                    "genie-t2t-run",
                    "-c",
                    "genie_config.json",
                    "--prompt_file",
                    str(prompt_path),
                    "--profile",
                    str(profile_path),
                ],
                cwd=self.bundle,
                env=qairt_env(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.args.timeout_s,
            )
            stdout = proc.stdout
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            returncode = -9
            timed_out = True
        elapsed_s = time.monotonic() - started
        log_path.write_text(stdout)
        raw = extract_genie_answer(stdout)
        split = split_reasoning(raw)
        answer = split["final_answer"]
        parse_input = answer or raw
        if self.args.parser == "strict":
            parsed, parse_status = strict_parse(parse_input)
        elif self.args.parser == "llama3_json":
            parsed, parse_status = llama3_json_parse(parse_input)
        elif self.args.parser == "mistral_tool":
            parsed, parse_status = mistral_tool_parse(parse_input)
        elif self.args.parser == "nemotron_native":
            parsed, parse_status = nemotron_native_parse(parse_input)
        elif self.args.parser == "qcom_tool":
            parsed, parse_status = qcom_tool_parse(parse_input)
        else:
            parsed, parse_status = tolerant_parse(parse_input)
        parsed = self.apply_multi_tool_policy(parsed, payload)
        message, finish_reason = openai_message(parsed, answer or raw)
        parsed_action = self.parsed_action_summary(parsed)
        record = {
            "request_index": self.counter,
            "request_id": request_id,
            "model": self.args.model_name,
            "mode": self.args.mode,
            "parser": self.args.parser,
            "multi_tool_policy": self.args.multi_tool_policy,
            "tool_output_mode": self.args.tool_output_mode,
            "elapsed_s": round(elapsed_s, 3),
            "returncode": returncode,
            "timed_out": timed_out,
            "parse_status": "timeout" if timed_out else parse_status,
            "parsed_ok": bool(parsed) and not timed_out,
            "parsed_action": parsed_action,
            "finish_reason": finish_reason,
            "raw_answer": raw,
            "final_answer": answer,
            "paths": {
                "prompt": str(prompt_path),
                "log": str(log_path),
                "profile": str(profile_path),
            },
        }
        REQUEST_LOG.append(record)
        if timed_out:
            message = {"role": "assistant", "content": ""}
            finish_reason = "stop"
        prompt_tokens = len(prompt_path.read_text()) // 4
        completion_tokens = len(raw) // 4
        return self.completion_payload(
            request_id, message, finish_reason, prompt_tokens, completion_tokens, record
        )


class Handler(BaseHTTPRequestHandler):
    server: GenieOpenAIServer

    def log_message(self, fmt: str, *args: Any) -> None:
        if not self.server.args.quiet:
            super().log_message(fmt, *args)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def send_json(self, payload: Any, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json({"ok": True, "model": self.server.args.model_name})
            return
        if self.path == "/v1/models":
            self.send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.args.model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "agent_arena",
                        }
                    ],
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
            self.send_json({"error": "streaming is not implemented by agent_arena shim"}, 400)
            return
        self.send_json(self.server.generate(payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-name", default="genie-local")
    parser.add_argument("--mode", choices=["stock", "thinking_off", "thinking_on"], default="thinking_off")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument(
        "--parser",
        choices=["strict", "tolerant", "llama3_json", "mistral_tool", "qcom_tool", "nemotron_native"],
        default="strict",
    )
    parser.add_argument("--multi-tool-policy", choices=["all", "first", "queue", "safe_batch"], default="all")
    parser.add_argument("--parallel-safe-tools", default="get_case,check_supplies,inspect_scene")
    parser.add_argument("--action-ready-tools", default="check_supplies,inspect_scene")
    parser.add_argument("--tool-output-mode", choices=["llama", "json"], default="llama")
    parser.add_argument("--work-dir", type=Path, default=Path.home() / "agent_arena_results" / "openai_genie_server")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = GenieOpenAIServer((args.host, args.port), args)
    print(
        f"OPENAI_GENIE_SERVER http://{args.host}:{args.port}/v1 "
        f"model={args.model_name} mode={args.mode} parser={args.parser}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
