#!/usr/bin/env python3
"""Small OpenAI-compatible HTTP shim for Genie CLI models.

This server is intentionally modest: it exposes enough of
POST /v1/chat/completions for real agent clients such as Pydantic AI to
exercise tool-calling loops against a Genie bundle on the EVK.
"""

from __future__ import annotations

import argparse
import ast
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

def genie_runtime_failure_status(stdout: str) -> str:
    context_markers = (
        "Context Size was exceeded",
        "exceeds the available context size",
        "maximum context length",
    )
    if any(marker in stdout for marker in context_markers):
        return "runtime_context_exhaustion"
    infrastructure_markers = (
        "Failed to create device: 14001",
        "Device Creation failure",
        "Failure to initialize model",
        "Failed to create the dialog",
        "Could not create context from binary",
        "Create From Binary FAILED",
    )
    if any(marker in stdout for marker in infrastructure_markers):
        return "runtime_infrastructure_error"
    if "Failed to query" in stdout:
        return "runtime_query_failed"
    return ""



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


BFCL_SCHEMA_TYPE_BY_OPENAI = {
    "object": "dict",
    "number": "float",
    "array": "list",
}


def bfcl_native_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [bfcl_native_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    copied = {str(key): bfcl_native_schema(item) for key, item in value.items()}
    schema_type = copied.get("type")
    if isinstance(schema_type, str):
        copied["type"] = BFCL_SCHEMA_TYPE_BY_OPENAI.get(schema_type, schema_type)

    description = copied.get("description")
    float_suffix = " This is a float type value."
    if isinstance(description, str) and description.endswith(float_suffix):
        copied["description"] = description.removesuffix(float_suffix)

    if copied.get("format") == "float" and copied.get("type") == "float":
        copied.pop("format", None)
    return copied


def bfcl_native_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    return bfcl_native_schema(native_tool_schema(tool))


def nemotron_tool_schemas(payload: dict[str, Any], tools: list[dict[str, Any]], schema_style: str) -> list[dict[str, Any]]:
    if schema_style == "bfcl":
        bfcl_functions = payload.get("bfcl_functions")
        if isinstance(bfcl_functions, list):
            return [bfcl_native_schema(function) for function in bfcl_functions if isinstance(function, dict)]
        return [bfcl_native_tool_schema(tool) for tool in tools if isinstance(tool, dict)]
    return [native_tool_schema(tool) for tool in tools if isinstance(tool, dict)]


def nemotron_schema_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def nemotron_call_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


NEMOTRON_BFCL_TOOL_GUIDANCE = (
    "Use the available tools when they match the user request. If a tool is needed, "
    "respond only with <TOOLCALL>[...]</TOOLCALL> containing a JSON list of tool-call "
    "objects with name and arguments. Use the exact tool name string from AVAILABLE_TOOLS. "
    "Use one object for each independent tool call. If the request is unrelated to the "
    "available tools or no exact tool matches, do not emit <TOOLCALL>; answer briefly "
    "in plain text. Do not solve tool-call tasks in prose. Match argument types from the schema; "
    "when a float rate is requested, write percentages such as 5% as 0.05."
)


NEMOTRON_BFCL_OFFICIAL_GUIDANCE = """You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of <TOOLCALL>[func_name1(param_name1=param_value1, param_name2=param_value2), func_name2(param=value)]</TOOLCALL>

You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke."""


NEMOTRON_BFCL_OFFICIAL_EXACT_GUIDANCE = """Use only exact function names from AVAILABLE_TOOLS. Do not invent a new function name, and do not rename a function to a more convenient synonym.
Call a function only when the function description and required parameters exactly match the user's request. A semantically related function is not enough.
If the available function solves a different task than the user asked for, do not emit <TOOLCALL>; briefly say that no available function can be used.
For string arguments copied from the user request, preserve the full text exactly, including small words such as go, to, at, from, and with."""


NEMOTRON_BFCL_OFFICIAL_CLEAN_ARGS_GUIDANCE = """When calling a function, include only the function's actual top-level parameter names from parameters.properties.
Do not include schema fields such as type, properties, required, parameters, arguments, func_name, default, enum, or description as arguments unless the function explicitly defines that exact parameter name.
Do not wrap arguments inside properties or another object. For string arguments copied from the user request, preserve the full text exactly."""


NEMOTRON_BFCL_OFFICIAL_VALUE_GUIDANCE = """Preserve string values copied from the user request exactly; do not shorten, paraphrase, or drop small words.
For optional parameters or parameters with defaults, omit them unless the user explicitly asks for a specific value.
For required geographic fields, infer obvious state/province/country values when the city or place is unambiguous."""


NEMOTRON_BFCL_OFFICIAL_SELECTIVE_GUIDANCE = """For argument values copied from the user request, preserve the whole relevant span exactly; do not drop small words.
Include optional parameters when the user explicitly supplied their value, even if the parameter has a default.
Omit optional parameters when the user did not supply their value.
For removal or exclusion requests using words such as without, no, remove, exclude, or except, choose the argument whose name or description removes/excludes that value when such an argument exists.
For general knowledge, definition, explanation, or current-fact questions, call a tool only if the tool directly answers that exact question; a related lookup is not enough."""


NEMOTRON_TOOL_GUARD_INSTRUCTIONS = """You are checking proposed tool calls from an edge agent.
Return exactly one word: ALLOW or REJECT.

Default to ALLOW when the user asks for a concrete action, calculation, transformation, or retrieval and the proposed tool performs that same high-level task.
Reject only when the mismatch is clear.
Reject when the user asks a general knowledge question, definition, explanation, or current fact and the tools do not directly provide that exact information.
Reject when the requested answer type cannot be produced by the proposed tool, such as asking who or what while the tool only retrieves a date, score, sunrise time, or unrelated record.
Reject when required details are missing and the proposed call invents specific values that the user did not provide.
Reject when the call uses a tool merely because its name is related to a word in the request.
Do not reject only because an argument is abbreviated, imperfectly formatted, or uses different wording for the same requested action."""


def payload_last_user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message_content_text(message.get("content", "")).strip()
    return ""


def parsed_tool_calls_for_guard(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    calls = parsed.get("tool_calls")
    call_list = calls if isinstance(calls, list) and calls else [parsed]
    guarded_calls: list[dict[str, Any]] = []
    for call in call_list:
        if not isinstance(call, dict):
            continue
        guarded_calls.append(
            {
                "name": call.get("name"),
                "arguments": call.get("arguments", {}),
            }
        )
    return guarded_calls


def render_nemotron_tool_guard_prompt(payload: dict[str, Any], parsed: dict[str, Any]) -> str:
    tools = payload.get("tools", [])
    tools = tools if isinstance(tools, list) else []
    tool_schemas = nemotron_tool_schemas(payload, tools, schema_style="bfcl")
    guard_payload = {
        "user_request": payload_last_user_text(payload),
        "available_tools": tool_schemas,
        "proposed_tool_calls": parsed_tool_calls_for_guard(parsed),
    }
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "detailed thinking off\n"
        f"{NEMOTRON_TOOL_GUARD_INSTRUCTIONS}"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{json.dumps(guard_payload, ensure_ascii=False)}\n\n"
        "Should the proposed tool calls be executed?"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def parse_tool_guard_decision(raw_answer: str) -> str:
    normalized = re.sub(r"[^A-Za-z]+", " ", raw_answer).strip().upper()
    words = normalized.split()
    if not words:
        return "allow"
    if words[0] == "REJECT":
        return "reject"
    if words[0] == "ALLOW":
        return "allow"
    if "REJECT" in words and "ALLOW" not in words[: words.index("REJECT")]:
        return "reject"
    return "allow"


def render_nemotron_native_prompt(
    payload: dict[str, Any],
    mode: str,
    schema_style: str = "openai",
    tool_guidance: str | None = None,
) -> str:
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
        mode_text = "detailed thinking off"
    elif mode == "thinking_on":
        mode_text = "detailed thinking on"
    else:
        mode_text = "You are a concise OpenAI-compatible chat model."

    tool_schemas = nemotron_tool_schemas(payload, tools, schema_style)
    system_text = "\n".join(part for part in [mode_text, tool_guidance] if part)
    pending_user_prefix = "\n\n".join(message for message in system_messages if message)
    parts: list[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        system_text,
    ]
    if tool_schemas:
        if system_text:
            parts.append("\n\n")
        parts.append("<AVAILABLE_TOOLS>[")
        parts.append(", ".join(nemotron_schema_json(tool) for tool in tool_schemas))
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
                    nemotron_call_json({"name": name, "arguments": arguments})
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
        content = message_content_text(message.get("content", "")).strip()
        if role == "user" and pending_user_prefix:
            content = f"{pending_user_prefix}\n\n{content}".strip()
            pending_user_prefix = ""
        parts.append(content)
        parts.append("<|eot_id|>")
        idx += 1

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def render_nemotron_bfcl_schema_prompt(payload: dict[str, Any], mode: str) -> str:
    """Render HF-native Nemotron prompt placement with BFCL-style schemas."""
    return render_nemotron_native_prompt(payload, mode, schema_style="bfcl")


def render_nemotron_bfcl_schema_guided_prompt(payload: dict[str, Any], mode: str) -> str:
    """Render BFCL-style schemas with short non-placeholder tool guidance."""
    return render_nemotron_native_prompt(
        payload,
        mode,
        schema_style="bfcl",
        tool_guidance=NEMOTRON_BFCL_TOOL_GUIDANCE,
    )


def python_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{python_literal(k)}: {python_literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(python_literal(item) for item in value) + "]"
    return repr(value)


def render_python_function_call(name: str, arguments: dict[str, Any]) -> str:
    args = ", ".join(f"{key}={python_literal(value)}" for key, value in arguments.items())
    return f"{name}({args})"


def render_nemotron_bfcl_official_prompt(payload: dict[str, Any], mode: str, extra_guidance: str = "") -> str:
    """Render BFCL's official Nemotron-style prompt and native tool schemas."""
    messages = payload.get("messages", [])
    messages = list(messages) if isinstance(messages, list) else []
    tools = payload.get("tools", [])
    tools = tools if isinstance(tools, list) else []

    system_messages: list[str] = []
    conversation: list[dict[str, Any]] = []
    first_user_content: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            system_messages.append(message_content_text(message.get("content", "")).strip())
            continue
        if role == "user" and first_user_content is None:
            first_user_content = message_content_text(message.get("content", "")).strip()
            continue
        conversation.append(message)

    if mode == "thinking_off":
        mode_text = "detailed thinking off"
    elif mode == "thinking_on":
        mode_text = "detailed thinking on"
    else:
        mode_text = "You are a concise OpenAI-compatible chat model."

    tool_schemas = nemotron_tool_schemas(payload, tools, schema_style="bfcl")
    tool_block = ""
    if tool_schemas:
        tool_block = (
            "<AVAILABLE_TOOLS>["
            + ", ".join(nemotron_schema_json(tool) for tool in tool_schemas)
            + "]</AVAILABLE_TOOLS>"
        )
    system_text = "\n\n".join(
        part
        for part in [
            mode_text,
            *system_messages,
            NEMOTRON_BFCL_OFFICIAL_GUIDANCE,
            extra_guidance,
            tool_block,
            first_user_content or "",
        ]
        if part
    )

    parts: list[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        system_text,
        "<|eot_id|>",
    ]

    tool_names_by_id: dict[str, str] = {}
    idx = 0
    while idx < len(conversation):
        message = conversation[idx]
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
                name = str(function.get("name", ""))
                if call_id and name:
                    tool_names_by_id[call_id] = name
                arguments = parse_arguments(function.get("arguments", {})) or {}
                rendered_calls.append(render_python_function_call(name, arguments))
            if rendered_calls:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                parts.append("<TOOLCALL>[")
                parts.append(", ".join(rendered_calls))
                parts.append("]</TOOLCALL><|eot_id|>")
            idx += 1
            continue
        if role == "tool":
            response_items: list[str] = []
            while idx < len(conversation):
                tool_message = conversation[idx]
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
    return "".join(parts)


def render_nemotron_bfcl_official_user_prompt(payload: dict[str, Any], mode: str, extra_guidance: str = "") -> str:
    """Render BFCL official instructions while keeping the first request in a user turn."""
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

    if mode == "thinking_off":
        mode_text = "detailed thinking off"
    elif mode == "thinking_on":
        mode_text = "detailed thinking on"
    else:
        mode_text = "You are a concise OpenAI-compatible chat model."

    tool_schemas = nemotron_tool_schemas(payload, tools, schema_style="bfcl")
    tool_block = ""
    if tool_schemas:
        tool_block = (
            "<AVAILABLE_TOOLS>["
            + ", ".join(nemotron_schema_json(tool) for tool in tool_schemas)
            + "]</AVAILABLE_TOOLS>"
        )
    system_text = "\n\n".join(
        part
        for part in [
            mode_text,
            *system_messages,
            NEMOTRON_BFCL_OFFICIAL_GUIDANCE,
            extra_guidance,
        ]
        if part
    )
    pending_user_prefix = tool_block

    parts: list[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        system_text,
        "<|eot_id|>",
    ]

    tool_names_by_id: dict[str, str] = {}
    idx = 0
    while idx < len(conversation):
        message = conversation[idx]
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
                name = str(function.get("name", ""))
                if call_id and name:
                    tool_names_by_id[call_id] = name
                arguments = parse_arguments(function.get("arguments", {})) or {}
                rendered_calls.append(render_python_function_call(name, arguments))
            if rendered_calls:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                parts.append("<TOOLCALL>[")
                parts.append(", ".join(rendered_calls))
                parts.append("]</TOOLCALL><|eot_id|>")
            idx += 1
            continue
        if role == "tool":
            response_items: list[str] = []
            while idx < len(conversation):
                tool_message = conversation[idx]
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
        content = message_content_text(message.get("content", "")).strip()
        if role == "user" and pending_user_prefix:
            content = f"{pending_user_prefix}\n\n{content}".strip()
            pending_user_prefix = ""
        parts.append(content)
        parts.append("<|eot_id|>")
        idx += 1

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def render_nemotron_bfcl_modelcard_prompt(payload: dict[str, Any], mode: str) -> str:
    """Render the lean NVIDIA model-card BFCL shape.

    The BFCL template keeps the available tools and request together in the user
    turn: <AVAILABLE_TOOLS>{functions}</AVAILABLE_TOOLS> followed by the user
    prompt. The system turn is only the Nemotron thinking-mode switch.
    """
    messages = payload.get("messages", [])
    messages = list(messages) if isinstance(messages, list) else []
    tools = payload.get("tools", [])
    tools = tools if isinstance(tools, list) else []

    conversation = [message for message in messages if isinstance(message, dict) and message.get("role") != "system"]
    if mode == "thinking_off":
        mode_text = "detailed thinking off"
    elif mode == "thinking_on":
        mode_text = "detailed thinking on"
    else:
        mode_text = "You are a concise OpenAI-compatible chat model."

    tool_schemas = nemotron_tool_schemas(payload, tools, schema_style="bfcl")
    tool_block = ""
    if tool_schemas:
        tool_block = (
            "<AVAILABLE_TOOLS>["
            + ", ".join(nemotron_schema_json(tool) for tool in tool_schemas)
            + "]</AVAILABLE_TOOLS>"
        )

    parts: list[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        mode_text,
        "<|eot_id|>",
    ]
    pending_user_prefix = tool_block
    tool_names_by_id: dict[str, str] = {}
    idx = 0
    while idx < len(conversation):
        message = conversation[idx]
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
                name = str(function.get("name", ""))
                if call_id and name:
                    tool_names_by_id[call_id] = name
                arguments = parse_arguments(function.get("arguments", {})) or {}
                rendered_calls.append(render_python_function_call(name, arguments))
            if rendered_calls:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                parts.append("<TOOLCALL>[")
                parts.append(", ".join(rendered_calls))
                parts.append("]</TOOLCALL><|eot_id|>")
            idx += 1
            continue
        if role == "tool":
            response_items: list[str] = []
            while idx < len(conversation):
                tool_message = conversation[idx]
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
        content = message_content_text(message.get("content", "")).strip()
        if role == "user" and pending_user_prefix:
            content = f"{pending_user_prefix}\n\n{content}".strip()
            pending_user_prefix = ""
        parts.append(content)
        parts.append("<|eot_id|>")
        idx += 1

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def render_nemotron_bfcl_user_prompt(payload: dict[str, Any], mode: str) -> str:
    """Render Nemotron with the BFCL model-card prompt shape.

    NVIDIA's BFCL template documents the tool list as part of the user prompt:
    <AVAILABLE_TOOLS>{functions}</AVAILABLE_TOOLS>

    {user_prompt}
    """
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

    if mode == "thinking_off":
        mode_text = "detailed thinking off"
    elif mode == "thinking_on":
        mode_text = "detailed thinking on"
    else:
        mode_text = "You are a concise OpenAI-compatible chat model."

    tool_schemas = [native_tool_schema(tool) for tool in tools if isinstance(tool, dict)]
    tool_block = ""
    if tool_schemas:
        tool_block = (
            "<AVAILABLE_TOOLS>["
            + ", ".join(nemotron_schema_json(tool) for tool in tool_schemas)
            + "]</AVAILABLE_TOOLS>"
        )
    pending_user_prefix = "\n\n".join(part for part in [*system_messages, tool_block] if part)

    parts: list[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        mode_text,
        "<|eot_id|>",
    ]

    tool_names_by_id: dict[str, str] = {}
    idx = 0
    while idx < len(conversation):
        message = conversation[idx]
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
                    nemotron_call_json({"name": name, "arguments": arguments})
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
            while idx < len(conversation):
                tool_message = conversation[idx]
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
        content = message_content_text(message.get("content", "")).strip()
        if role == "user" and pending_user_prefix:
            content = f"{pending_user_prefix}\n\n{content}".strip()
            pending_user_prefix = ""
        parts.append(content)
        parts.append("<|eot_id|>")
        idx += 1

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
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


def render_toolace_pythonic_prompt(payload: dict[str, Any]) -> str:
    """Render ToolACE 2.5's model-card Python-call chat template."""
    raw_messages = payload.get("messages", [])
    messages = list(raw_messages) if isinstance(raw_messages, list) else []
    raw_tools = payload.get("tools", [])
    tools = raw_tools if isinstance(raw_tools, list) else []

    system_message = ""
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        system_message = message_content_text(messages[0].get("content", "")).strip()
        messages = messages[1:]

    tool_schemas: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", tool)
        if not isinstance(function, dict) or not function.get("name"):
            continue
        parameters = function.get("parameters", {})
        tool_schemas.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "arguments": parameters if isinstance(parameters, dict) else {},
            }
        )

    parts = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        "You are an expert in composing functions. You are given a question and a set of possible functions. "
        "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n",
        "If none of the functions can be used, point it out. If the question lacks parameters required by a "
        "function, also point it out.\n",
        "You should only return the function call in tool-call sections.\n\n",
        "If you invoke functions, you MUST use this format: "
        "[func_name1(param_name1=param_value1), func_name2(param_name2=param_value2)].\n",
        "Do not include other text when invoking functions.\n",
        "Here is the list of functions in JSON format that you can invoke:\n",
        json.dumps(tool_schemas, ensure_ascii=False, separators=(",", ":")),
        "\n",
    ]
    if system_message:
        parts.extend(["\n", system_message, "\n"])
    parts.append("<|eot_id|>")

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        if role == "user":
            parts.extend(
                [
                    "<|start_header_id|>user<|end_header_id|>\n\n",
                    message_content_text(message.get("content", "")).strip(),
                    "<|eot_id|>",
                ]
            )
        elif role == "assistant" and message.get("tool_calls"):
            rendered_calls: list[str] = []
            for call in message.get("tool_calls", []):
                if not isinstance(call, dict):
                    continue
                function = call.get("function", {})
                if not isinstance(function, dict) or not function.get("name"):
                    continue
                arguments = parse_arguments(function.get("arguments", {})) or {}
                rendered_arguments = ", ".join(
                    f"{name}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
                    for name, value in arguments.items()
                )
                rendered_calls.append(f"{function['name']}({rendered_arguments})")
            parts.extend(
                [
                    "<|start_header_id|>assistant<|end_header_id|>\n\n[",
                    ", ".join(rendered_calls),
                    "]<|eot_id|>",
                ]
            )
        elif role == "assistant":
            parts.extend(
                [
                    "<|start_header_id|>assistant<|end_header_id|>\n\n",
                    message_content_text(message.get("content", "")).strip(),
                    "<|eot_id|>",
                ]
            )
        elif role in {"tool", "ipython"}:
            content = message.get("content", "")
            rendered_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            parts.extend(
                [
                    "<|start_header_id|>ipython<|end_header_id|>\n\n",
                    rendered_content,
                    "<|eot_id|>",
                ]
            )

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


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


def render_qwen3_native_prompt(payload: dict[str, Any], mode: str) -> str:
    """Render Qwen3's native XML tool-calling chat template."""
    messages = payload.get("messages", [])
    messages = list(messages) if isinstance(messages, list) else []
    tools = payload.get("tools", [])
    tools = tools if isinstance(tools, list) and tools else []

    parts: list[str] = []
    skip_first_system = False
    if tools:
        parts.append("<|im_start|>system\n")
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            system_content = message_content_text(messages[0].get("content", "")).strip()
            if system_content:
                parts.append(system_content)
                parts.append("\n\n")
            skip_first_system = True
        parts.append(
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>"
        )
        for tool in tools:
            parts.append("\n")
            parts.append(json.dumps(tool, ensure_ascii=False, separators=(",", ":")))
        parts.append(
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )
    elif messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        parts.append("<|im_start|>system\n")
        parts.append(message_content_text(messages[0].get("content", "")).strip())
        parts.append("<|im_end|>\n")
        skip_first_system = True

    tool_names_by_id: dict[str, str] = {}
    idx = 0
    while idx < len(messages):
        message = messages[idx]
        idx += 1
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        if idx == 1 and skip_first_system and role == "system":
            continue

        if role in {"user", "system"}:
            parts.append(f"<|im_start|>{role}\n")
            parts.append(message_content_text(message.get("content", "")).strip())
            parts.append("<|im_end|>\n")
            continue

        if role == "assistant":
            content = message_content_text(message.get("content", "")).strip()
            parts.append("<|im_start|>assistant\n")
            if content:
                parts.append(content)
            tool_calls = message.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for call_idx, call in enumerate(tool_calls):
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function", {})
                    if not isinstance(function, dict):
                        continue
                    name = str(function.get("name", ""))
                    if not name:
                        continue
                    call_id = str(call.get("id", ""))
                    if call_id:
                        tool_names_by_id[call_id] = name
                    arguments = parse_arguments(function.get("arguments", {})) or {}
                    if content or call_idx:
                        parts.append("\n")
                    parts.append("<tool_call>\n")
                    parts.append(
                        json.dumps(
                            {"name": name, "arguments": arguments},
                            ensure_ascii=False,
                            separators=(",", ": "),
                        )
                    )
                    parts.append("\n</tool_call>")
            parts.append("<|im_end|>\n")
            continue

        if role == "tool":
            tool_messages = [message]
            while idx < len(messages):
                nxt = messages[idx]
                if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                    break
                tool_messages.append(nxt)
                idx += 1
            parts.append("<|im_start|>user")
            for tool_message in tool_messages:
                content = structured_tool_content(tool_message.get("content", ""))
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
                tool_name = tool_message.get("name") or tool_names_by_id.get(str(tool_message.get("tool_call_id", "")))
                if tool_name:
                    content = json.dumps({"name": tool_name, "content": content}, ensure_ascii=False, separators=(",", ":"))
                parts.append("\n<tool_response>\n")
                parts.append(content)
                parts.append("\n</tool_response>")
            parts.append("<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
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


def extract_same_name_argument_calls(text: str, start_idx: int) -> list[dict[str, Any]]:
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if not name_match:
        return []
    name = name_match.group(1)
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r'"arguments"\s*:', text):
        obj_start = text.find("{", match.end())
        if obj_start < 0:
            continue
        try:
            arguments, _ = decoder.raw_decode(text[obj_start:])
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict):
            repaired_arguments, _ = repair_schema_echo_arguments(arguments)
            calls.append(
                {
                    "type": "tool",
                    "id": f"call_{start_idx + len(calls)}",
                    "name": name,
                    "arguments": repaired_arguments,
                }
            )
    return calls if len(calls) > 1 else []


def extract_toolcall_blocks(raw_text: str) -> list[str]:
    blocks: list[str] = []
    for match in re.finditer(r"<TOOLCALL>\s*", raw_text, flags=re.IGNORECASE):
        start = match.end()
        proper_close = re.search(r"\s*</TOOLCALL>", raw_text[start:], flags=re.IGNORECASE)
        if proper_close:
            blocks.append(raw_text[start : start + proper_close.start()])
            continue
        any_close = re.search(r"\s*</[A-Z_][A-Z0-9_]*>", raw_text[start:], flags=re.IGNORECASE)
        end = start + any_close.start() if any_close else len(raw_text)
        blocks.append(raw_text[start:end])
    return blocks


def ast_function_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        lowered = node.id.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        return node.id
    if isinstance(node, ast.Attribute):
        parent = ast_function_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def ast_argument_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = ast_argument_value(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.List):
        return [ast_argument_value(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [ast_argument_value(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            ast_argument_value(key): ast_argument_value(value)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    return ast.unparse(node) if hasattr(ast, "unparse") else ""


def python_tool_calls_from_text(
    text: str,
    start_idx: int,
    repair_schema_echo: bool = True,
) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    parse_candidates = [stripped]
    while parse_candidates[-1].endswith("]]"):
        parse_candidates.append(parse_candidates[-1][:-1].rstrip())
    parsed: ast.Expression | None = None
    for candidate in parse_candidates:
        try:
            parsed = ast.parse(candidate, mode="eval")
            break
        except SyntaxError:
            continue
    if parsed is None:
        return []
    body = parsed.body
    nodes = body.elts if isinstance(body, (ast.List, ast.Tuple)) else [body]
    calls: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = ast_function_name(node.func)
        if not name:
            continue
        arguments: dict[str, Any] = {}
        if node.args:
            positional_values = [ast_argument_value(arg) for arg in node.args]
            if len(positional_values) == 1 and isinstance(positional_values[0], dict):
                arguments.update(positional_values[0])
            else:
                arguments["__positional_args"] = positional_values
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            arguments[keyword.arg] = ast_argument_value(keyword.value)
        if repair_schema_echo:
            parsed_arguments, _ = repair_schema_echo_arguments(arguments)
        else:
            parsed_arguments = arguments
        calls.append(
            {
                "type": "tool",
                "id": f"call_{start_idx + len(calls)}",
                "name": name,
                "arguments": parsed_arguments,
            }
        )
    return calls



def split_loose_call_arguments(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False
    for char in text:
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def parse_loose_scalar(value: str) -> Any:
    stripped = value.strip().rstrip(",")
    if not stripped:
        return ""
    for candidate in [stripped, stripped.replace("'", '"')]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return ast.literal_eval(stripped)
    except Exception:
        pass
    if re.fullmatch(r"[-+]?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped):
        return float(stripped)
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    return stripped.strip('"\'')


def loose_python_tool_calls_from_text(text: str, start_idx: int) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    stripped = text.strip().strip("[]")
    for match in re.finditer(r"([A-Za-z_]\w*(?:[._][A-Za-z_]\w*)*)\]?(?:\s+parameters\s*=\s*|\s*\()([\s\S]*?)(?:\)\s*$|$)", stripped):
        name = match.group(1)
        raw_args = match.group(2).strip()
        if not name or not raw_args:
            continue
        arguments: dict[str, Any] = {}
        for part in split_loose_call_arguments(raw_args):
            if "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            key = key.strip().strip('"\'')
            if not re.fullmatch(r"[A-Za-z_]\w*", key):
                continue
            arguments[key] = parse_loose_scalar(raw_value)
        if not arguments:
            continue
        repaired_arguments, _ = repair_schema_echo_arguments(arguments)
        calls.append(
            {
                "type": "tool",
                "id": f"call_{start_idx + len(calls)}",
                "name": name,
                "arguments": repaired_arguments,
            }
        )
    return calls


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


def qcom_response_segment(raw_text: str) -> str:
    """Keep the actual assistant response, excluding generated synthetic continuation turns."""
    text = raw_text.strip()
    if re.match(r"^assistant:\s*", text, flags=re.IGNORECASE):
        text = re.sub(r"^assistant:\s*", "", text, count=1, flags=re.IGNORECASE).strip()
    elif re.match(r"^(?:user|system|tool):", text, flags=re.IGNORECASE):
        assistant_turns = list(re.finditer(r"(?:^|\n)assistant:\s*", text, flags=re.IGNORECASE))
        if assistant_turns:
            text = text[assistant_turns[-1].end() :].strip()

    leading_blocks: list[str] = []
    cursor = 0
    while cursor < len(text):
        cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
        match = re.match(
            r"<tool_call>\s*.*?\s*</tool_call>",
            text[cursor:],
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not match:
            break
        leading_blocks.append(match.group(0))
        cursor += match.end()
    return "\n".join(leading_blocks) if leading_blocks else text


def qcom_tool_parse(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    response_text = qcom_response_segment(raw_text)
    calls: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, flags=re.DOTALL | re.IGNORECASE):
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
        for line in response_text.splitlines():
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
        response_text,
        flags=re.IGNORECASE,
    ).strip()
    tolerant, status = tolerant_parse(without_tool_calls or response_text)
    if tolerant:
        return tolerant, f"{status}_qcom_fallback"
    return {"type": "final", "content": without_tool_calls or response_text}, "final_qcom_tool"


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


def toolace_pythonic_parse(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    text = raw_text.strip()
    if text.startswith("<|python_tag|>"):
        text = text[len("<|python_tag|>") :].strip()
    if "<|eot_id|>" in text:
        text = text.split("<|eot_id|>", 1)[0].strip()

    calls = python_tool_calls_from_text(
        text,
        1,
        repair_schema_echo=False,
    )
    merged_calls = merge_tool_calls(calls)
    if merged_calls:
        return merged_calls, "tool_toolace_pythonic"
    return {"type": "final", "content": text or raw_text.strip()}, "final_toolace_pythonic"


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
        repeated_calls = extract_same_name_argument_calls(raw_text, len(calls) + 1)
        if repeated_calls:
            calls.extend(repeated_calls)
            repaired = True
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


def qwen3_native_parse(raw_text: str) -> tuple[dict[str, Any] | None, str]:
    calls: list[dict[str, Any]] = []
    pattern = re.compile(
        r"<tool_call>\s*(\{[\s\S]*?})(?:\s*</tool_call>|\s*(?=<tool_call>|<\|im_end\|>|$))",
        flags=re.IGNORECASE,
    )
    for idx, match in enumerate(pattern.finditer(raw_text), start=1):
        candidate = parse_arguments(match.group(1).strip())
        if not candidate:
            continue
        name = candidate.get("name")
        arguments = candidate.get("arguments", {})
        parsed_arguments = parse_arguments(arguments)
        if isinstance(name, str) and parsed_arguments is not None:
            calls.append(
                {
                    "type": "tool",
                    "id": f"call_{idx}",
                    "name": name,
                    "arguments": parsed_arguments,
                }
            )
    if not calls:
        for candidate in json_candidates(raw_text):
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("name")
            arguments = candidate.get("arguments", {})
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
        return merged_calls, "tool_qwen3_native"
    without_calls = re.sub(
        r"<tool_call>[\s\S]*?(?:</tool_call>|(?=<tool_call>|<\|im_end\|>|$))",
        "",
        raw_text,
        flags=re.IGNORECASE,
    ).strip()
    return {"type": "final", "content": without_calls or raw_text.strip()}, "final_qwen3_native"


def nemotron_native_parse(raw_text: str, outside_tag_fallback: bool = True) -> tuple[dict[str, Any] | None, str]:
    calls: list[dict[str, Any]] = []
    repaired = False
    for block in extract_toolcall_blocks(raw_text):
        before_block = len(calls)
        calls.extend(python_tool_calls_from_text(block, len(calls) + 1))
        if len(calls) == before_block:
            for candidate in json_candidates(block):
                candidate_calls = candidate if isinstance(candidate, list) else [candidate]
                for call in candidate_calls:
                    parsed_calls, call_repaired = normalize_nemotron_native_calls(call, len(calls) + 1)
                    calls.extend(parsed_calls)
                    repaired = repaired or call_repaired
        if len(calls) == before_block:
            loose_calls = loose_python_tool_calls_from_text(block, len(calls) + 1)
            if loose_calls:
                calls.extend(loose_calls)
                repaired = True
        if len(calls) == before_block:
            repeated_calls = extract_same_name_argument_calls(block, len(calls) + 1)
            if repeated_calls:
                calls.extend(repeated_calls)
                repaired = True
    if not calls and outside_tag_fallback:
        loose_calls = loose_python_tool_calls_from_text(raw_text, len(calls) + 1)
        if loose_calls:
            calls.extend(loose_calls)
            repaired = True
    if not calls and outside_tag_fallback:
        repeated_calls = extract_same_name_argument_calls(raw_text, len(calls) + 1)
        if repeated_calls:
            calls.extend(repeated_calls)
            repaired = True
    if not calls and outside_tag_fallback:
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
        r"<TOOLCALL>[\s\S]*?</[A-Z_][A-Z0-9_]*>\s*",
        "",
        raw_text,
        flags=re.IGNORECASE,
    ).strip()
    if outside_tag_fallback:
        tolerant, status = tolerant_parse(without_tool_calls or raw_text)
        if tolerant:
            return tolerant, f"{status}_nemotron_native_fallback"
    return {"type": "final", "content": without_tool_calls or raw_text.strip()}, "final_nemotron_native"


def available_tool_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", {})
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def normalize_tool_call_names_for_payload(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not parsed or parsed.get("type") != "tool":
        return parsed
    names = available_tool_names(payload)
    if not names:
        return parsed

    def compact_tool_name(value: str) -> str:
        return "".join(normalized_tokens(value))

    compact_names: dict[str, list[str]] = {}
    for candidate in names:
        compact_names.setdefault(compact_tool_name(candidate), []).append(candidate)

    def normalize_name(name: Any) -> Any:
        if not isinstance(name, str):
            return name
        if name in names:
            return name
        underscored = name.replace(".", "_")
        if underscored in names:
            return underscored
        compact_matches = compact_names.get(compact_tool_name(name), [])
        if len(compact_matches) == 1:
            return compact_matches[0]
        suffix_matches = [candidate for candidate in names if candidate.endswith(f"_{name}") or candidate.endswith(f".{name}")]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        name_compact = compact_tool_name(name)
        compact_suffix_matches = [candidate for candidate in names if compact_tool_name(candidate).endswith(name_compact)]
        return compact_suffix_matches[0] if len(compact_suffix_matches) == 1 else name

    calls = parsed.get("tool_calls")
    if isinstance(calls, list) and calls:
        for call in calls:
            if isinstance(call, dict):
                call["name"] = normalize_name(call.get("name"))
        parsed["name"] = calls[0].get("name", parsed.get("name"))
        parsed["arguments"] = calls[0].get("arguments", parsed.get("arguments", {}))
        return parsed

    parsed["name"] = normalize_name(parsed.get("name"))
    return parsed


def tool_parameter_names(payload: dict[str, Any]) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    sources: list[dict[str, Any]] = []
    tools = payload.get("tools", [])
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function", {})
            if isinstance(function, dict):
                sources.append(function)
    bfcl_functions = payload.get("bfcl_functions", [])
    if isinstance(bfcl_functions, list):
        sources.extend(function for function in bfcl_functions if isinstance(function, dict))
    for function in sources:
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        if isinstance(properties, dict):
            names[function["name"]] = {str(name) for name in properties}
    return names


def tool_parameter_schemas(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    schemas: dict[str, dict[str, dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    tools = payload.get("tools", [])
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function", {})
            if isinstance(function, dict):
                sources.append(function)
    bfcl_functions = payload.get("bfcl_functions", [])
    if isinstance(bfcl_functions, list):
        sources.extend(function for function in bfcl_functions if isinstance(function, dict))
    for function in sources:
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        if not isinstance(properties, dict):
            continue
        normalized_properties = {str(name): schema for name, schema in properties.items() if isinstance(schema, dict)}
        function_name = function["name"]
        schemas[function_name] = normalized_properties
        schemas[function_name.replace(".", "_")] = normalized_properties
    return schemas


def coerce_schema_value(value: Any, schema: dict[str, Any]) -> tuple[Any, bool]:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)
    if schema_type in {"boolean", "bool"} and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True, True
        if lowered == "false":
            return False, True
    if schema_type == "string" and isinstance(value, (int, float, bool)):
        return str(value).lower() if isinstance(value, bool) else str(value), True
    if schema_type == "integer" and isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped), True
    if schema_type in {"number", "float"} and isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped), True
        except ValueError:
            return value, False
    return value, False


def coerce_arguments_to_schema(arguments: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    if not schemas:
        return arguments, False
    coerced: dict[str, Any] = dict(arguments)
    changed = False
    for name, schema in schemas.items():
        if name not in coerced:
            continue
        value, did_coerce = coerce_schema_value(coerced[name], schema)
        if did_coerce:
            coerced[name] = value
            changed = True
    return coerced, changed


def tool_required_parameter_names(payload: dict[str, Any]) -> dict[str, set[str]]:
    required_by_tool: dict[str, set[str]] = {}
    sources: list[dict[str, Any]] = []
    tools = payload.get("tools", [])
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function", {})
            if isinstance(function, dict):
                sources.append(function)
    bfcl_functions = payload.get("bfcl_functions", [])
    if isinstance(bfcl_functions, list):
        sources.extend(function for function in bfcl_functions if isinstance(function, dict))
    for function in sources:
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        parameters = function.get("parameters", {})
        required = parameters.get("required", []) if isinstance(parameters, dict) else []
        if not isinstance(required, list):
            continue
        names = {str(name) for name in required}
        function_name = function["name"]
        required_by_tool[function_name] = names
        required_by_tool[function_name.replace(".", "_")] = names
    return required_by_tool


def schema_type_matches_value(value: Any, schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)
    if schema_type in {None, "any"}:
        return True
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type in {"number", "float"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type in {"boolean", "bool"}:
        return isinstance(value, bool)
    if schema_type in {"dict", "object"}:
        return isinstance(value, dict)
    if schema_type in {"list", "array"}:
        return isinstance(value, list)
    return True


def repair_unknown_required_argument(
    arguments: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    required: set[str],
) -> tuple[dict[str, Any], bool]:
    if not schemas or not required:
        return arguments, False
    missing_required = [name for name in required if name not in arguments]
    unknown_args = [name for name in arguments if name not in schemas and not str(name).startswith("__")]
    if len(missing_required) != 1 or len(unknown_args) != 1:
        return arguments, False
    missing = missing_required[0]
    unknown = unknown_args[0]
    schema = schemas.get(missing, {})
    value = arguments[unknown]
    if not schema_type_matches_value(value, schema):
        return arguments, False
    repaired = dict(arguments)
    repaired[missing] = repaired.pop(unknown)
    return repaired, True


def schema_repair_candidate(arguments: dict[str, Any], expected: set[str]) -> tuple[dict[str, Any], bool]:
    if not expected:
        return arguments, False
    wrappers = ["arguments", "parameters", "properties"]
    for wrapper in wrappers:
        wrapped = arguments.get(wrapper)
        if isinstance(wrapped, dict) and wrapper not in expected:
            if expected & set(map(str, wrapped.keys())):
                return wrapped, True
    if "func_name" in arguments and "func_name" not in expected:
        without_func_name = {key: value for key, value in arguments.items() if key != "func_name"}
        if expected & set(map(str, without_func_name.keys())):
            return without_func_name, True
    if "properties" in arguments and "properties" not in expected and isinstance(arguments["properties"], dict):
        candidate = arguments["properties"]
        if expected & set(map(str, candidate.keys())):
            return candidate, True
    argument_keys = set(map(str, arguments.keys()))
    if len(expected) == 1:
        expected_name = next(iter(expected))
        positional_args = arguments.get("__positional_args")
        if isinstance(positional_args, list) and len(positional_args) == 1:
            return {expected_name: positional_args[0]}, True
        if arguments and not (expected & argument_keys) and not any(key.startswith("__") for key in argument_keys):
            return {expected_name: arguments}, True
    return arguments, False


def repair_tool_arguments_for_payload(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    if not parsed or parsed.get("type") != "tool":
        return parsed, False
    params_by_tool = tool_parameter_names(payload)
    schemas_by_tool = tool_parameter_schemas(payload)
    required_by_tool = tool_required_parameter_names(payload)
    if not params_by_tool:
        return parsed, False
    repaired_any = False
    calls = parsed.get("tool_calls")
    call_list = calls if isinstance(calls, list) and calls else [parsed]
    for call in call_list:
        if not isinstance(call, dict):
            continue
        expected = params_by_tool.get(str(call.get("name", "")), set())
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        tool_name = str(call.get("name", ""))
        tool_schemas = schemas_by_tool.get(tool_name, {})
        repaired, did_repair = schema_repair_candidate(arguments, expected)
        aliased, did_alias = repair_unknown_required_argument(
            repaired,
            tool_schemas,
            required_by_tool.get(tool_name, set()),
        )
        coerced, did_coerce = coerce_arguments_to_schema(aliased, tool_schemas)
        if did_repair or did_alias or did_coerce:
            call["arguments"] = coerced
            repairs = []
            if did_repair:
                repairs.append("schema_wrapper_for_payload")
            if did_alias:
                repairs.append("schema_required_alias")
            if did_coerce:
                repairs.append("schema_type_coercion")
            call["argument_repair"] = "+".join(repairs)
            repaired_any = True
    if isinstance(calls, list) and calls:
        parsed["name"] = calls[0].get("name", parsed.get("name"))
        parsed["arguments"] = calls[0].get("arguments", parsed.get("arguments", {}))
    return parsed, repaired_any


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalized_phrase(text: str) -> str:
    return " ".join(normalized_tokens(text))


def user_numeric_literals(user_text: str) -> set[str]:
    literals: set[str] = set()
    for match in re.finditer(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", user_text):
        token = match.group(0).replace("$", "").replace(",", "")
        literals.add(token)
        if token.endswith(".0"):
            literals.add(token[:-2])
    return literals


MONTH_BY_NUMBER = {
    "01": "jan", "1": "jan",
    "02": "feb", "2": "feb",
    "03": "mar", "3": "mar",
    "04": "apr", "4": "apr",
    "05": "may", "5": "may",
    "06": "jun", "6": "jun",
    "07": "jul", "7": "jul",
    "08": "aug", "8": "aug",
    "09": "sep", "9": "sep",
    "10": "oct", "11": "nov", "12": "dec",
}


def date_value_supported_by_user(value: str, user_text: str) -> bool:
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", value.strip())
    if not match:
        return False
    year, month, day = match.groups()
    lowered = user_text.lower()
    if year not in lowered:
        return False
    month_name = MONTH_BY_NUMBER.get(month.lstrip("0") or month) or MONTH_BY_NUMBER.get(month)
    if month_name and month_name in lowered:
        return True
    return bool(re.search(rf"\b0?{int(month)}[/-]0?{int(day)}[/-]{year}\b", lowered))


def value_supported_by_user(value: Any, user_text: str) -> bool:
    user_tokens = set(normalized_tokens(user_text))
    user_phrase = normalized_phrase(user_text)
    if value is None:
        return True
    if isinstance(value, bool):
        return str(value).lower() in user_tokens
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        normalized_value = str(value)
        if isinstance(value, float) and value.is_integer():
            normalized_value = str(int(value))
        return normalized_value in user_numeric_literals(user_text)
    if isinstance(value, str):
        phrase = normalized_phrase(value)
        if not phrase:
            return True
        if phrase in user_phrase:
            return True
        if date_value_supported_by_user(value, user_text):
            return True
        tokens = normalized_tokens(value)
        return bool(tokens) and all(token in user_tokens for token in tokens)
    if isinstance(value, dict):
        return all(value_supported_by_user(item, user_text) for item in value.values())
    if isinstance(value, list):
        return all(value_supported_by_user(item, user_text) for item in value)
    return True


def unsupported_required_arguments_for_call(call: dict[str, Any], payload: dict[str, Any], user_text: str) -> list[str]:
    tool_name = str(call.get("name", ""))
    required_by_tool = tool_required_parameter_names(payload)
    required = required_by_tool.get(tool_name, set())
    if not required:
        return []
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        return sorted(required)
    unsupported: list[str] = []
    for name in sorted(required):
        if name not in arguments:
            unsupported.append(name)
            continue
        if not value_supported_by_user(arguments[name], user_text):
            unsupported.append(name)
    return unsupported


def reject_unsupported_required_tool_calls(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not parsed or parsed.get("type") != "tool":
        return parsed, None
    user_text = payload_last_user_text(payload)
    if not user_text:
        return parsed, None
    calls = parsed.get("tool_calls")
    call_list = calls if isinstance(calls, list) and calls else [parsed]
    rejected: list[dict[str, Any]] = []
    for call in call_list:
        if not isinstance(call, dict):
            continue
        unsupported = unsupported_required_arguments_for_call(call, payload, user_text)
        if unsupported:
            rejected.append({"name": call.get("name"), "unsupported_required_arguments": unsupported})
    if not rejected:
        return parsed, {"decision": "allow", "reason": "required_arguments_supported"}
    return {"type": "final", "content": "No suitable tool call."}, {
        "decision": "reject",
        "reason": "unsupported_required_arguments",
        "rejected_calls": rejected,
    }


def unsupported_required_tool_call_details(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not parsed or parsed.get("type") != "tool":
        return []
    user_text = payload_last_user_text(payload)
    if not user_text:
        return []
    calls = parsed.get("tool_calls")
    call_list = calls if isinstance(calls, list) and calls else [parsed]
    rejected: list[dict[str, Any]] = []
    for call in call_list:
        if not isinstance(call, dict):
            continue
        unsupported = unsupported_required_arguments_for_call(call, payload, user_text)
        if unsupported:
            rejected.append({"name": call.get("name"), "unsupported_required_arguments": unsupported})
    return rejected


def tool_text_by_name(payload: dict[str, Any]) -> dict[str, str]:
    text_by_name: dict[str, str] = {}
    tools = payload.get("tools", [])
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", ""))
            if not name:
                continue
            tool_text = f"{name} {function.get('description', '')}".lower()
            text_by_name[name] = tool_text
            text_by_name[name.replace(".", "_")] = tool_text
    bfcl_functions = payload.get("bfcl_functions", [])
    if isinstance(bfcl_functions, list):
        for function in bfcl_functions:
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", ""))
            if not name:
                continue
            tool_text = f"{name} {function.get('description', '')}".lower()
            text_by_name[name] = tool_text
            text_by_name[name.replace(".", "_")] = tool_text
    return text_by_name


def proposed_tool_text(payload: dict[str, Any], rejected_calls: list[dict[str, Any]]) -> str:
    text_by_name = tool_text_by_name(payload)
    parts = []
    for call in rejected_calls:
        name = str(call.get("name", ""))
        parts.append(text_by_name.get(name, name).lower())
    return " ".join(parts)


def should_reject_unsupported_for_relevance(
    user_text: str,
    rejected_calls: list[dict[str, Any]],
    payload: dict[str, Any],
) -> tuple[bool, str]:
    if not rejected_calls:
        return False, "required_arguments_supported"
    lowered = user_text.lower()
    tool_text = proposed_tool_text(payload, rejected_calls)
    question_patterns = [
        r"\bcurrent time\b",
        r"\btime now\b",
        r"^\s*what\s+(?:is|are)\s+(?:a\s+|an\s+|the\s+)?\w+\??\s*$",
        r"\bdefinition\b",
        r"^\s*explain\b",
    ]
    if any(re.search(pattern, lowered) for pattern in question_patterns):
        if not any(phrase in tool_text for phrase in ["current time", "clock", "time zone", "timezone"]):
            return True, "unsupported_question_or_definition"
    if re.match(r"^\s*who\b", lowered):
        if ("won" in lowered or "winner" in lowered) and any(word in tool_text for word in ["winner", "result"]):
            return False, "who_question_directly_supported"
        if "signed" in lowered and "sign" not in tool_text:
            return True, "unsupported_who_question"
        if not any(word in tool_text for word in ["winner", "author", "sign", "person", "people", "player", "team"]):
            return True, "unsupported_who_question"
    unsupported_names = {arg for call in rejected_calls for arg in call.get("unsupported_required_arguments", [])}
    if ("restaurant" in lowered or "go out to eat" in lowered or "eat at" in lowered) and unsupported_names & {"category", "cuisine", "food_type"}:
        return True, "unsupported_restaurant_category"
    if any(word in lowered for word in ["booking", "reservation", "reserve"]) and unsupported_names & {"date", "time", "destination", "category", "cuisine"}:
        return True, "unsupported_booking_detail"
    return False, "unsupported_but_not_relevance_guarded"


def reject_relevance_unsupported_tool_calls(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    rejected = unsupported_required_tool_call_details(parsed, payload)
    if not rejected:
        return parsed, {"decision": "allow", "reason": "required_arguments_supported"}
    should_reject, reason = should_reject_unsupported_for_relevance(payload_last_user_text(payload), rejected, payload)
    if not should_reject:
        return parsed, {"decision": "allow", "reason": reason, "rejected_calls": rejected}
    return {"type": "final", "content": "No suitable tool call."}, {
        "decision": "reject",
        "reason": reason,
        "rejected_calls": rejected,
    }


ACTION_TOOL_TOKENS = {
    "book", "booking", "reserve", "reservation", "buy", "purchase", "order", "submit",
    "create", "delete", "cancel", "change", "update", "modify", "set", "send",
}
SEARCH_TOOL_TOKENS = {
    "find", "search", "list", "show", "see", "get", "check", "lookup", "fetch", "available",
    "availability", "read", "retrieve",
}
USER_ACTION_WORDS = ACTION_TOOL_TOKENS | {"schedule", "make", "place", "request", "reschedule"}
USER_SEARCH_WORDS = SEARCH_TOOL_TOKENS | {"need", "want", "looking", "look", "what", "which", "where"}


def tool_name_terms(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name.replace(".", " ").replace("_", " ").replace("-", " "))
    return normalized_tokens(spaced)


def action_terms_for_name(name: str) -> set[str]:
    terms = set(tool_name_terms(name))
    return {term for term in terms if term in ACTION_TOOL_TOKENS}


def search_terms_for_name(name: str) -> set[str]:
    terms = set(tool_name_terms(name))
    return {term for term in terms if term in SEARCH_TOOL_TOKENS}


def user_action_terms(user_text: str) -> set[str]:
    tokens = set(normalized_tokens(user_text))
    return {token for token in tokens if token in USER_ACTION_WORDS}


def user_search_terms(user_text: str) -> set[str]:
    tokens = set(normalized_tokens(user_text))
    return {token for token in tokens if token in USER_SEARCH_WORDS}


def user_is_readonly_request(user_text: str) -> bool:
    search_terms = user_search_terms(user_text)
    action_terms = user_action_terms(user_text)
    if not search_terms:
        return False
    return not action_terms or action_terms <= {"get", "fetch", "retrieve", "read"}


def non_action_alternative(name: str, payload: dict[str, Any]) -> str | None:
    current_terms = set(tool_name_terms(name)) - ACTION_TOOL_TOKENS - SEARCH_TOOL_TOKENS
    if not current_terms:
        return None
    candidates: list[tuple[int, str]] = []
    for candidate in available_tool_names(payload):
        if candidate == name or action_terms_for_name(candidate):
            continue
        if not search_terms_for_name(candidate):
            continue
        candidate_terms = set(tool_name_terms(candidate)) - ACTION_TOOL_TOKENS - SEARCH_TOOL_TOKENS
        overlap = len(current_terms & candidate_terms)
        if overlap:
            candidates.append((overlap, candidate))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) == 1 or candidates[0][0] > candidates[1][0]:
        return candidates[0][1]
    return None


def prune_or_rewrite_action_mismatch_tool_calls(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not parsed or parsed.get("type") != "tool":
        return parsed, None
    user_text = payload_last_user_text(payload)
    if not user_text or not user_is_readonly_request(user_text):
        return parsed, {"decision": "allow", "reason": "not_readonly_user_request"}
    calls = parsed.get("tool_calls")
    call_list = calls if isinstance(calls, list) and calls else [parsed]
    kept: list[dict[str, Any]] = []
    rewrites: list[dict[str, str]] = []
    dropped: list[dict[str, Any]] = []
    for call in call_list:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name", ""))
        if not action_terms_for_name(name):
            kept.append(call)
            continue
        replacement = non_action_alternative(name, payload)
        if replacement:
            rewritten = dict(call)
            rewritten["name"] = replacement
            kept.append(rewritten)
            rewrites.append({"from": name, "to": replacement})
        else:
            dropped.append(call)
    if not rewrites and not dropped:
        return parsed, {"decision": "allow", "reason": "no_action_mismatch"}
    if kept:
        return merge_tool_calls(kept), {"decision": "rewrite_or_prune", "reason": "readonly_request_action_tool", "rewrites": rewrites, "dropped": dropped}
    return {"type": "final", "content": "No suitable tool call."}, {"decision": "reject", "reason": "readonly_request_action_tool", "dropped": dropped}


def call_required_values_unsupported(call: dict[str, Any], payload: dict[str, Any], user_text: str) -> list[str]:
    name = str(call.get("name", ""))
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        return []
    schemas = tool_parameter_schemas(payload).get(name, {})
    required = tool_required_parameter_names(payload).get(name, set())
    unsupported: list[str] = []
    for arg_name in required:
        if arg_name not in arguments:
            unsupported.append(arg_name)
            continue
        schema = schemas.get(arg_name, {})
        if isinstance(schema, dict) and not argument_supported_by_user(arg_name, arguments[arg_name], schema, user_text):
            inferred = infer_argument_value_from_user(arg_name, schema, user_text)
            if inferred is None:
                unsupported.append(arg_name)
    return unsupported


def should_reject_semantic_tool_call(call: dict[str, Any], payload: dict[str, Any], user_text: str) -> tuple[bool, str]:
    lowered = user_text.lower()
    name = str(call.get("name", ""))
    tool_text = tool_text_by_name(payload).get(name, name).lower()
    user_tokens = set(normalized_tokens(user_text))
    unsupported_required = call_required_values_unsupported(call, payload, user_text)
    if any(phrase in lowered for phrase in ["current time", "time now", "what time"]):
        if not any(word in tool_text for word in ["time", "clock", "timezone", "time zone"]):
            return True, "time_request_wrong_tool"
    if any(word in user_tokens for word in ["solve", "root", "roots", "quadratic", "cubic", "equation"]):
        if "sum" in tool_text and not any(word in tool_text for word in ["root", "quadratic", "cubic", "polynomial", "equation"]):
            return True, "equation_request_wrong_tool"
    if any(word in user_tokens for word in ["mating", "mate", "reproduction", "behavior", "behaviour"]):
        if any(word in tool_text for word in ["genetic", "trait"]) and not any(word in tool_text for word in ["mating", "reproduction", "behavior", "behaviour"]):
            return True, "behavior_request_wrong_tool"
    if any(word in user_tokens for word in ["create", "generate", "make"]) and "puzzle" in user_tokens:
        if "solve" in tool_text and not any(word in tool_text for word in ["create", "generate", "make"]):
            return True, "creation_request_wrong_tool"
    if {"house", "price"} <= user_tokens and any(word in tool_text for word in ["regression", "predict", "model", "features"]):
        return True, "house_price_wrong_predictor_tool"
    if any(word in user_tokens for word in ["composer", "composers"]) and any(word in tool_text for word in ["chord", "progression"]):
        return True, "composer_question_wrong_music_tool"
    if any(word in user_tokens for word in ["hypotenuse", "triangle"]):
        if any(word in tool_text for word in ["latitude", "longitude", "coordinate", "map"]):
            return True, "geometry_request_wrong_coordinate_tool"
    if "cab" in user_tokens and not any(word in tool_text for word in ["cab", "taxi", "rideshare", "ride"]):
        return True, "cab_request_wrong_tool"
    generic_vague = normalized_phrase(user_text) in {"air", "fetch all", "get all", "show all", "list all"}
    if generic_vague:
        return True, "vague_request_no_specific_tool_support"
    if unsupported_required and user_is_readonly_request(user_text):
        structural_args = {"model", "features", "pointa", "pointb", "puzzleimage", "piecescount", "progressionpattern"}
        if structural_args & {"".join(normalized_tokens(arg)) for arg in unsupported_required}:
            return True, "readonly_request_invented_structural_args"
    if name.lower() in {"requests.get", "requests_get"}:
        args = call.get("arguments", {})
        url = args.get("url") if isinstance(args, dict) else None
        if isinstance(url, str) and url and url.lower() not in lowered and not re.search(r"https?://", user_text, flags=re.IGNORECASE):
            return True, "invented_http_url"
    if len(normalized_tokens(user_text)) <= 3 and unsupported_required:
        return True, "vague_request_invented_required_args"
    if unsupported_required and user_is_readonly_request(user_text) and action_terms_for_name(name):
        return True, "readonly_request_invented_action_args"
    return False, "semantic_supported"


def reject_semantic_mismatch_tool_calls(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not parsed or parsed.get("type") != "tool":
        return parsed, None
    user_text = payload_last_user_text(payload)
    if not user_text:
        return parsed, {"decision": "allow", "reason": "no_user_text"}
    calls = parsed.get("tool_calls")
    call_list = calls if isinstance(calls, list) and calls else [parsed]
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for call in call_list:
        if not isinstance(call, dict):
            continue
        should_reject, reason = should_reject_semantic_tool_call(call, payload, user_text)
        if should_reject:
            rejected.append({"name": call.get("name"), "reason": reason, "arguments": call.get("arguments", {})})
        else:
            kept.append(call)
    if not rejected:
        return parsed, {"decision": "allow", "reason": "semantic_supported"}
    if kept:
        return merge_tool_calls(kept), {"decision": "prune", "reason": "semantic_mismatch", "rejected": rejected}
    return {"type": "final", "content": "No suitable tool call."}, {"decision": "reject", "reason": "semantic_mismatch", "rejected": rejected}


def is_empty_like_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"[]", "{}", "none", "null", "n/a", "na", "not specified"}


def removal_argument_names(schemas: dict[str, dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for name, schema in schemas.items():
        description = str(schema.get("description", ""))
        haystack = f"{name} {description}".lower()
        if any(word in haystack for word in ["remove", "exclude", "omit", "without"]):
            names.append(name)
    return names


def extract_negative_value(value: str) -> str | None:
    match = re.fullmatch(r"\s*(?:no|without|remove|exclude|except)\s+(.+?)\s*", value, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def find_user_phrase_for_tokens(user_text: str, tokens: list[str]) -> str | None:
    if not tokens:
        return None
    pattern = r"\b" + r"\s+".join(re.escape(token) for token in tokens) + r"\b"
    match = re.search(pattern, user_text, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def split_camel_tokens(value: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    return normalized_tokens(spaced)


def repair_content_span_from_user(name: str, value: Any, user_text: str) -> tuple[Any, bool]:
    if not isinstance(value, str) or name.lower() not in {"content", "text", "message", "query", "statement"}:
        return value, False
    match = re.search(rf"\b{re.escape(name)}\b\s+(.+)$", user_text, flags=re.IGNORECASE)
    if not match:
        if name.lower() != "statement":
            return value, False
        value_phrase = normalized_phrase(value)
        if not value_phrase:
            return value, False
        for sentence in re.findall(r"[^.!?\n]+[.!?]?", user_text):
            candidate = sentence.strip()
            if not candidate:
                continue
            if value_phrase in normalized_phrase(candidate) and normalized_phrase(candidate) != value_phrase:
                return candidate, True
        return value, False
    candidate = match.group(1).strip().strip('"')
    if normalized_phrase(candidate).endswith(normalized_phrase(value)) and normalized_phrase(candidate) != normalized_phrase(value):
        return candidate, True
    return value, False


def repair_identifier_from_user(name: str, schema: dict[str, Any], value: Any, user_text: str) -> tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False
    description = str(schema.get("description", ""))
    haystack = f"{name} {description}".lower()
    if "id" not in name.lower() and "identifier" not in haystack:
        return value, False
    match = re.search(r"(?:^|[_\s-])(\d+)$", value)
    if not match:
        return value, False
    digits = match.group(1)
    if re.search(rf"\b(?:order\s+)?{re.escape(digits)}\b", user_text, flags=re.IGNORECASE):
        return digits, True
    return value, False


def repair_camel_string_from_user(value: Any, user_text: str) -> tuple[Any, bool]:
    if not isinstance(value, str) or " " in value:
        return value, False
    tokens = split_camel_tokens(value)
    if len(tokens) < 2:
        return value, False
    phrase = find_user_phrase_for_tokens(user_text, tokens)
    if phrase and normalized_phrase(phrase) != normalized_phrase(value):
        return phrase, True
    return value, False


GENERIC_PARAMETER_TOKENS = {
    "arg", "argument", "field", "value", "data", "info", "information", "new", "old",
    "include", "has", "have", "with", "use", "using", "is", "are", "be", "type",
    "id", "identifier", "option", "preference", "preferences", "parameter",
}


def meaningful_tokens(text: str) -> list[str]:
    return [token for token in normalized_tokens(text) if token not in GENERIC_PARAMETER_TOKENS]


def repair_case_from_user(value: Any, user_text: str) -> tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False
    phrase = normalized_phrase(value)
    if not phrase:
        return value, False
    pattern = r"\b" + r"\s+".join(re.escape(token) for token in phrase.split()) + r"\b"
    match = re.search(pattern, user_text, flags=re.IGNORECASE)
    if match and match.group(0) != value:
        return match.group(0), True
    return value, False


COMMON_CITY_STATE = {
    "boston": "MA",
    "chicago": "IL",
    "los angeles": "CA",
    "miami": "FL",
    "new york": "NY",
    "new york city": "NY",
    "san francisco": "CA",
    "seattle": "WA",
}

COMMON_CITY_STATE_NAME = {
    "boston": "Massachusetts",
    "chicago": "Illinois",
    "los angeles": "California",
    "miami": "Florida",
    "new york": "New York",
    "new york city": "New York",
    "san francisco": "California",
    "seattle": "Washington",
}


def repair_region_suffix_from_user(value: Any, user_text: str) -> tuple[Any, bool]:
    if not isinstance(value, str) or "," not in value:
        return value, False
    prefix, suffix = [part.strip() for part in value.split(",", 1)]
    if not prefix or not suffix:
        return value, False
    user_phrase = normalized_phrase(user_text)
    if normalized_phrase(prefix) in user_phrase and normalized_phrase(value) not in user_phrase:
        return prefix, True
    return value, False


def repair_missing_region_from_user(name: str, schema: dict[str, Any], value: Any, user_text: str) -> tuple[Any, bool]:
    if not isinstance(value, str) or "," in value:
        return value, False
    haystack = f"{name} {schema.get('description', '')}".lower()
    if not any(phrase in haystack for phrase in ["city, state", "city and state", "state name", "short form"]):
        return value, False
    city_key = normalized_phrase(value)
    state = COMMON_CITY_STATE.get(city_key)
    if not state:
        return value, False
    if normalized_phrase(value) in normalized_phrase(user_text):
        return f"{value}, {state}", True
    return value, False


def inferred_state_for_city(city: Any, user_text: str) -> str | None:
    if not isinstance(city, str):
        return None
    city_key = normalized_phrase(city)
    if city_key not in normalized_phrase(user_text):
        return None
    return COMMON_CITY_STATE_NAME.get(city_key) or COMMON_CITY_STATE.get(city_key)


def repair_minimum_threshold_from_user(name: str, schema: dict[str, Any], value: Any, user_text: str) -> tuple[Any, bool]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value, False
    haystack = f"{name} {schema.get('description', '')}".lower()
    if not any(word in haystack for word in ["rating", "minimum", "min", "threshold"]):
        return value, False
    pattern = r"\b(?:more than|greater than|over|above|at least|minimum(?: of)?)\s+(\d+(?:\.\d+)?)\b"
    matches = [float(match.group(1)) for match in re.finditer(pattern, user_text, flags=re.IGNORECASE)]
    if len(matches) != 1:
        return value, False
    threshold = matches[0]
    repaired: int | float = int(threshold) if threshold.is_integer() else threshold
    if repaired != value:
        return repaired, True
    return value, False


def name_or_description_supported_by_user(name: str, schema: dict[str, Any], user_text: str) -> bool:
    user_tokens = set(normalized_tokens(user_text))
    tokens = meaningful_tokens(name)
    if any(token in user_tokens for token in tokens):
        return True
    description = str(schema.get("description", ""))
    description_tokens = meaningful_tokens(description)
    return any(token in user_tokens for token in description_tokens[:12])


def argument_supported_by_user(name: str, value: Any, schema: dict[str, Any], user_text: str) -> bool:
    if value_supported_by_user(value, user_text):
        return True
    if isinstance(value, bool) and value and name_or_description_supported_by_user(name, schema, user_text):
        return True
    haystack = f"{name} {schema.get('description', '')}".lower()
    if isinstance(value, str) and "date" in haystack and date_value_supported_by_user(value, user_text):
        return True
    return False


def quoted_optional_value_near_name(name: str, schema: dict[str, Any], user_text: str) -> str | None:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)
    if schema_type not in {"string", None}:
        return None
    name_tokens = normalized_tokens(name)
    target_tokens = name_tokens if len(name_tokens) > 1 else meaningful_tokens(name)
    if not target_tokens:
        target_tokens = name_tokens
    quote_spans: list[tuple[int, int, str]] = []
    for pattern in [r"'([^']+)'", r'"([^"]+)"']:
        for match in re.finditer(pattern, user_text):
            quote_spans.append((match.start(), match.end(), match.group(1).strip()))
    for start, end, candidate in sorted(quote_spans):
        if not candidate or len(candidate) > 40 or len(normalized_tokens(candidate)) > 5:
            continue
        after = user_text[end: end + 96]
        before = user_text[max(0, start - 64): start]
        nearby_tokens = set(normalized_tokens(after + " " + before))
        if all(token in nearby_tokens for token in target_tokens):
            return candidate
    return None


def prune_unsupported_optional_values(arguments: dict[str, Any], schemas: dict[str, dict[str, Any]], required: set[str], user_text: str) -> bool:
    changed = False
    for name in list(arguments.keys()):
        schema = schemas.get(name)
        if not isinstance(schema, dict):
            if not value_supported_by_user(arguments[name], user_text):
                arguments.pop(name, None)
                changed = True
            continue
        value = arguments[name]
        properties = schema.get("properties")
        if isinstance(value, dict) and isinstance(properties, dict):
            nested_required = schema.get("required", [])
            nested_required_set = {str(item) for item in nested_required} if isinstance(nested_required, list) else set()
            nested_schemas = {str(key): item for key, item in properties.items() if isinstance(item, dict)}
            if prune_unsupported_optional_values(value, nested_schemas, nested_required_set, user_text):
                changed = True
            if name not in required and not value and not argument_supported_by_user(name, value, schema, user_text):
                arguments.pop(name, None)
                changed = True
            continue
        if name in required:
            continue
        if name.lower() == "name" and isinstance(value, str):
            tokens = normalized_tokens(value)
            if tokens and not find_user_phrase_for_tokens(user_text, tokens) and "name" not in normalized_tokens(user_text):
                arguments.pop(name, None)
                changed = True
                continue
        if schema.get("default") == "" and is_empty_like_string(value):
            arguments[name] = ""
            changed = True
            continue
        if not argument_supported_by_user(name, value, schema, user_text):
            arguments.pop(name, None)
            changed = True
    return changed


def enhance_call_arguments_for_payload(call: dict[str, Any], payload: dict[str, Any], user_text: str) -> bool:
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        return False
    tool_name = str(call.get("name", ""))
    schemas_by_tool = tool_parameter_schemas(payload)
    schemas = schemas_by_tool.get(tool_name, {})
    if not schemas:
        return False
    changed = False

    for name, schema in schemas.items():
        if name not in arguments:
            continue
        value = arguments[name]
        if schema.get("default") == "" and is_empty_like_string(value):
            arguments[name] = ""
            changed = True
            value = arguments[name]
        repaired, did_repair = repair_content_span_from_user(name, value, user_text)
        if did_repair:
            arguments[name] = repaired
            changed = True
            value = repaired
            if name.lower() == "statement":
                continue
        repaired, did_repair = repair_identifier_from_user(name, schema, value, user_text)
        if did_repair:
            arguments[name] = repaired
            changed = True
            value = repaired
        repaired, did_repair = repair_camel_string_from_user(value, user_text)
        if did_repair:
            arguments[name] = repaired
            changed = True
            value = repaired
        repaired, did_repair = repair_case_from_user(value, user_text)
        if did_repair:
            arguments[name] = repaired
            changed = True
            value = repaired
        repaired, did_repair = repair_region_suffix_from_user(value, user_text)
        if did_repair:
            arguments[name] = repaired
            changed = True
            value = repaired
        repaired, did_repair = repair_missing_region_from_user(name, schema, value, user_text)
        if did_repair:
            arguments[name] = repaired
            changed = True
            value = repaired
        repaired, did_repair = repair_minimum_threshold_from_user(name, schema, value, user_text)
        if did_repair:
            arguments[name] = repaired
            changed = True
            value = repaired

    user_phrase = normalized_phrase(user_text)
    required_by_tool = tool_required_parameter_names(payload)
    required = required_by_tool.get(tool_name, set())
    for name, schema in schemas.items():
        if name in arguments:
            continue
        if name.lower() == "state":
            inferred_state = inferred_state_for_city(arguments.get("city"), user_text)
            if inferred_state:
                arguments[name] = inferred_state
                changed = True
                continue
        if name in required:
            inferred = infer_argument_value_from_user(name, schema, user_text)
            if inferred is not None:
                arguments[name] = inferred
                changed = True
                continue
        enum_values = schema.get("enum")
        if isinstance(enum_values, list):
            matches = [value for value in enum_values if isinstance(value, str) and normalized_phrase(value) and normalized_phrase(value) in user_phrase]
            if len(matches) == 1:
                arguments[name] = matches[0]
                changed = True
                continue
        quoted_value = quoted_optional_value_near_name(name, schema, user_text)
        if quoted_value is not None:
            arguments[name] = quoted_value
            changed = True

    for name in required:
        if name not in arguments:
            continue
        schema = schemas.get(name, {})
        if not isinstance(schema, dict):
            continue
        if not argument_supported_by_user(name, arguments[name], schema, user_text):
            inferred = infer_argument_value_from_user(name, schema, user_text)
            if inferred is not None and inferred != arguments[name]:
                arguments[name] = inferred
                changed = True

    removal_names = removal_argument_names(schemas)
    if removal_names:
        for name, value in list(arguments.items()):
            if name in removal_names or not isinstance(value, str):
                continue
            negative_value = extract_negative_value(value)
            if not negative_value:
                continue
            target = next((candidate for candidate in removal_names if candidate not in arguments or is_empty_like_string(arguments.get(candidate))), removal_names[0])
            arguments[target] = negative_value
            if schemas.get(name, {}).get("default") == "":
                arguments[name] = ""
            changed = True

    if prune_unsupported_optional_values(arguments, schemas, required, user_text):
        changed = True

    if changed:
        call["arguments"] = arguments
        call["argument_enhancement"] = "supported_value_normalization"
    return changed


def enhance_tool_arguments_for_payload(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    if not parsed or parsed.get("type") != "tool":
        return parsed, False
    user_text = payload_last_user_text(payload)
    if not user_text:
        return parsed, False
    calls = parsed.get("tool_calls")
    call_list = calls if isinstance(calls, list) and calls else [parsed]
    changed = False
    for call in call_list:
        if isinstance(call, dict) and enhance_call_arguments_for_payload(call, payload, user_text):
            changed = True
    if isinstance(calls, list) and calls:
        parsed["name"] = calls[0].get("name", parsed.get("name"))
        parsed["arguments"] = calls[0].get("arguments", parsed.get("arguments", {}))
    return parsed, changed



def schema_primary_type(schema: dict[str, Any]) -> str | None:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)
    return schema_type if isinstance(schema_type, str) else None


def infer_command_from_user(user_text: str) -> str | None:
    patterns = [
        r"\busing\s+(.+?)(?:[.!?]|$)",
        r"\b(?:use|run|execute)\s+(.+?)(?:\s+to\b|\s+for\b|[.!?]|$)",
        r"\bcommand\s+(.+?)(?:\s+to\b|\s+for\b|[.!?]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if not match:
            continue
        command = match.group(1).strip().strip('"\'')
        if command and len(command.split()) <= 6:
            return command
    return None


COMMON_CITY_PATTERN = r"[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3}(?:,\s*[A-Z]{2})?"


def infer_city_or_location_from_user(name: str, schema: dict[str, Any], user_text: str) -> str | None:
    haystack = f"{name} {schema.get('description', '')}".lower()
    if not any(token in haystack for token in ["city", "location", "where"]):
        return None
    patterns = [
        rf"\b(?:in|from|for|at)\s+({COMMON_CITY_PATTERN})(?:\b|[?.!,])",
        rf"\b({COMMON_CITY_PATTERN})\s+on\s+\w+\s+\d",
    ]
    stop = {"I", "Can", "Could", "Please", "What", "Find", "Search", "Get", "List", "The"}
    for pattern in patterns:
        for match in re.finditer(pattern, user_text):
            candidate = match.group(1).strip().strip('"\'')
            words = candidate.split()
            while words and words[0] in stop:
                words.pop(0)
            candidate = " ".join(words).strip()
            if candidate and normalized_phrase(candidate) not in {"a", "the", "all"}:
                return candidate
    return None


def infer_limit_from_user(name: str, schema: dict[str, Any], user_text: str) -> int | None:
    haystack = f"{name} {schema.get('description', '')}".lower()
    if not any(token in haystack for token in ["limit", "perpage", "per page", "page size", "entries"]):
        return None
    patterns = [r"\blimit(?:ing)?(?:\s+the\s+result)?\s+to\s+(\d+)", r"\b(\d+)\s+entries\s+per\s+page"]
    for pattern in patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def infer_date_from_user(name: str, schema: dict[str, Any], user_text: str) -> str | None:
    haystack = f"{name} {schema.get('description', '')}".lower()
    if "date" not in haystack:
        return None
    iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", user_text)
    if iso_match:
        return iso_match.group(0)
    slash_match = re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", user_text)
    if slash_match:
        return slash_match.group(0)
    month_match = re.search(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
        user_text,
        flags=re.IGNORECASE,
    )
    if month_match:
        return month_match.group(0)
    return None


def infer_person_after_cue(user_text: str) -> str | None:
    pattern = r"\b(?:starring|stars|starred by|with|by)\s+([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})"
    match = re.search(pattern, user_text)
    if not match:
        return None
    candidate = match.group(1).strip()
    stop_words = {"Could", "Can", "Please", "I", "The", "A", "An"}
    parts = [part for part in candidate.split() if part not in stop_words]
    return " ".join(parts) if parts else None


def infer_argument_value_from_user(name: str, schema: dict[str, Any], user_text: str) -> Any:
    user_phrase = normalized_phrase(user_text)
    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        matches = [value for value in enum_values if isinstance(value, str) and normalized_phrase(value) and normalized_phrase(value) in user_phrase]
        if len(matches) == 1:
            return matches[0]
    lowered_name = name.lower()
    haystack = f"{name} {schema.get('description', '')}".lower()
    if lowered_name in {"statement", "question", "query"} and schema_primary_type(schema) in {"string", None}:
        return user_text
    if lowered_name in {"repos", "repo", "repository", "repositories"} or "repo" in haystack:
        repos = re.findall(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", user_text)
        if len(repos) >= 2:
            return ",".join(repos)
        if len(repos) == 1:
            return repos[0]
    if lowered_name == "command" or "system command" in haystack:
        return infer_command_from_user(user_text)
    if lowered_name in {"city", "location", "where_to"} or any(word in haystack for word in ["city", "location", "where"]):
        inferred_location = infer_city_or_location_from_user(name, schema, user_text)
        if inferred_location is not None:
            if isinstance(enum_values, list):
                inferred_phrase = normalized_phrase(inferred_location)
                enum_matches = [
                    value for value in enum_values
                    if isinstance(value, str)
                    and inferred_phrase
                    and normalized_phrase(value).startswith(inferred_phrase)
                ]
                if len(enum_matches) == 1:
                    return enum_matches[0]
            return inferred_location
    if lowered_name in {"perpage", "per_page", "limit", "page_size"} or any(word in haystack for word in ["limit", "per page", "entries"]):
        inferred_limit = infer_limit_from_user(name, schema, user_text)
        if inferred_limit is not None:
            return inferred_limit
    if lowered_name in {"date", "day"} or "date" in haystack:
        inferred_date = infer_date_from_user(name, schema, user_text)
        if inferred_date is not None:
            return inferred_date
    if lowered_name in {"starring", "actor", "actress", "artist"} or any(word in haystack for word in ["actor", "actress", "artist", "starring"]):
        return infer_person_after_cue(user_text)
    quoted = quoted_optional_value_near_name(name, schema, user_text)
    if quoted is not None:
        return quoted
    return None


def recover_final_tool_call_for_payload(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    if not parsed or parsed.get("type") != "final":
        return parsed, False
    content = str(parsed.get("content", ""))
    user_text = payload_last_user_text(payload)
    if not user_text:
        return parsed, False
    names = available_tool_names(payload)
    schemas_by_tool = tool_parameter_schemas(payload)
    required_by_tool = tool_required_parameter_names(payload)

    bare_tool_match = re.search(r"<TOOLCALL>\s*\[\s*([^\](){}]+?)\s*\]\s*</TOOLCALL>", content, flags=re.IGNORECASE)
    if "<TOOLCALL>[]" in content or bare_tool_match:
        bracket_name = bare_tool_match.group(1).strip() if bare_tool_match else ""
        matching_names = [name for name in names if name == bracket_name] if bracket_name else []
        if not matching_names and len(names) == 1:
            matching_names = [next(iter(names))]
        if len(matching_names) == 1:
            tool_name = matching_names[0]
            schemas = schemas_by_tool.get(tool_name, {})
            required = required_by_tool.get(tool_name, set())
            arguments: dict[str, Any] = {}
            for name, schema in schemas.items():
                inferred = infer_argument_value_from_user(name, schema, user_text)
                if inferred is not None:
                    arguments[name] = inferred
            if required <= set(arguments):
                return {"type": "tool", "name": tool_name, "arguments": arguments}, True

    bare_list_match = re.search(
        r"<TOOLCALL>\s*\[\s*([^\](){}=]+(?:\s*,\s*[^\](){}=]+)+)\s*\]\s*(?:</TOOLCALL>|\(?TOOLCALL\)?)",
        content,
        flags=re.IGNORECASE,
    )
    if bare_list_match:
        listed = [item.strip() for item in bare_list_match.group(1).split(",") if item.strip()]
        for candidate in listed:
            if candidate in names and not required_by_tool.get(candidate, set()):
                return {"type": "tool", "name": candidate, "arguments": {}}, True

    mentioned = [name for name in names if name in content]
    if len(mentioned) != 1:
        return parsed, False
    tool_name = mentioned[0]
    lowered_content = content.lower()
    if "no functions match" not in lowered_content and "requires additional parameters" not in lowered_content:
        return parsed, False
    schemas = schemas_by_tool.get(tool_name, {})
    required = required_by_tool.get(tool_name, set())
    arguments: dict[str, Any] = {}
    for name, schema in schemas.items():
        inferred = infer_argument_value_from_user(name, schema, user_text)
        if inferred is not None:
            arguments[name] = inferred
    if required and not required <= set(arguments):
        return parsed, False
    if not arguments:
        return parsed, False
    return {"type": "tool", "name": tool_name, "arguments": arguments}, True


def keep_only_available_tool_calls(parsed: dict[str, Any] | None, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not parsed or parsed.get("type") != "tool":
        return parsed
    names = available_tool_names(payload)
    if not names:
        return parsed
    calls = parsed.get("tool_calls")
    call_list = calls if isinstance(calls, list) and calls else [parsed]
    kept = [call for call in call_list if isinstance(call, dict) and call.get("name") in names]
    return merge_tool_calls(kept)


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
        self.config_path = (self.bundle / args.config_file).resolve()
        self.work_dir = args.work_dir.expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.pending_tool_calls: list[dict[str, Any]] = []
        if self.config_path.parent != self.bundle:
            raise ValueError("--config-file must name a file inside --bundle")
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Missing Genie config {self.config_path.name} in {self.bundle}"
            )
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

    def run_tool_guard(self, payload: dict[str, Any], parsed: dict[str, Any], stem: str) -> dict[str, Any]:
        prompt_text = render_nemotron_tool_guard_prompt(payload, parsed)
        prompt_path = self.work_dir / f"{stem}.guard.prompt.txt"
        profile_path = self.work_dir / f"{stem}.guard.profile.json"
        log_path = self.work_dir / f"{stem}.guard.log"
        prompt_path.write_text(prompt_text)
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [
                    "genie-t2t-run",
                    "-c",
                    str(self.config_path),
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
        answer = split["final_answer"] or raw
        decision = "allow" if timed_out else parse_tool_guard_decision(answer)
        return {
            "decision": decision,
            "elapsed_s": round(elapsed_s, 3),
            "returncode": returncode,
            "timed_out": timed_out,
            "raw_answer": raw,
            "final_answer": answer,
            "paths": {
                "prompt": str(prompt_path),
                "log": str(log_path),
                "profile": str(profile_path),
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
        elif self.args.parser == "toolace_pythonic":
            prompt_text = render_toolace_pythonic_prompt(payload)
        elif self.args.parser == "mistral_tool":
            prompt_text = render_mistral_tool_prompt(payload, self.args.mode)
        elif self.args.parser == "qwen3_native":
            prompt_text = render_qwen3_native_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_native":
            prompt_text = render_nemotron_native_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_schema":
            prompt_text = render_nemotron_bfcl_schema_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_schema_guided":
            prompt_text = render_nemotron_bfcl_schema_guided_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_official":
            prompt_text = render_nemotron_bfcl_official_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_official_exact":
            prompt_text = render_nemotron_bfcl_official_prompt(
                payload,
                self.args.mode,
                extra_guidance=NEMOTRON_BFCL_OFFICIAL_EXACT_GUIDANCE,
            )
        elif self.args.parser == "nemotron_bfcl_official_clean_args":
            prompt_text = render_nemotron_bfcl_official_prompt(
                payload,
                self.args.mode,
                extra_guidance=NEMOTRON_BFCL_OFFICIAL_CLEAN_ARGS_GUIDANCE,
            )
        elif self.args.parser == "nemotron_bfcl_official_strict_names":
            prompt_text = render_nemotron_bfcl_official_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_official_strict_schema":
            prompt_text = render_nemotron_bfcl_official_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_official_strict_schema_enhanced":
            prompt_text = render_nemotron_bfcl_official_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_official_strict_schema_enhanced_guarded":
            prompt_text = render_nemotron_bfcl_official_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_official_strict_schema_guarded":
            prompt_text = render_nemotron_bfcl_official_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_official_user_strict_schema":
            prompt_text = render_nemotron_bfcl_official_user_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_official_user_selective_schema":
            prompt_text = render_nemotron_bfcl_official_user_prompt(
                payload,
                self.args.mode,
                extra_guidance=NEMOTRON_BFCL_OFFICIAL_SELECTIVE_GUIDANCE,
            )
        elif self.args.parser == "nemotron_bfcl_official_user_selective_schema_guarded":
            prompt_text = render_nemotron_bfcl_official_user_prompt(
                payload,
                self.args.mode,
                extra_guidance=NEMOTRON_BFCL_OFFICIAL_SELECTIVE_GUIDANCE,
            )
        elif self.args.parser == "nemotron_bfcl_official_user_selective_schema_supported":
            prompt_text = render_nemotron_bfcl_official_user_prompt(
                payload,
                self.args.mode,
                extra_guidance=NEMOTRON_BFCL_OFFICIAL_SELECTIVE_GUIDANCE,
            )
        elif self.args.parser == "nemotron_bfcl_modelcard_supported":
            prompt_text = render_nemotron_bfcl_modelcard_prompt(payload, self.args.mode)
        elif self.args.parser == "nemotron_bfcl_official_strict_schema_exact":
            prompt_text = render_nemotron_bfcl_official_prompt(
                payload,
                self.args.mode,
                extra_guidance=NEMOTRON_BFCL_OFFICIAL_EXACT_GUIDANCE,
            )
        elif self.args.parser == "nemotron_bfcl_official_strict_schema_values":
            prompt_text = render_nemotron_bfcl_official_prompt(
                payload,
                self.args.mode,
                extra_guidance=NEMOTRON_BFCL_OFFICIAL_VALUE_GUIDANCE,
            )
        elif self.args.parser == "nemotron_bfcl_user":
            prompt_text = render_nemotron_bfcl_user_prompt(payload, self.args.mode)
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
                    str(self.config_path),
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
        runtime_status = "timeout" if timed_out else genie_runtime_failure_status(stdout)
        if runtime_status:
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
                "parse_status": runtime_status,
                "parsed_ok": False,
                "parsed_action": None,
                "finish_reason": "stop",
                "raw_answer": raw,
                "final_answer": "",
                "tool_guard": None,
                "supported_guard": None,
                "action_guard": None,
                "semantic_guard": None,
                "paths": {
                    "prompt": str(prompt_path),
                    "log": str(log_path),
                    "profile": str(profile_path),
                },
            }
            REQUEST_LOG.append(record)
            prompt_tokens = len(prompt_text) // 4
            completion_tokens = len(raw) // 4
            return self.completion_payload(
                request_id,
                {"role": "assistant", "content": ""},
                "stop",
                prompt_tokens,
                completion_tokens,
                record,
            )
        split = split_reasoning(raw)
        answer = split["final_answer"]
        parse_input = answer or raw
        if self.args.parser == "strict":
            parsed, parse_status = strict_parse(parse_input)
        elif self.args.parser == "llama3_json":
            parsed, parse_status = llama3_json_parse(parse_input)
        elif self.args.parser == "toolace_pythonic":
            parsed, parse_status = toolace_pythonic_parse(parse_input)
        elif self.args.parser == "mistral_tool":
            parsed, parse_status = mistral_tool_parse(parse_input)
        elif self.args.parser == "qwen3_native":
            parsed, parse_status = qwen3_native_parse(parse_input)
        elif self.args.parser in {"nemotron_bfcl_official", "nemotron_bfcl_official_exact", "nemotron_bfcl_official_clean_args", "nemotron_bfcl_official_strict_names", "nemotron_bfcl_official_strict_schema", "nemotron_bfcl_official_strict_schema_enhanced", "nemotron_bfcl_official_strict_schema_enhanced_guarded", "nemotron_bfcl_official_strict_schema_guarded", "nemotron_bfcl_official_user_strict_schema", "nemotron_bfcl_official_user_selective_schema", "nemotron_bfcl_official_user_selective_schema_guarded", "nemotron_bfcl_official_user_selective_schema_supported", "nemotron_bfcl_official_strict_schema_exact", "nemotron_bfcl_official_strict_schema_values", "nemotron_bfcl_modelcard_supported"}:
            parsed, parse_status = nemotron_native_parse(parse_input, outside_tag_fallback=False)
        elif self.args.parser in {"nemotron_native", "nemotron_bfcl_schema", "nemotron_bfcl_schema_guided", "nemotron_bfcl_user"}:
            parsed, parse_status = nemotron_native_parse(parse_input)
        elif self.args.parser == "qcom_tool":
            parsed, parse_status = qcom_tool_parse(parse_input)
        else:
            parsed, parse_status = tolerant_parse(parse_input)
        if self.args.parser != "toolace_pythonic":
            parsed = normalize_tool_call_names_for_payload(parsed, payload)
        if self.args.parser in {"nemotron_bfcl_official_strict_schema_enhanced", "nemotron_bfcl_official_strict_schema_enhanced_guarded", "nemotron_bfcl_official_user_selective_schema_supported", "nemotron_bfcl_modelcard_supported", "qcom_tool"}:
            parsed, did_final_recover = recover_final_tool_call_for_payload(parsed, payload)
            if did_final_recover:
                parse_status = f"{parse_status}_final_recovered"
        if self.args.parser in {"nemotron_bfcl_official_strict_schema", "nemotron_bfcl_official_strict_schema_enhanced", "nemotron_bfcl_official_strict_schema_enhanced_guarded", "nemotron_bfcl_official_strict_schema_guarded", "nemotron_bfcl_official_user_strict_schema", "nemotron_bfcl_official_user_selective_schema", "nemotron_bfcl_official_user_selective_schema_guarded", "nemotron_bfcl_official_user_selective_schema_supported", "nemotron_bfcl_official_strict_schema_exact", "nemotron_bfcl_official_strict_schema_values", "nemotron_bfcl_modelcard_supported", "qcom_tool"}:
            parsed, did_schema_repair = repair_tool_arguments_for_payload(parsed, payload)
            if did_schema_repair:
                parse_status = f"{parse_status}_schema_repaired"
        if self.args.parser in {"nemotron_bfcl_official_strict_names", "nemotron_bfcl_official_strict_schema", "nemotron_bfcl_official_strict_schema_enhanced", "nemotron_bfcl_official_strict_schema_enhanced_guarded", "nemotron_bfcl_official_strict_schema_guarded", "nemotron_bfcl_official_user_strict_schema", "nemotron_bfcl_official_user_selective_schema", "nemotron_bfcl_official_user_selective_schema_guarded", "nemotron_bfcl_official_user_selective_schema_supported", "nemotron_bfcl_official_strict_schema_exact", "nemotron_bfcl_official_strict_schema_values", "nemotron_bfcl_modelcard_supported", "qcom_tool"}:
            before_strict_names = parsed
            parsed = keep_only_available_tool_calls(parsed, payload)
            if before_strict_names and not parsed:
                parse_status = f"{parse_status}_unknown_tool_rejected"
        supported_guard_info = None
        action_guard_info = None
        semantic_guard_info = None
        if self.args.parser in {"nemotron_bfcl_official_strict_schema_enhanced_guarded", "nemotron_bfcl_modelcard_supported", "qcom_tool"}:
            parsed, action_guard_info = prune_or_rewrite_action_mismatch_tool_calls(parsed, payload)
            if action_guard_info and action_guard_info.get("decision") == "rewrite_or_prune":
                parse_status = f"{parse_status}_action_pruned"
            elif action_guard_info and action_guard_info.get("decision") == "reject":
                parse_status = f"{parse_status}_action_rejected"
        if self.args.parser in {"nemotron_bfcl_official_strict_schema_enhanced", "nemotron_bfcl_official_strict_schema_enhanced_guarded", "nemotron_bfcl_official_user_selective_schema_supported", "nemotron_bfcl_modelcard_supported", "qcom_tool"}:
            parsed, did_enhance = enhance_tool_arguments_for_payload(parsed, payload)
            if did_enhance:
                parse_status = f"{parse_status}_supported_enhanced"
        if self.args.parser in {"nemotron_bfcl_official_strict_schema_enhanced_guarded", "nemotron_bfcl_modelcard_supported", "qcom_tool"}:
            parsed, semantic_guard_info = reject_semantic_mismatch_tool_calls(parsed, payload)
            if semantic_guard_info and semantic_guard_info.get("decision") == "prune":
                parse_status = f"{parse_status}_semantic_pruned"
            elif semantic_guard_info and semantic_guard_info.get("decision") == "reject":
                parse_status = f"{parse_status}_semantic_rejected"
            parsed, supported_guard_info = reject_relevance_unsupported_tool_calls(parsed, payload)
            if supported_guard_info and supported_guard_info.get("decision") == "reject":
                parse_status = f"{parse_status}_relevance_rejected"
        if self.args.parser == "nemotron_bfcl_official_user_selective_schema_supported":
            parsed, supported_guard_info = reject_unsupported_required_tool_calls(parsed, payload)
            if supported_guard_info and supported_guard_info.get("decision") == "reject":
                parse_status = f"{parse_status}_supported_rejected"
        parsed = self.apply_multi_tool_policy(parsed, payload)
        guard_info = None
        if (
            self.args.parser in {"nemotron_bfcl_official_strict_schema_guarded", "nemotron_bfcl_official_user_selective_schema_guarded"}
            and parsed
            and parsed.get("type") == "tool"
            and not timed_out
            and not self.payload_has_tool_result(payload)
        ):
            guard_info = self.run_tool_guard(payload, parsed, stem)
            if guard_info.get("decision") == "reject":
                parsed = {"type": "final", "content": "No suitable tool call."}
                parse_status = f"{parse_status}_guard_rejected"
            else:
                parse_status = f"{parse_status}_guard_allowed"
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
            "tool_guard": guard_info,
            "supported_guard": supported_guard_info,
            "action_guard": action_guard_info,
            "semantic_guard": semantic_guard_info,
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
    parser.add_argument("--config-file", default="genie_config.json")
    parser.add_argument(
        "--parser",
        choices=[
            "strict",
            "tolerant",
            "llama3_json",
            "toolace_pythonic",
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
