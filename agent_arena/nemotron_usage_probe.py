#!/usr/bin/env python3
"""Focused Nemotron usage probe for BFCL-style tool calls.

This is intentionally smaller and lower-level than the full benchmark runner:
load the non-quantized HF model once, render a handful of representative cases
with different Nemotron prompt shapes, capture raw text, parse it with the same
native parser used by the Genie shim, and optionally run BFCL scoring on the
parsed one-turn outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_arena.openai_genie_server import nemotron_native_parse


DEFAULT_CASE_IDS = [
    "simple_python_0",
    "simple_python_145",
    "simple_python_290",
    "multiple_0",
    "parallel_57",
    "live_relevance_0-0-0",
    "live_simple_57-26-1",
    "live_parallel_5-2-0",
    "irrelevance_34",
]

DEFAULT_VARIANTS = [
    "hf_native_off",
    "bfcl_user_off",
    "bfcl_user_strict_off",
    "bfcl_user_light_schema_off",
    "bfcl_user_on",
    "hf_native_on",
]


@dataclass(frozen=True)
class Variant:
    name: str
    template: str
    thinking: str
    strict: bool = False
    light_schema: bool = False


VARIANTS = {
    "hf_native_off": Variant("hf_native_off", "hf_native", "off"),
    "hf_native_on": Variant("hf_native_on", "hf_native", "on"),
    "bfcl_user_off": Variant("bfcl_user_off", "bfcl_user", "off"),
    "bfcl_user_on": Variant("bfcl_user_on", "bfcl_user", "on"),
    "bfcl_user_strict_off": Variant("bfcl_user_strict_off", "bfcl_user", "off", strict=True),
    "bfcl_user_light_schema_off": Variant(
        "bfcl_user_light_schema_off", "bfcl_user", "off", light_schema=True
    ),
}


def load_bfcl_runner(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "bfcl_v4_subset_runner", project_root / "agent_arena" / "bfcl_v4_subset_runner.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load bfcl_v4_subset_runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def first_turn_messages(entry: dict[str, Any]) -> list[dict[str, Any]]:
    question = entry.get("question", [])
    if question and isinstance(question[0], list):
        return [m for m in question[0] if isinstance(m, dict)]
    return [m for m in question if isinstance(m, dict)]


def with_underscored_tool_names(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for function in functions:
        copied = json.loads(json.dumps(function))
        if isinstance(copied.get("name"), str):
            copied["name"] = copied["name"].replace(".", "_")
        tools.append({"type": "function", "function": copied})
    return tools


def light_function_schema(function: dict[str, Any]) -> dict[str, Any]:
    params = function.get("parameters", {})
    properties = params.get("properties", {}) if isinstance(params, dict) else {}
    required = params.get("required", []) if isinstance(params, dict) else []
    keep_props: dict[str, Any] = {}
    for name, schema in properties.items():
        if not isinstance(schema, dict):
            keep_props[name] = schema
            continue
        slim = {
            key: schema[key]
            for key in ("type", "description", "enum", "default", "items")
            if key in schema
        }
        keep_props[name] = slim
    return {
        "name": function.get("name"),
        "description": function.get("description", ""),
        "parameters": {
            "type": params.get("type", "dict") if isinstance(params, dict) else "dict",
            "properties": keep_props,
            "required": required,
        },
    }


def maybe_light_tools(tools: list[dict[str, Any]], light_schema: bool) -> list[dict[str, Any]]:
    if not light_schema:
        return tools
    light_tools = []
    for tool in tools:
        function = tool.get("function", tool)
        light_tools.append({"type": "function", "function": light_function_schema(function)})
    return light_tools


def split_system(messages: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    system_parts = []
    rest = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
        else:
            rest.append(message)
    return system_parts, rest


def prefix_first_user(messages: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    if not prefix.strip():
        return messages
    copied = [dict(message) for message in messages]
    for message in copied:
        if message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            message["content"] = f"{prefix.strip()}\n\n{content}".strip()
            return copied
    return [{"role": "user", "content": prefix.strip()}, *copied]


def tool_payloads(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool.get("function", tool) for tool in tools]


def render_bfcl_user_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    variant: Variant,
) -> str:
    system_parts, rest = split_system(messages)
    mode = f"detailed thinking {variant.thinking}"
    rendered_tools = json.dumps(
        tool_payloads(tools), ensure_ascii=False, separators=(",", ":")
    )
    prefix_parts = [*system_parts, f"<AVAILABLE_TOOLS>{rendered_tools}</AVAILABLE_TOOLS>"]
    if variant.strict:
        prefix_parts.append(
            "Return only one of these forms and no prose:\n"
            "<TOOLCALL>[{\"name\":\"function_name\",\"arguments\":{\"arg\":\"value\"}}]</TOOLCALL>\n"
            "or a short direct final answer if no tool should be called. "
            "Use every independent tool call required by the user request."
        )
    prompt_messages = [{"role": "system", "content": mode}]
    prompt_messages.extend(prefix_first_user(rest, "\n\n".join(prefix_parts)))
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def render_hf_native_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    variant: Variant,
) -> str:
    system_parts, rest = split_system(messages)
    mode = f"detailed thinking {variant.thinking}"
    prompt_messages = [{"role": "system", "content": mode}]
    prompt_messages.extend(prefix_first_user(rest, "\n\n".join(system_parts)))
    return tokenizer.apply_chat_template(
        prompt_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def render_prompt(
    tokenizer: Any,
    entry: dict[str, Any],
    variant: Variant,
) -> tuple[str, int]:
    messages = first_turn_messages(entry)
    tools = with_underscored_tool_names(entry.get("function", []))
    tools = maybe_light_tools(tools, variant.light_schema)
    if variant.template == "hf_native":
        prompt = render_hf_native_prompt(tokenizer, messages, tools, variant)
    elif variant.template == "bfcl_user":
        prompt = render_bfcl_user_prompt(tokenizer, messages, tools, variant)
    else:
        raise ValueError(f"Unknown template: {variant.template}")
    return prompt, len(tools)


def generation_kwargs(tokenizer: Any, variant: Variant, max_new_tokens: int) -> dict[str, Any]:
    eos_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if variant.thinking == "on":
        kwargs.update({"do_sample": True, "temperature": 0.6, "top_p": 0.95})
    else:
        kwargs.update({"do_sample": False})
    return kwargs


def parsed_to_bfcl_result(parsed: dict[str, Any] | None, raw: str) -> Any:
    if not parsed:
        return raw
    if parsed.get("type") == "final":
        return parsed.get("content", raw)
    calls = parsed.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        calls = [parsed]
    result = []
    for call in calls:
        name = str(call.get("name", ""))
        arguments = call.get("arguments", {})
        result.append({name: json.dumps(arguments, ensure_ascii=False)})
    return result


def load_entries(bfcl: Any, case_ids: list[str]) -> dict[str, dict[str, Any]]:
    entries = {}
    categories = sorted({bfcl.utils.extract_test_category_from_id(case_id) for case_id in case_ids})
    for category in categories:
        for entry in bfcl.utils.load_dataset_entry(category):
            if entry["id"] in case_ids:
                entries[entry["id"]] = entry
    missing = [case_id for case_id in case_ids if case_id not in entries]
    if missing:
        raise ValueError(f"Missing BFCL case ids: {missing}")
    return entries


def load_evk_result(run_root: Path, model_id: str, case_id: str, category: str) -> Any:
    path = (
        run_root
        / "result"
        / model_id
        / ("live" if category.startswith("live_") else "agentic" if category.startswith("web_search") else "non_live")
        / f"BFCL_v4_{category}_result.json"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") == case_id:
                return row.get("result")
    return None


def write_result_files(
    output_root: Path,
    model_id: str,
    rows: list[dict[str, Any]],
    bfcl: Any,
) -> tuple[Path, Path]:
    result_root = output_root / "result" / model_id
    score_root = output_root / "score"
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        category = bfcl.utils.extract_test_category_from_id(row["id"])
        if category.startswith("web_search"):
            continue
        by_category.setdefault(category, []).append(
            {
                "id": row["id"],
                "result": row["bfcl_result"],
                "input_token_count": row["input_token_count"],
                "output_token_count": row["output_token_count"],
                "latency": row["latency"],
            }
        )

    for category, category_rows in by_category.items():
        section = "live" if category.startswith("live_") else "non_live"
        path = result_root / section / f"BFCL_v4_{category}_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in sorted(category_rows, key=lambda row: bfcl.utils.sort_key({"id": row["id"]})):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_root / "result", score_root


def run_eval(
    bfcl_runner: Any,
    bfcl: Any,
    output_root: Path,
    model_ids: list[str],
    categories: list[str],
) -> None:
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    for model_id in model_ids:
        bfcl_runner.register_openai_endpoint(bfcl, model_id, model_id, model_id)
    bfcl.eval_runner.main(
        model_ids,
        categories,
        output_root / "result",
        output_root / "score",
        partial_eval=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    parser.add_argument("--output-root", type=Path, default=Path("agent_arena_results/nemotron_usage_probe"))
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--evk-run-root", type=Path, default=Path("agent_arena_results/bfcl_v4_100/bfcl_v4_signal100_ctxsafe_20260624"))
    parser.add_argument("--evk-model-id", default="evk-nemotron-bfcl-user-off-ctxsafe-bfcl")
    args = parser.parse_args()

    project_root = Path.cwd()
    output_root = args.output_root.resolve()
    case_ids = args.case_ids or DEFAULT_CASE_IDS
    variant_names = args.variants or DEFAULT_VARIANTS
    variants = [VARIANTS[name] for name in variant_names]

    bfcl_runner = load_bfcl_runner(project_root)
    bfcl = bfcl_runner.import_bfcl(Path("agent_arena_results/bfcl_v4_100"))
    entries = load_entries(bfcl, case_ids)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=True,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.to("cuda")
    model.eval()

    raw_rows: list[dict[str, Any]] = []
    scored_models: list[str] = []
    categories = sorted({bfcl.utils.extract_test_category_from_id(case_id) for case_id in case_ids})

    for variant in variants:
        model_label = f"host-nemotron-{variant.name.replace('_', '-')}"
        scored_models.append(model_label)
        variant_rows: list[dict[str, Any]] = []
        print(f"\n## {variant.name}")
        for case_id in case_ids:
            entry = entries[case_id]
            category = bfcl.utils.extract_test_category_from_id(case_id)
            prompt, tool_count = render_prompt(tokenizer, entry, variant)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            start = time.time()
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    **generation_kwargs(tokenizer, variant, args.max_new_tokens),
                )
            latency = time.time() - start
            generated = output[0][inputs.input_ids.shape[-1] :]
            raw = tokenizer.decode(generated, skip_special_tokens=False)
            parsed, parse_status = nemotron_native_parse(raw)
            bfcl_result = parsed_to_bfcl_result(parsed, raw)
            row = {
                "variant": variant.name,
                "model_id": model_label,
                "id": case_id,
                "category": category,
                "tool_count": tool_count,
                "input_token_count": int(inputs.input_ids.shape[-1]),
                "output_token_count": int(generated.shape[-1]),
                "latency": latency,
                "parse_status": parse_status,
                "raw": raw,
                "parsed": parsed,
                "bfcl_result": bfcl_result,
                "evk_bfcl_user_off_result": load_evk_result(
                    args.evk_run_root, args.evk_model_id, case_id, category
                ),
            }
            raw_rows.append(row)
            variant_rows.append(row)
            print(
                f"{case_id:28s} tokens={row['output_token_count']:4d} "
                f"parse={parse_status:32s} result={str(bfcl_result)[:160]}"
            )
        write_result_files(output_root, model_label, variant_rows, bfcl)

    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "raw_generations.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nRaw generations: {raw_path}")

    if not args.skip_eval:
        scorable_categories = [category for category in categories if not category.startswith("web_search")]
        run_eval(bfcl_runner, bfcl, output_root, scored_models, scorable_categories)


if __name__ == "__main__":
    main()
