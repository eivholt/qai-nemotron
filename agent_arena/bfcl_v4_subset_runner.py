#!/usr/bin/env python3
"""Run a fixed BFCL V4 subset against an OpenAI-compatible EVK endpoint.

The official BFCL package already contains the prompt data, tool simulators,
multi-turn loop, and scorers. This wrapper keeps the run honest by registering
our EVK endpoint as a custom OpenAI-compatible function-calling model at runtime
and by selecting a deterministic stratified 100-case scored subset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


RUNTIME_FAILURE_TOOL_NAME = "__agent_arena_runtime_failure__"


ALLOCATION_PROFILES = {
    "signal100": {
        "simple_python": 12,
        "multiple": 8,
        "parallel": 8,
        "parallel_multiple": 7,
        "irrelevance": 8,
        "live_simple": 10,
        "live_multiple": 8,
        "live_parallel": 4,
        "live_parallel_multiple": 4,
        "live_relevance": 4,
        "live_irrelevance": 7,
        "multi_turn_base": 3,
        "multi_turn_miss_func": 2,
        "multi_turn_miss_param": 3,
        "multi_turn_long_context": 2,
        "web_search_base": 5,
        "web_search_no_snippet": 5,
    },
    "practical100": {
        "simple_python": 4,
        "multiple": 2,
        "parallel": 2,
        "parallel_multiple": 2,
        "irrelevance": 5,
        "live_simple": 3,
        "live_multiple": 3,
        "live_parallel": 1,
        "live_parallel_multiple": 1,
        "live_relevance": 2,
        "live_irrelevance": 5,
        "multi_turn_base": 10,
        "multi_turn_miss_func": 7,
        "multi_turn_miss_param": 7,
        "multi_turn_long_context": 6,
        "web_search_base": 20,
        "web_search_no_snippet": 20,
    },
    "memory100": {
        "simple_python": 4,
        "multiple": 2,
        "parallel": 2,
        "parallel_multiple": 2,
        "irrelevance": 5,
        "live_simple": 3,
        "live_multiple": 3,
        "live_parallel": 1,
        "live_parallel_multiple": 1,
        "live_relevance": 2,
        "live_irrelevance": 5,
        "multi_turn_base": 10,
        "multi_turn_miss_func": 7,
        "multi_turn_miss_param": 7,
        "multi_turn_long_context": 6,
        "web_search_base": 10,
        "web_search_no_snippet": 10,
        "memory_kv": 7,
        "memory_vector": 7,
        "memory_rec_sum": 6,
    },
}

GROUPS = {
    "non_live_ast": {
        "simple_python",
        "multiple",
        "parallel",
        "parallel_multiple",
    },
    "irrelevance": {"irrelevance", "live_irrelevance"},
    "live_ast": {
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_relevance",
    },
    "multi_turn": {
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    },
    "agentic": {
        "web_search_base",
        "web_search_no_snippet",
        "memory_kv",
        "memory_vector",
        "memory_rec_sum",
    },
}

KNOWN_EVK_MODEL_IDS = {
    "evk-stock-llama-bfcl": "EVK_stock_Llama_3.1_8B_W4A16_BFCL",
    "evk-nemotron-native-off-bfcl": "EVK_Nemotron_Nano_8B_W4A16_native_off_BFCL",
    "evk-nemotron-native-on-bfcl": "EVK_Nemotron_Nano_8B_W4A16_native_on_BFCL",
    "evk-nemotron-bfcl-user-off-bfcl": "EVK_Nemotron_Nano_8B_W4A16_BFCL_user_off",
    "evk-nemotron-bfcl-user-on-bfcl": "EVK_Nemotron_Nano_8B_W4A16_BFCL_user_on",
    "evk-nemotron-native-off-ctxsafe-bfcl": "EVK_Nemotron_Nano_8B_W4A16_native_off_ctxsafe",
    "evk-nemotron-native-on-ctxsafe-bfcl": "EVK_Nemotron_Nano_8B_W4A16_native_on_ctxsafe",
    "evk-nemotron-bfcl-user-off-ctxsafe-bfcl": "EVK_Nemotron_Nano_8B_W4A16_BFCL_user_off_ctxsafe",
    "evk-nemotron-bfcl-user-on-ctxsafe-bfcl": "EVK_Nemotron_Nano_8B_W4A16_BFCL_user_on_ctxsafe",
    "evk-ministral-q4-bfcl": "EVK_Ministral_3_3B_Q4_BFCL",
}

DEFAULT_CONTEXT_EXCESS_EXCLUSIONS = Path("agent_arena/bfcl_v4_context_excess_exclusions.json")


@dataclass(frozen=True)
class BfclModules:
    model_config: object
    generation: object
    eval_runner: object
    utils: object
    openai_handler: object
    config_cls: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "generate", "evaluate", "run", "summarize"],
        help="prepare writes the subset manifest; run performs generate+evaluate.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("agent_arena_results/bfcl_v4_100"),
        help="Root used by BFCL for result, score, and helper files.",
    )
    parser.add_argument("--run-name", default="bfcl_v4_100")
    parser.add_argument("--model-id", default="evk-model-bfcl")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--endpoint", default="http://192.168.1.92:8001/v1")
    parser.add_argument("--endpoint-model", default=None)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--include-input-log", action="store_true")
    parser.add_argument("--exclude-state-log", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--native-bfcl-functions",
        action="store_true",
        help="Pass original BFCL function docs through to compatible local shims as bfcl_functions.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(ALLOCATION_PROFILES),
        default="signal100",
        help=(
            "signal100 emphasizes categories where at least one EVK model showed signal; "
            "practical100 is the original web-search/multi-turn-heavy profile; "
            "memory100 includes BFCL memory setup."
        ),
    )
    parser.add_argument(
        "--limit-scored",
        type=int,
        default=None,
        help="Optional scored-case cap for smoke tests; keeps category order deterministic.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Run explicit BFCL case ID(s). May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--case-file",
        type=Path,
        help="JSON file containing a case_ids list (or a top-level JSON list).",
    )
    parser.add_argument(
        "--context-excess-exclusions",
        type=Path,
        default=DEFAULT_CONTEXT_EXCESS_EXCLUSIONS,
        help="JSON file containing BFCL case IDs to exclude because they exceeded the Genie context window.",
    )
    parser.add_argument(
        "--no-context-excess-exclusions",
        action="store_true",
        help="Disable context-excess exclusions even when the exclusion file exists.",
    )
    return parser.parse_args()


def import_bfcl(project_root: Path) -> BfclModules:
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root.resolve())
    from bfcl_eval import _llm_response_generation as generation
    from bfcl_eval.constants import model_config
    from bfcl_eval.constants.model_config import ModelConfig
    from bfcl_eval.eval_checker import eval_runner
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval import utils

    return BfclModules(
        model_config=model_config,
        generation=generation,
        eval_runner=eval_runner,
        utils=utils,
        openai_handler=OpenAICompletionsHandler,
        config_cls=ModelConfig,
    )


def model_system_message(path: Path | None) -> dict | None:
    if path is None:
        return None
    text = path.expanduser().read_text()
    think_start = text.find("[THINK]")
    think_end = text.find("[/THINK]")
    if think_start < 0 or think_end < think_start:
        return {"role": "system", "content": text}
    return {
        "role": "system",
        "content": [
            {"type": "text", "text": text[:think_start]},
            {
                "type": "thinking",
                "thinking": text[think_start + len("[THINK]") : think_end],
                "closed": True,
            },
            {"type": "text", "text": text[think_end + len("[/THINK]") :]},
        ],
    }


def make_bfcl_request_handler(
    openai_handler_cls: object,
    *,
    top_p: float | None,
    max_tokens: int | None,
    system_message: dict | None,
    native_bfcl_functions: bool,
) -> object:
    class BfclPassthroughOpenAIHandler(openai_handler_cls):  # type: ignore[misc, valid-type]
        def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
            inference_data = super()._compile_tools(inference_data, test_entry)
            inference_data["agent_arena_case_id"] = test_entry.get("id")
            if native_bfcl_functions:
                inference_data["bfcl_functions"] = deepcopy(test_entry.get("function", []))
            return inference_data

        def _query_FC(self, inference_data: dict):
            message: list[dict] = deepcopy(inference_data["message"])
            if system_message is not None:
                message.insert(0, deepcopy(system_message))
            tools = inference_data["tools"]
            bfcl_functions = inference_data.get("bfcl_functions", [])
            inference_data["inference_input_log"] = {
                "message": repr(message),
                "tools": tools,
                "bfcl_functions": bfcl_functions,
            }
            kwargs = {
                "messages": message,
                "model": self.model_name,
                "temperature": self.temperature,
                "store": False,
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            extra_body = {
                "agent_arena_strict_runtime_failures": True,
                "agent_arena_case_id": inference_data.get("agent_arena_case_id"),
            }
            if native_bfcl_functions:
                extra_body["bfcl_functions"] = bfcl_functions
            kwargs["extra_body"] = extra_body
            if len(tools) > 0:
                kwargs["tools"] = tools
            return self.generate_with_backoff(**kwargs)

    return BfclPassthroughOpenAIHandler


def register_openai_endpoint(
    bfcl: BfclModules,
    model_id: str,
    endpoint_model: str,
    display_name: str,
    top_p: float | None = None,
    max_tokens: int | None = None,
    system_message: dict | None = None,
    native_bfcl_functions: bool = False,
) -> None:
    handler = make_bfcl_request_handler(
        bfcl.openai_handler,
        top_p=top_p,
        max_tokens=max_tokens,
        system_message=system_message,
        native_bfcl_functions=native_bfcl_functions,
    )
    config = bfcl.config_cls(
        model_name=endpoint_model,
        display_name=display_name,
        url="local OpenAI-compatible EVK Genie shim",
        org="local",
        license="local",
        model_handler=handler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        # BFCL's OpenAI tool conversion rewrites dotted function names to
        # underscores because Chat Completions tool names cannot contain dots.
        underscore_to_dot=True,
    )
    bfcl.model_config.MODEL_CONFIG_MAPPING[model_id] = config


def register_known_evk_models(bfcl: BfclModules) -> None:
    for model_id, display_name in KNOWN_EVK_MODEL_IDS.items():
        if model_id not in bfcl.model_config.MODEL_CONFIG_MAPPING:
            register_openai_endpoint(
                bfcl,
                model_id,
                model_id,
                display_name,
            )


def spread_select(entries: list[dict], count: int) -> list[dict]:
    if count <= 0:
        return []
    if count >= len(entries):
        return list(entries)
    if count == 1:
        return [entries[0]]

    raw_indices = [round(i * (len(entries) - 1) / (count - 1)) for i in range(count)]
    selected: list[int] = []
    used: set[int] = set()
    for raw in raw_indices:
        idx = raw
        while idx in used and idx + 1 < len(entries):
            idx += 1
        while idx in used and idx - 1 >= 0:
            idx -= 1
        used.add(idx)
        selected.append(idx)
    selected.sort()
    return [entries[idx] for idx in selected]


def category_for_id(bfcl: BfclModules, entry_id: str) -> str:
    return bfcl.utils.extract_test_category_from_id(entry_id)


def is_scored_memory_entry(bfcl: BfclModules, entry: dict) -> bool:
    entry_id = entry["id"]
    category = category_for_id(bfcl, entry_id)
    return bfcl.utils.is_memory(category) and not bfcl.utils.is_memory_prereq(category)


def explicit_case_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.case_file:
        data = json.loads(args.case_file.read_text(encoding="utf-8"))
        file_ids = data.get("case_ids") if isinstance(data, dict) else data
        if not isinstance(file_ids, list) or not all(isinstance(item, str) for item in file_ids):
            raise ValueError(f"{args.case_file} must contain a case_ids string list")
        ids.extend(file_ids)
    for item in args.case_ids or []:
        ids.extend(part.strip() for part in str(item).split(",") if part.strip())
    return ids


def build_subset_entries(
    bfcl: BfclModules,
    allocation: dict[str, int],
    limit_scored: int | None = None,
    excluded_ids: set[str] | None = None,
) -> tuple[list[dict], dict]:
    selected_scored: list[dict] = []
    all_loaded_by_id: dict[str, dict] = {}
    requested_by_category: dict[str, int] = {}
    excluded_ids = excluded_ids or set()

    for category, count in allocation.items():
        requested_by_category[category] = count
        entries = bfcl.utils.load_dataset_entry(category)
        for entry in entries:
            all_loaded_by_id[entry["id"]] = entry

        if bfcl.utils.is_memory(category):
            candidates = [entry for entry in entries if is_scored_memory_entry(bfcl, entry)]
        else:
            candidates = entries

        chosen = spread_select(candidates, count)
        selected_scored.extend(deepcopy(chosen))

    excluded_selected = [entry for entry in selected_scored if entry["id"] in excluded_ids]
    selected_scored = [entry for entry in selected_scored if entry["id"] not in excluded_ids]

    if limit_scored is not None:
        selected_scored = selected_scored[:limit_scored]

    selected_ids = {entry["id"] for entry in selected_scored}
    auxiliary_ids: set[str] = set()
    for entry in selected_scored:
        for dependency_id in entry.get("depends_on", []):
            if dependency_id not in selected_ids:
                auxiliary_ids.add(dependency_id)

    auxiliary_entries = [
        deepcopy(all_loaded_by_id[entry_id])
        for entry_id in sorted(auxiliary_ids, key=lambda item: bfcl.utils.sort_key({"id": item}))
    ]
    all_entries = auxiliary_entries + selected_scored
    all_entries = sorted(all_entries, key=bfcl.utils.sort_key)

    scored_by_category: dict[str, int] = {}
    auxiliary_by_category: dict[str, int] = {}
    excluded_by_category: dict[str, int] = {}
    for entry in selected_scored:
        category = category_for_id(bfcl, entry["id"])
        scored_by_category[category] = scored_by_category.get(category, 0) + 1
    for entry in auxiliary_entries:
        category = category_for_id(bfcl, entry["id"])
        auxiliary_by_category[category] = auxiliary_by_category.get(category, 0) + 1
    for entry in excluded_selected:
        category = category_for_id(bfcl, entry["id"])
        excluded_by_category[category] = excluded_by_category.get(category, 0) + 1

    manifest = {
        "bfcl_version": "BFCL_v4",
        "profile": "custom",
        "selection": "deterministic-spread",
        "requested_scored_total": sum(allocation.values()),
        "scored_total": len(selected_scored),
        "auxiliary_total": len(auxiliary_entries),
        "excluded_context_excess_total": len(excluded_selected),
        "requested_by_category": requested_by_category,
        "scored_by_category": dict(sorted(scored_by_category.items())),
        "auxiliary_by_category": dict(sorted(auxiliary_by_category.items())),
        "excluded_context_excess_by_category": dict(sorted(excluded_by_category.items())),
        "scored_ids": [entry["id"] for entry in selected_scored],
        "auxiliary_ids": [entry["id"] for entry in auxiliary_entries],
        "excluded_context_excess_ids": [entry["id"] for entry in excluded_selected],
    }
    return all_entries, manifest


def build_case_id_entries(
    bfcl: BfclModules,
    case_ids: list[str],
    excluded_ids: set[str] | None = None,
) -> tuple[list[dict], dict]:
    excluded_ids = excluded_ids or set()
    all_loaded_by_id: dict[str, dict] = {}
    selected_scored: list[dict] = []
    requested_by_category: dict[str, int] = {}
    excluded_selected: list[dict] = []

    for case_id in case_ids:
        category = category_for_id(bfcl, case_id)
        requested_by_category[category] = requested_by_category.get(category, 0) + 1
        if not any(category_for_id(bfcl, loaded_id) == category for loaded_id in all_loaded_by_id):
            for entry in bfcl.utils.load_dataset_entry(category):
                all_loaded_by_id[entry["id"]] = entry
        if case_id not in all_loaded_by_id:
            raise ValueError(f"Unknown BFCL case ID: {case_id}")
        entry = deepcopy(all_loaded_by_id[case_id])
        if case_id in excluded_ids:
            excluded_selected.append(entry)
        else:
            selected_scored.append(entry)

    selected_ids = {entry["id"] for entry in selected_scored}
    auxiliary_ids: set[str] = set()
    for entry in selected_scored:
        for dependency_id in entry.get("depends_on", []):
            if dependency_id not in selected_ids:
                auxiliary_ids.add(dependency_id)

    for dependency_id in list(auxiliary_ids):
        if dependency_id not in all_loaded_by_id:
            category = category_for_id(bfcl, dependency_id)
            for entry in bfcl.utils.load_dataset_entry(category):
                all_loaded_by_id[entry["id"]] = entry

    auxiliary_entries = [
        deepcopy(all_loaded_by_id[entry_id])
        for entry_id in sorted(auxiliary_ids, key=lambda item: bfcl.utils.sort_key({"id": item}))
    ]
    all_entries = sorted(auxiliary_entries + selected_scored, key=bfcl.utils.sort_key)

    scored_by_category: dict[str, int] = {}
    auxiliary_by_category: dict[str, int] = {}
    excluded_by_category: dict[str, int] = {}
    for entry in selected_scored:
        category = category_for_id(bfcl, entry["id"])
        scored_by_category[category] = scored_by_category.get(category, 0) + 1
    for entry in auxiliary_entries:
        category = category_for_id(bfcl, entry["id"])
        auxiliary_by_category[category] = auxiliary_by_category.get(category, 0) + 1
    for entry in excluded_selected:
        category = category_for_id(bfcl, entry["id"])
        excluded_by_category[category] = excluded_by_category.get(category, 0) + 1

    manifest = {
        "bfcl_version": "BFCL_v4",
        "profile": "explicit-case-ids",
        "selection": "explicit-case-ids",
        "requested_scored_total": len(case_ids),
        "scored_total": len(selected_scored),
        "auxiliary_total": len(auxiliary_entries),
        "excluded_context_excess_total": len(excluded_selected),
        "requested_by_category": requested_by_category,
        "scored_by_category": dict(sorted(scored_by_category.items())),
        "auxiliary_by_category": dict(sorted(auxiliary_by_category.items())),
        "excluded_context_excess_by_category": dict(sorted(excluded_by_category.items())),
        "scored_ids": [entry["id"] for entry in selected_scored],
        "auxiliary_ids": [entry["id"] for entry in auxiliary_entries],
        "excluded_context_excess_ids": [entry["id"] for entry in excluded_selected],
    }
    return all_entries, manifest


def run_root(args: argparse.Namespace) -> Path:
    return (args.project_root / args.run_name).resolve()


def write_manifest(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_context_excess_exclusions(args: argparse.Namespace) -> tuple[set[str], dict]:
    if args.no_context_excess_exclusions:
        return set(), {"enabled": False, "reason": "disabled"}
    path = args.context_excess_exclusions
    if not path.exists():
        return set(), {"enabled": False, "reason": "file_not_found", "path": str(path)}
    data = json.loads(path.read_text())
    ids = data.get("excluded_ids", [])
    if not isinstance(ids, list):
        raise ValueError(f"{path} must contain an excluded_ids list")
    return {str(item) for item in ids}, {
        "enabled": True,
        "path": str(path),
        "metadata": {k: v for k, v in data.items() if k != "excluded_ids"},
    }


def prepare(args: argparse.Namespace, bfcl: BfclModules) -> tuple[list[dict], dict]:
    root = run_root(args)
    excluded_ids, exclusion_info = load_context_excess_exclusions(args)
    case_ids = explicit_case_ids(args)
    if case_ids:
        entries, manifest = build_case_id_entries(
            bfcl,
            case_ids,
            excluded_ids=excluded_ids,
        )
    else:
        entries, manifest = build_subset_entries(
            bfcl,
            ALLOCATION_PROFILES[args.profile],
            args.limit_scored,
            excluded_ids=excluded_ids,
        )
        manifest["profile"] = args.profile
    manifest["context_excess_exclusions"] = exclusion_info
    manifest["request_settings"] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "system_prompt_file": str(args.system_prompt_file) if args.system_prompt_file else None,
    }
    write_manifest(root, manifest)
    return entries, manifest


def result_dir(args: argparse.Namespace) -> Path:
    return run_root(args) / "result"


def score_dir(args: argparse.Namespace) -> Path:
    return run_root(args) / "score"


def clean_model_outputs(args: argparse.Namespace) -> None:
    model_dir_name = args.model_id.replace("/", "_")
    for base in [result_dir(args), score_dir(args)]:
        target = base / model_dir_name
        if target.exists():
            shutil.rmtree(target)


def generate(args: argparse.Namespace, bfcl: BfclModules) -> None:
    os.environ["OPENAI_BASE_URL"] = args.endpoint
    os.environ["OPENAI_API_KEY"] = args.api_key
    register_openai_endpoint(
        bfcl,
        args.model_id,
        args.endpoint_model or args.model_id,
        args.display_name or args.model_id,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        system_message=model_system_message(args.system_prompt_file),
        native_bfcl_functions=args.native_bfcl_functions,
    )
    entries, manifest = prepare(args, bfcl)
    root = run_root(args)
    if args.overwrite:
        clean_model_outputs(args)

    generation_args = SimpleNamespace(
        model=[args.model_id],
        test_category=list(ALLOCATION_PROFILES[args.profile]),
        temperature=args.temperature,
        include_input_log=args.include_input_log,
        exclude_state_log=args.exclude_state_log,
        num_threads=args.num_threads,
        num_gpus=1,
        backend="vllm",
        gpu_memory_utilization=0.9,
        result_dir=result_dir(args),
        run_ids=False,
        allow_overwrite=args.overwrite,
        skip_server_setup=True,
        local_model_path=None,
    )
    test_categories = sorted(set(manifest["scored_by_category"]) | {
        category.replace("_prereq", "")
        for category in manifest["auxiliary_by_category"]
    })
    test_cases = bfcl.generation.collect_test_cases(
        generation_args,
        args.model_id,
        test_categories,
        deepcopy(entries),
    )
    print(
        f"Generating {len(test_cases)} BFCL entries for {args.model_id} "
        f"({manifest['scored_total']} scored, {manifest['auxiliary_total']} auxiliary)."
    )
    bfcl.generation.generate_results(generation_args, args.model_id, test_cases)
    for model_result_json in result_dir(args).rglob("BFCL_v4_*_result.json"):
        bfcl.utils.sort_file_content_by_id(model_result_json)
    print(f"Results written under {result_dir(args) / args.model_id.replace('/', '_')}")


def evaluate(args: argparse.Namespace, bfcl: BfclModules) -> None:
    os.environ["OPENAI_BASE_URL"] = args.endpoint
    os.environ["OPENAI_API_KEY"] = args.api_key
    register_known_evk_models(bfcl)
    register_openai_endpoint(
        bfcl,
        args.model_id,
        args.endpoint_model or args.model_id,
        args.display_name or args.model_id,
        native_bfcl_functions=args.native_bfcl_functions,
    )
    manifest_path = run_root(args) / "subset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        categories = sorted(
            set(manifest.get("scored_by_category", {}))
            | {
                category.replace("_prereq", "")
                for category in manifest.get("auxiliary_by_category", {})
            }
        )
    else:
        categories = list(ALLOCATION_PROFILES[args.profile])
    bfcl.eval_runner.main(
        [args.model_id],
        categories,
        result_dir(args),
        score_dir(args),
        partial_eval=True,
    )
    summary = summarize(args)
    print(json.dumps(summary["overall"], indent=2))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_for_category(category: str) -> str:
    for group, categories in GROUPS.items():
        if category in categories:
            return group
    return "other"


def score_files(args: argparse.Namespace) -> Iterable[Path]:
    model_dir_name = args.model_id.replace("/", "_")
    base = score_dir(args) / model_dir_name
    if not base.exists():
        return []
    return sorted(base.rglob("BFCL_v4_*_score.json"))


def runtime_failure_rows(args: argparse.Namespace) -> list[dict]:
    model_dir_name = args.model_id.replace("/", "_")
    base = result_dir(args) / model_dir_name
    failures: list[dict] = []
    if not base.exists():
        return failures
    for path in sorted(base.rglob("BFCL_v4_*_result.json")):
        for row in read_jsonl(path):
            result = row.get("result")
            if RUNTIME_FAILURE_TOOL_NAME not in json.dumps(result, sort_keys=True):
                continue
            status = "runtime_failure"
            if isinstance(result, list):
                for item in result:
                    if not isinstance(item, dict) or RUNTIME_FAILURE_TOOL_NAME not in item:
                        continue
                    try:
                        detail = json.loads(item[RUNTIME_FAILURE_TOOL_NAME])
                    except (TypeError, json.JSONDecodeError):
                        detail = {}
                    status = str(detail.get("status", status))
                    break
            failures.append(
                {
                    "id": row.get("id"),
                    "status": status,
                    "latency": row.get("latency"),
                    "result_file": str(path),
                }
            )
    return failures


def summarize(args: argparse.Namespace) -> dict:
    root = run_root(args)
    model_id = args.model_id.replace("/", "_")
    category_rows: list[dict] = []
    group_totals: dict[str, dict[str, float]] = {}
    overall_correct = 0.0
    overall_total = 0

    for path in score_files(args):
        entries = read_jsonl(path)
        if not entries:
            continue
        header = entries[0]
        category = path.name.removeprefix("BFCL_v4_").removesuffix("_score.json")
        total = int(header.get("total_count", 0))
        correct = float(header.get("correct_count", header.get("accuracy", 0) * total))
        accuracy = float(header.get("accuracy", 0))
        group = group_for_category(category)
        category_rows.append(
            {
                "category": category,
                "group": group,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "failures_recorded": max(0, len(entries) - 1),
                "score_file": str(path),
            }
        )
        totals = group_totals.setdefault(group, {"correct": 0.0, "total": 0})
        totals["correct"] += correct
        totals["total"] += total
        overall_correct += correct
        overall_total += total

    group_rows = []
    for group, totals in sorted(group_totals.items()):
        total = int(totals["total"])
        correct = float(totals["correct"])
        group_rows.append(
            {
                "group": group,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )

    runtime_failures = runtime_failure_rows(args)
    summary = {
        "model_id": args.model_id,
        "run_root": str(root),
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": overall_correct / overall_total if overall_total else 0.0,
        },
        "groups": group_rows,
        "categories": sorted(category_rows, key=lambda row: (row["group"], row["category"])),
        "runtime_failures": {
            "count": len(runtime_failures),
            "cases": runtime_failures,
            "scoring": "reserved invalid tool calls; always non-passing",
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / f"summary_{model_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown_summary(root / f"summary_{model_id}.md", summary)
    return summary


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_count(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def write_markdown_summary(path: Path, summary: dict) -> None:
    lines = [
        f"# BFCL V4 Subset Summary: {summary['model_id']}",
        "",
        f"Overall: {format_count(summary['overall']['correct'])}/{summary['overall']['total']} "
        f"({format_pct(summary['overall']['accuracy'])})",
        "",
        f"Runtime failures: {summary['runtime_failures']['count']} "
        "(encoded as reserved invalid tool calls and counted as non-passing).",
        "",
        "## Groups",
        "",
        "| Group | Correct | Total | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| {row['group']} | {format_count(row['correct'])} | {row['total']} | "
            f"{format_pct(row['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Categories",
            "",
            "| Category | Group | Correct | Total | Accuracy | Failures Recorded |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["categories"]:
        lines.append(
            f"| {row['category']} | {row['group']} | {format_count(row['correct'])} | "
            f"{row['total']} | {format_pct(row['accuracy'])} | {row['failures_recorded']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    bfcl = import_bfcl(args.project_root)

    if args.command == "prepare":
        _, manifest = prepare(args, bfcl)
        print(json.dumps({k: manifest[k] for k in ["scored_total", "auxiliary_total", "scored_by_category", "auxiliary_by_category"]}, indent=2))
    elif args.command == "generate":
        generate(args, bfcl)
    elif args.command == "evaluate":
        evaluate(args, bfcl)
    elif args.command == "summarize":
        print(json.dumps(summarize(args), indent=2))
    elif args.command == "run":
        generate(args, bfcl)
        evaluate(args, bfcl)


if __name__ == "__main__":
    main()
