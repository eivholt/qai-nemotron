#!/usr/bin/env python3
"""Run the EVK benchmark cases against Hugging Face models on the host GPU."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EVK_RUNNER = REPO_ROOT / "evk_bench" / "run_genie_bench.py"
spec = importlib.util.spec_from_file_location("evk_runner", EVK_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {EVK_RUNNER}")
evk_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evk_runner)


MODEL_IDS = {
    "nemotron": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "stock": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def resolve_prompt_profile(case: dict[str, Any], prompt_profile: str) -> str:
    if prompt_profile == "best_practical":
        return "strict_v2" if case["category"] in {"linux", "http"} else "native"
    if prompt_profile == "best_practical_v2":
        return "direct_final" if case["category"] in {"linux", "http"} else "native"
    if prompt_profile == "best_practical_v3":
        if case["category"] == "linux":
            return "direct_final"
        if case["category"] == "http":
            return "strict_v2"
        return "native"
    if prompt_profile == "best_practical_v4":
        return "final_only"
    return prompt_profile


def user_body(case: dict[str, Any], prompt_profile: str) -> str:
    effective = resolve_prompt_profile(case, prompt_profile)
    text = case["prompt"]
    if effective == "native":
        return text
    if effective == "direct_final":
        return (
            "Return exactly one final artifact. Do not include analysis, "
            "self-correction, alternatives, placeholders, or follow-up prose. If "
            "the task asks for JSON, output JSON only. If the task asks for code, "
            "output one code block only.\n\n"
            f"{text}"
        )
    if effective == "final_only":
        return (
            f"{evk_runner.FINAL_ONLY_POLICY}\n"
            f"Task:\n{text}"
        )
    if effective == "native_final":
        return (
            "Answer the task below. If you reason, put reasoning inside "
            "<think>...</think>. Put the final answer after </think>, and put only "
            "the requested final format there.\n\n"
            f"{text}"
        )
    if effective == "legacy":
        return (
            "Answer with only the command or commands. Do not use markdown fences. "
            "Do not explain.\n\n"
            f"{text}"
        )
    if effective == "strict_v2":
        return f"{evk_runner.STRICT_COMMAND_POLICY}\nRequest: {text}"
    if effective == "strict_fewshot":
        return f"{evk_runner.STRICT_FEWSHOT_POLICY}\nRequest: {text}\nAnswer:"
    raise ValueError(f"Unknown prompt profile: {prompt_profile}")


def build_chat_prompt(
    tokenizer: Any,
    model_kind: str,
    mode: str,
    case: dict[str, Any],
    prompt_profile: str,
) -> tuple[str, str]:
    if model_kind == "nemotron":
        system = "detailed thinking on" if mode == "thinking_on" else "detailed thinking off"
    else:
        system = "You are a concise Linux and HTTP command assistant."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_body(case, prompt_profile)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if model_kind == "nemotron" and mode == "thinking_off":
        prompt += "<think>\n</think>\n"
    return prompt, resolve_prompt_profile(case, prompt_profile)


def eos_ids(tokenizer: Any) -> list[int]:
    ids = []
    for token in ["<|eot_id|>", "<|end_of_text|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted(set(ids))


def clean_generation(text: str) -> str:
    for marker in ["<|eot_id|>", "<|end_of_text|>"]:
        text = text.replace(marker, "")
    return text.strip()


def generation_kwargs(mode: str, max_new_tokens: int) -> dict[str, Any]:
    if mode == "thinking_on":
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
        }
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }


def run_cases(args: argparse.Namespace) -> int:
    model_id = MODEL_IDS.get(args.model_kind, args.model_id or "")
    if not model_id:
        raise SystemExit("--model-id is required for custom model_kind")

    cases = list(evk_runner.SUITES[args.suite])
    if args.case_ids:
        wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted]
    if args.categories:
        wanted = {item.strip() for item in args.categories.split(",") if item.strip()}
        cases = [case for case in cases if case["category"] in wanted]
    if not cases:
        raise SystemExit("No cases matched filters")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{timestamp}__{args.model_name}__{args.mode}"
    result_dir = args.out_root.expanduser() / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=args.local_files_only
    )
    print(f"Loading model {model_id}", flush=True)
    started_load = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    model.to(args.device).eval()
    load_s = time.monotonic() - started_load
    print(f"Loaded in {load_s:.1f}s", flush=True)

    results: list[dict[str, Any]] = []
    stop_ids = eos_ids(tokenizer)
    for index, case in enumerate(cases, 1):
        print(f"Running {args.model_name} {args.mode} {case['id']} ({index}/{len(cases)})", flush=True)
        prompt, effective_prompt_profile = build_chat_prompt(
            tokenizer, args.model_kind, args.mode, case, args.prompt_profile
        )
        prompt_path = result_dir / f"{args.model_name}__{args.mode}__{case['id']}.prompt.txt"
        prompt_path.write_text(prompt)

        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs(args.mode, args.max_new_tokens),
            )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_s = time.monotonic() - start

        input_tokens = inputs["input_ids"].shape[1]
        new_tokens = generated.shape[1] - input_tokens
        raw_answer = tokenizer.decode(
            generated[0, input_tokens:], skip_special_tokens=False
        )
        raw_answer = clean_generation(raw_answer)
        reasoning_split = evk_runner.split_reasoning(raw_answer)
        answer = raw_answer if args.score_full_answer else reasoning_split["final_answer"]
        score = evk_runner.score_answer(case, answer)
        decode_tps = new_tokens / elapsed_s if elapsed_s else 0

        log_path = result_dir / f"{args.model_name}__{args.mode}__{case['id']}.answer.txt"
        log_path.write_text(raw_answer)
        results.append(
            {
                "model": args.model_name,
                "model_id": model_id,
                "mode": args.mode,
                "case_id": case["id"],
                "category": case["category"],
                "returncode": 0,
                "elapsed_s": round(elapsed_s, 3),
                "input_tokens": input_tokens,
                "new_tokens": new_tokens,
                "answer": answer,
                "raw_answer": raw_answer,
                "reasoning": reasoning_split["reasoning"],
                "reasoning_open": reasoning_split["reasoning_open"],
                "has_think_tags": reasoning_split["has_think_tags"],
                "score_basis": "raw_answer" if args.score_full_answer else "final_answer",
                "score": score,
                "profile": {
                    "token-generation-rate": decode_tps,
                    "load_s": load_s,
                    "host_elapsed_s": elapsed_s,
                },
                "prompt_profile": args.prompt_profile,
                "effective_prompt_profile": effective_prompt_profile,
                "paths": {
                    "prompt": str(prompt_path),
                    "log": str(log_path),
                },
            }
        )

    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    (result_dir / "summary.md").write_text(evk_runner.summarize(results))
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())

    del model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=["nemotron", "stock", "custom"], required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=["stock", "thinking_off", "thinking_on"], required=True)
    parser.add_argument("--suite", choices=sorted(evk_runner.SUITES), default="all")
    parser.add_argument("--prompt-profile", default="best_practical")
    parser.add_argument("--case-ids")
    parser.add_argument("--categories")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "host_bench_results")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--score-full-answer", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run_cases(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
