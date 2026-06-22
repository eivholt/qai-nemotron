#!/usr/bin/env python3
"""Model clients shared by the agent arena demos."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def qairt_env() -> dict[str, str]:
    env = os.environ.copy()
    qairt_home = env.get("QAIRT_HOME", "/opt/qairt/current")
    target = env.get("QAIRT_TARGET", "aarch64-oe-linux-gcc11.2")
    env.update(
        {
            "QAIRT_HOME": qairt_home,
            "QAIRT_SDK_ROOT": qairt_home,
            "QNN_SDK_ROOT": qairt_home,
            "QAIRT_TARGET": target,
            "PRODUCT_SOC": env.get("PRODUCT_SOC", "9075"),
            "DSP_ARCH": env.get("DSP_ARCH", "73"),
            "ADSP_LIBRARY_PATH": env.get(
                "ADSP_LIBRARY_PATH", f"{qairt_home}/lib/hexagon-v73/unsigned"
            ),
            "LD_LIBRARY_PATH": env.get(
                "LD_LIBRARY_PATH",
                f"{qairt_home}/lib/{target}:{qairt_home}/lib/aarch64-oe-linux-gcc11.2:"
                f"{qairt_home}/lib/aarch64-oe-linux-gcc8.2:"
                "/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu",
            ),
        }
    )
    env["PATH"] = (
        f"{qairt_home}/bin/{target}:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    return env


def build_chat_prompt(system_text: str, user_text: str, assistant_prefill: str = "") -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_prefill}"
    )


def extract_genie_answer(raw_output: str) -> str:
    match = re.search(r"\[BEGIN\]:(.*?)\[END\]", raw_output, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_output.strip()


def split_reasoning(answer: str) -> dict[str, Any]:
    text = answer.strip()
    lower = text.lower()
    open_tag = "<think>"
    close_tag = "</think>"
    open_idx = lower.find(open_tag)
    close_idx = lower.rfind(close_tag)
    if open_idx == -1:
        return {
            "reasoning": "",
            "final_answer": text if close_idx == -1 else text[close_idx + len(close_tag) :].strip(),
            "reasoning_open": False,
            "has_think_tags": close_idx != -1,
        }
    if close_idx == -1 or close_idx < open_idx:
        return {
            "reasoning": text[open_idx + len(open_tag) :].strip(),
            "final_answer": "",
            "reasoning_open": True,
            "has_think_tags": True,
        }
    return {
        "reasoning": text[open_idx + len(open_tag) : close_idx].strip(),
        "final_answer": text[close_idx + len(close_tag) :].strip(),
        "reasoning_open": False,
        "has_think_tags": True,
    }


def extract_first_json(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def extract_python_code(text: str) -> str:
    match = re.search(r"```python\s+(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    generic = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return text.strip()


@dataclass
class Generation:
    raw_text: str
    answer: str
    elapsed_s: float
    returncode: int
    profile: dict[str, Any]
    paths: dict[str, str]


class GenieClient:
    def __init__(
        self,
        bundle: Path,
        model: str,
        mode: str,
        work_dir: Path,
        timeout_s: int = 180,
    ) -> None:
        self.bundle = bundle.expanduser().resolve()
        self.model = model
        self.mode = mode
        self.work_dir = work_dir
        self.timeout_s = timeout_s
        self.counter = 0
        if not (self.bundle / "genie_config.json").exists():
            raise FileNotFoundError(f"Missing genie_config.json in {self.bundle}")
        if shutil.which("genie-t2t-run", path=qairt_env()["PATH"]) is None:
            raise FileNotFoundError("genie-t2t-run not found in QAIRT PATH")

    def system_text(self) -> str:
        if self.mode == "thinking_on":
            return (
                "detailed thinking on\n"
                "Use a short reasoning budget for agent-arena tasks. Close any "
                "<think> block quickly, then provide the requested final artifact."
            )
        if self.mode == "thinking_off":
            return (
                "detailed thinking off\n"
                "Provide only the requested final artifact."
            )
        return "You are a concise agent that follows the requested protocol exactly."

    def assistant_prefill(self) -> str:
        if self.mode == "thinking_off":
            return "<think>\n</think>\n"
        return ""

    def generate(self, user_text: str, tag: str) -> Generation:
        self.counter += 1
        stem = f"{self.model}__{self.mode}__{tag}__step{self.counter:02d}"
        prompt_path = self.work_dir / f"{stem}.prompt.txt"
        profile_path = self.work_dir / f"{stem}.profile.json"
        log_path = self.work_dir / f"{stem}.log"
        prompt_path.write_text(
            build_chat_prompt(self.system_text(), user_text, self.assistant_prefill())
        )
        command = [
            "genie-t2t-run",
            "-c",
            "genie_config.json",
            "--prompt_file",
            str(prompt_path),
            "--profile",
            str(profile_path),
        ]
        started = time.monotonic()
        proc = subprocess.run(
            command,
            cwd=self.bundle,
            env=qairt_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_s,
        )
        elapsed_s = time.monotonic() - started
        log_path.write_text(proc.stdout)
        raw = extract_genie_answer(proc.stdout)
        split = split_reasoning(raw)
        answer = split["final_answer"]
        profile: dict[str, Any] = {}
        if profile_path.exists():
            try:
                profile = json.loads(profile_path.read_text())
            except json.JSONDecodeError:
                profile = {}
        return Generation(
            raw_text=raw,
            answer=answer,
            elapsed_s=elapsed_s,
            returncode=proc.returncode,
            profile=profile,
            paths={
                "prompt": str(prompt_path),
                "log": str(log_path),
                "profile": str(profile_path),
            },
        )


def score_text(answer: str, required_regex: list[str]) -> dict[str, Any]:
    missing = [
        pattern
        for pattern in required_regex
        if not re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
    ]
    score = (len(required_regex) - len(missing)) / len(required_regex) if required_regex else 1.0
    return {"passed": not missing, "score": round(score, 3), "missing": missing}


def write_summary(path: Path, title: str, results: list[dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "| model | mode | cases | pass | avg score |",
        "|---|---|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault((item["model"], item["mode"]), []).append(item)
    for (model, mode), items in sorted(grouped.items()):
        passed = sum(1 for item in items if item["score"]["passed"])
        avg = sum(item["score"]["score"] for item in items) / len(items)
        lines.append(f"| {model} | {mode} | {len(items)} | {passed} | {avg:.3f} |")
    lines.extend(["", "## Cases", ""])
    lines.append("| case | model | mode | score | pass | failure | notes |")
    lines.append("|---|---|---|---:|---|---|---|")
    for item in results:
        notes = item.get("notes", "")
        failure = item.get("failure_kind", "")
        lines.append(
            f"| {item['case_id']} | {item['model']} | {item['mode']} | "
            f"{item['score']['score']:.3f} | {'yes' if item['score']['passed'] else 'no'} | "
            f"{failure} | {notes} |"
        )
    path.write_text("\n".join(lines) + "\n")
