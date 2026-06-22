#!/usr/bin/env python3
"""Run practical Genie command-generation benchmarks on the IQ-9075 EVK."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMMAND_CASES: list[dict[str, Any]] = [
    {
        "id": "linux_env_var",
        "category": "linux",
        "prompt": (
            "On Ubuntu bash, give the single best command to print the current "
            "value of the environment variable FOO."
        ),
        "required_any": [[r"\becho\s+\$FOO\b", r"\bprintenv\s+FOO\b"]],
        "forbidden": [r"\bsource\s+FOO\b", r"\benv\s+FOO\b"],
    },
    {
        "id": "linux_find_recent_large_logs",
        "category": "linux",
        "prompt": (
            "Give one Ubuntu command that finds regular files under /var/log "
            "larger than 10 MB and modified in the last 7 days."
        ),
        "required_any": [
            [r"\bfind\b"],
            [r"/var/log"],
            [r"-type\s+f"],
            [r"-size\s+\+10M"],
            [r"-mtime\s+-7", r"-newermt"],
        ],
    },
    {
        "id": "linux_journal_errors",
        "category": "linux",
        "prompt": (
            "Give one Ubuntu command to show only error-priority logs for the "
            "ssh service from the last hour, without opening a pager."
        ),
        "required_any": [
            [r"\bjournalctl\b"],
            [r"-u\s+ssh", r"-u\s+sshd"],
            [r"-p\s+err", r"-p\s+3"],
            [r"--since"],
            [r"--no-pager"],
        ],
    },
    {
        "id": "linux_du_sorted_home",
        "category": "linux",
        "prompt": (
            "Give one Ubuntu command to show first-level disk usage for folders "
            "in the current user's home directory, human-readable, sorted by size."
        ),
        "required_any": [
            [r"\bdu\b"],
            [r"-h"],
            [r"--max-depth=1", r"-d\s*1"],
            [r"~", r"\$HOME"],
            [r"\bsort\b"],
        ],
    },
    {
        "id": "linux_fastrpc_group",
        "category": "linux",
        "prompt": (
            "On Ubuntu, give the command to add user ubuntu to the fastrpc group "
            "without removing existing group memberships."
        ),
        "required_any": [
            [r"\busermod\b"],
            [r"-aG\s+fastrpc", r"-a\s+-G\s+fastrpc"],
            [r"\bubuntu\b"],
        ],
        "forbidden": [r"\busermod\s+-G\s+fastrpc\b"],
    },
    {
        "id": "http_get_status",
        "category": "http",
        "prompt": (
            "Give one curl command that makes an HTTP GET request to "
            "https://example.com, fails on HTTP errors, waits at most 5 seconds "
            "to connect, discards the body, and prints only the HTTP status code."
        ),
        "required_any": [
            [r"\bcurl\b"],
            [r"-f", r"--fail"],
            [r"--connect-timeout\s+5"],
            [r"-o\s+/dev/null"],
            [r"-w\s+['\"]?%[{]http_code[}]", r"--write-out\s+['\"]?%[{]http_code[}]"],
            [r"https://example\.com"],
        ],
    },
    {
        "id": "http_post_json",
        "category": "http",
        "prompt": (
            "Give one curl command that POSTs JSON {\"name\":\"evk\"} to "
            "https://httpbin.org/post and sets the correct JSON content type."
        ),
        "required_any": [
            [r"\bcurl\b"],
            [r"-X\s+POST", r"--request\s+POST"],
            [r"Content-Type:\s*application/json"],
            [r"\{\"name\"\s*:\s*\"evk\"\}", r"\{'name'\s*:\s*'evk'\}"],
            [r"https://httpbin\.org/post"],
        ],
    },
    {
        "id": "http_headers_only",
        "category": "http",
        "prompt": (
            "Give one curl command to fetch only the response headers from "
            "https://example.com."
        ),
        "required_any": [
            [r"\bcurl\b"],
            [r"-I\b", r"--head\b", r"-sSI\b", r"-sI\b"],
            [r"https://example\.com"],
        ],
    },
    {
        "id": "http_bearer_header",
        "category": "http",
        "prompt": (
            "Give one curl command that sends a GET request to "
            "https://api.example.com/v1/devices with an Authorization bearer "
            "token stored in the shell variable API_TOKEN."
        ),
        "required_any": [
            [r"\bcurl\b"],
            [r"Authorization:\s*Bearer\s+\$API_TOKEN"],
            [r"https://api\.example\.com/v1/devices"],
        ],
        "forbidden": [r"API_TOKEN=", r"Bearer\s+[A-Za-z0-9_-]{12,}"],
    },
    {
        "id": "http_jq_extract",
        "category": "http",
        "prompt": (
            "Give one shell pipeline that downloads JSON from "
            "https://httpbin.org/json and prints only .slideshow.title using jq."
        ),
        "required_any": [
            [r"\bcurl\b"],
            [r"https://httpbin\.org/json"],
            [r"\|\s*jq\b"],
            [r"\.slideshow\.title"],
        ],
    },
]


NEMOTRON_CASES: list[dict[str, Any]] = [
    {
        "id": "tool_http_status",
        "category": "tool_call",
        "scorer": "json_tool",
        "prompt": """<AVAILABLE_TOOLS>
[
  {
    "name": "http_get",
    "description": "Make an HTTP GET request.",
    "parameters": {
      "url": "string",
      "headers": "object",
      "return": "string"
    }
  },
  {
    "name": "run_shell",
    "description": "Run a safe read-only Ubuntu shell command.",
    "parameters": {
      "command": "string"
    }
  }
]
</AVAILABLE_TOOLS>

Fetch https://example.com and return only the HTTP status code.

Return exactly one JSON object with keys "name" and "parameters".""",
        "expected_tool": "http_get",
        "expected_fields": {
            "parameters.url": "https://example.com",
            "parameters.return": "status_code",
        },
    },
    {
        "id": "tool_profile_parse",
        "category": "tool_call",
        "scorer": "json_tool",
        "prompt": """<AVAILABLE_TOOLS>
[
  {
    "name": "read_genie_profile",
    "description": "Read a Genie profile JSON file and extract LLM latency metrics.",
    "parameters": {
      "path": "string",
      "metrics": "array"
    }
  },
  {
    "name": "run_shell",
    "description": "Run a safe read-only Ubuntu shell command.",
    "parameters": {
      "command": "string"
    }
  }
]
</AVAILABLE_TOOLS>

Read /home/ubuntu/nemotron_genie/profile.txt and extract time-to-first-token and token-generation-rate.

Return exactly one JSON object with keys "name" and "parameters".""",
        "expected_tool": "read_genie_profile",
        "expected_fields": {
            "parameters.path": "/home/ubuntu/nemotron_genie/profile.txt",
        },
        "required_any": [
            [r"time-to-first-token"],
            [r"token-generation-rate"],
        ],
    },
    {
        "id": "diag_libatomic",
        "category": "diagnostic",
        "scorer": "regex",
        "prompt": """A Genie run on Ubuntu fails with:

libatomic.so.1: cannot open shared object file
Qnn getQnnSystemInterface FAILED

Return a JSON array of 3 to 5 Ubuntu shell commands that diagnose and fix the library path. The commands may include one final edit command, but the earlier commands should be read-only.""",
        "required_any": [
            [r"find\s+/opt/qairt", r"ldconfig\s+-p.*libatomic", r"locate\s+libatomic"],
            [r"libatomic\.so\.1"],
            [r"LD_LIBRARY_PATH"],
            [r"aarch64-oe-linux-gcc8\.2"],
            [r"qairt-env\.sh"],
        ],
    },
    {
        "id": "diag_htp_device",
        "category": "diagnostic",
        "scorer": "regex",
        "prompt": """Genie fails on an IQ-9075 EVK with:

Failed to create device: 14001
Device Creation failure

Return a JSON array of 4 to 6 Ubuntu shell commands to diagnose whether FastRPC, the cDSP daemon, device permissions, and the QNN HTP backend are healthy.""",
        "required_any": [
            [r"qnn-platform-validator.*--backend\s+dsp.*--testBackend", r"qnn-platform-validator.*--backend\s+dsp"],
            [r"systemctl.*cdsprpcd", r"journalctl.*cdsprpcd"],
            [r"/dev/fastrpc-cdsp"],
            [r"\bid\b", r"\bgroups\b"],
            [r"ADSP_LIBRARY_PATH", r"qairt-env\.sh"],
        ],
    },
    {
        "id": "math_decode_rate",
        "category": "math",
        "scorer": "boxed_number",
        "prompt": """Below is a math question. I want you to reason through the steps and then give a final answer. Your final answer should be in \\boxed{}.
Question: A Genie profile reports 31 generated tokens and token-generation-time of 3.1 seconds. What is the generated tokens per second rate?""",
        "expected_number": 10.0,
        "tolerance": 0.05,
    },
    {
        "id": "math_ttft_ms",
        "category": "math",
        "scorer": "boxed_number",
        "prompt": """Below is a math question. I want you to reason through the steps and then give a final answer. Your final answer should be in \\boxed{}.
Question: A Genie profile reports time-to-first-token as 183500 microseconds. What is that latency in milliseconds?""",
        "expected_number": 183.5,
        "tolerance": 0.05,
    },
    {
        "id": "code_parse_profile",
        "category": "code",
        "scorer": "python_code",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function parse_profile(profile_json) that returns a dict with keys "ttft_ms" and "decode_tps" from a Genie profile JSON dict. The input has components[].events[] entries, and the query event has fields "time-to-first-token" and "token-generation-rate", each containing a dict with a "value".
Please return all completed code in one python code block.""",
        "required_any": [
            [r"def\s+parse_profile\s*\("],
            [r"time-to-first-token"],
            [r"token-generation-rate"],
            [r"ttft_ms"],
            [r"decode_tps"],
        ],
    },
    {
        "id": "code_build_prompt",
        "category": "code",
        "scorer": "python_code",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function build_llama31_prompt(system_text, user_text) that returns a Llama 3.1 chat-template string using <|begin_of_text|>, system header, user header, assistant header, and <|eot_id|> turn separators.
Please return all completed code in one python code block.""",
        "required_any": [
            [r"def\s+build_llama31_prompt\s*\("],
            [r"<\|begin_of_text\|>"],
            [r"<\|start_header_id\|>system<\|end_header_id\|>"],
            [r"<\|start_header_id\|>user<\|end_header_id\|>"],
            [r"<\|start_header_id\|>assistant<\|end_header_id\|>"],
            [r"<\|eot_id\|>"],
        ],
    },
    {
        "id": "ifeval_three_checks",
        "category": "instruction_following",
        "scorer": "constraint_checks",
        "prompt": """Explain how to verify that Genie is using QnnHtp instead of CPU.

Constraints:
- Answer in exactly three bullet points.
- Each bullet must start with CHECK:
- Do not mention Python.
- Mention profile.txt exactly once.""",
        "constraints": {
            "exact_check_lines": 3,
            "must_include": [r"QnnHtp", r"profile\.txt"],
            "must_not_include": [r"Python"],
            "exact_counts": {r"profile\.txt": 1},
        },
    },
    {
        "id": "ifeval_json_status",
        "category": "instruction_following",
        "scorer": "json_exact_keys",
        "prompt": """Return exactly one JSON object and no prose.

The object must have exactly these keys:
- status
- command
- reason

Use it to recommend the safest first command for checking whether the cDSP daemon is active on Ubuntu.""",
        "expected_keys": ["status", "command", "reason"],
        "required_any": [
            [r"systemctl"],
            [r"cdsprpcd"],
        ],
    },
    {
        "id": "code_run_length_encode",
        "category": "coding_unit",
        "scorer": "python_unit_tests",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function run_length_encode(items) that returns a list of [value, count] pairs for each consecutive run in the input list. It must work for an empty list and for values of any equality-comparable type.
Please return all completed code in one python code block.""",
        "unit_tests": """
assert run_length_encode([]) == []
assert run_length_encode(["a", "a", "b", "a", "a", "a"]) == [["a", 2], ["b", 1], ["a", 3]]
assert run_length_encode([1, 1, 1, 2, 3, 3]) == [[1, 3], [2, 1], [3, 2]]
""",
        "required_any": [[r"def\s+run_length_encode\s*\("]],
    },
    {
        "id": "code_merge_intervals",
        "category": "coding_unit",
        "scorer": "python_unit_tests",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function merge_intervals(intervals) that accepts a list of [start, end] integer intervals and returns a new list with all overlapping or touching intervals merged. The returned intervals must be sorted by start.
Please return all completed code in one python code block.""",
        "unit_tests": """
assert merge_intervals([]) == []
assert merge_intervals([[5, 7], [1, 3], [2, 4], [8, 8]]) == [[1, 4], [5, 8]]
assert merge_intervals([[1, 1], [2, 2], [4, 6], [6, 9]]) == [[1, 2], [4, 9]]
""",
        "required_any": [[r"def\s+merge_intervals\s*\("]],
    },
    {
        "id": "code_parse_kv_lines",
        "category": "coding_unit",
        "scorer": "python_unit_tests",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function parse_kv_lines(text) that parses newline-separated key=value records. Ignore blank lines and lines without '='. Strip whitespace around keys and values. If a key appears multiple times, the later value wins.
Please return all completed code in one python code block.""",
        "unit_tests": """
sample = " alpha = 1\\nignored\\nbeta=two\\nalpha = final\\n\\n gamma = spaced value "
assert parse_kv_lines(sample) == {"alpha": "final", "beta": "two", "gamma": "spaced value"}
assert parse_kv_lines("") == {}
""",
        "required_any": [[r"def\s+parse_kv_lines\s*\("]],
    },
    {
        "id": "rag_widget_policy",
        "category": "rag",
        "scorer": "json_fields",
        "prompt": """Use only the reference text below. Return exactly one JSON object with keys "owner", "escalate", and "retention_days".

Reference:
The fictional HelioWidget support policy says warranty cases with code HW-17 are owned by Team Indigo. If the customer reports a smoke smell, the case must be escalated. Logs for HW-17 cases are retained for 45 days. Cases with code HW-22 are owned by Team Saffron and are not relevant here.

Question:
A customer opens a warranty case with code HW-17 and reports a smoke smell. Who owns it, should it be escalated, and how many days are logs retained?""",
        "expected_fields": {
            "owner": "Team Indigo",
            "escalate": True,
            "retention_days": 45,
        },
    },
    {
        "id": "rag_table_lookup",
        "category": "rag",
        "scorer": "json_fields",
        "prompt": """Use only the table below. Return exactly one JSON object with keys "item", "rack", and "backup".

Table:
Item | Rack | Backup
Flux pin | R2 | blue crate
Amber lens | R5 | gray drawer
Cobalt seal | R3 | red pouch

Question:
Where is the Amber lens stored and what is its backup location?""",
        "expected_fields": {
            "item": "Amber lens",
            "rack": "R5",
            "backup": "gray drawer",
        },
    },
    {
        "id": "reasoning_boxed_choice",
        "category": "logical_reasoning",
        "scorer": "boxed_choice",
        "prompt": """What is the correct answer to this question:
Exactly one of these statements is true.
A. Statement B is true.
B. Statement C is true.
C. Statements A and B are both false.
D. Statements A, B, and C are all true.
Let's think step by step, and put the final answer (a single letter A, B, C, or D) into \\boxed{}.""",
        "expected_choice": "C",
    },
    {
        "id": "reasoning_schedule",
        "category": "logical_reasoning",
        "scorer": "boxed_number",
        "prompt": """Below is a math and scheduling question. I want you to reason through the steps and then give a final answer. Your final answer should be in \\boxed{}.
Question: A worker starts at minute 0. Task A takes 7 minutes. Task B starts after A and takes 5 minutes. A mandatory inspection then takes 3 minutes. What minute does the inspection finish?""",
        "expected_number": 15,
        "tolerance": 0.001,
    },
    {
        "id": "tool_select_lookup",
        "category": "tool_call",
        "scorer": "json_tool",
        "prompt": """<AVAILABLE_TOOLS>
[
  {
    "name": "lookup_record",
    "description": "Look up a record by id.",
    "parameters": {
      "record_id": "string",
      "fields": "array"
    }
  },
  {
    "name": "send_email",
    "description": "Send an email message.",
    "parameters": {
      "to": "string",
      "subject": "string"
    }
  }
]
</AVAILABLE_TOOLS>

Find record R-104 and return only its owner and status fields.

Return exactly one JSON object with keys "name" and "parameters".""",
        "expected_tool": "lookup_record",
        "expected_fields": {
            "parameters.record_id": "R-104",
        },
        "required_any": [[r"owner"], [r"status"]],
    },
    {
        "id": "code_top_k_counts",
        "category": "coding_unit",
        "scorer": "python_unit_tests",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function top_k_counts(items, k) that returns a list of [item, count] pairs for the k most common items. Sort by descending count, then by the string representation of the item for ties. If k is less than or equal to 0, return [].
Please return all completed code in one python code block.""",
        "unit_tests": """
assert top_k_counts(["b", "a", "b", "c", "a", "b"], 2) == [["b", 3], ["a", 2]]
assert top_k_counts(["x", "y", "x", "y"], 5) == [["x", 2], ["y", 2]]
assert top_k_counts([3, 3, 2, 1, 2], 2) == [[2, 2], [3, 2]]
assert top_k_counts(["a"], 0) == []
""",
        "required_any": [[r"def\s+top_k_counts\s*\("]],
    },
    {
        "id": "code_normalize_slug",
        "category": "coding_unit",
        "scorer": "python_unit_tests",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function normalize_slug(text) that lowercases text, replaces each run of non-alphanumeric characters with one hyphen, and strips leading/trailing hyphens.
Please return all completed code in one python code block.""",
        "unit_tests": """
assert normalize_slug("Hello, World!") == "hello-world"
assert normalize_slug("  A__B---C  ") == "a-b-c"
assert normalize_slug("Already123") == "already123"
assert normalize_slug("!!!") == ""
""",
        "required_any": [[r"def\s+normalize_slug\s*\("]],
    },
    {
        "id": "code_window_sums",
        "category": "coding_unit",
        "scorer": "python_unit_tests",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function window_sums(nums, size) that returns the sum of each consecutive window of length size. If size is <= 0 or larger than nums, return [].
Please return all completed code in one python code block.""",
        "unit_tests": """
assert window_sums([1, 2, 3, 4], 2) == [3, 5, 7]
assert window_sums([5, -1, 2], 3) == [6]
assert window_sums([1, 2], 3) == []
assert window_sums([1, 2], 0) == []
""",
        "required_any": [[r"def\s+window_sums\s*\("]],
    },
    {
        "id": "rag_access_matrix",
        "category": "rag",
        "scorer": "json_fields",
        "prompt": """Use only the reference text below. Return exactly one JSON object with keys "can_deploy", "approval", and "environment".

Reference:
In the fictional DeltaOps access matrix, role Builder may deploy to staging with peer approval. Role Builder may not deploy to production. Role Operator may restart staging services without approval. Role Auditor is read-only.

Question:
A user with role Builder wants to deploy to staging. Is it allowed, what approval is required, and which environment is involved?""",
        "expected_fields": {
            "can_deploy": True,
            "approval": "peer approval",
            "environment": "staging",
        },
    },
    {
        "id": "rag_incident_priority",
        "category": "rag",
        "scorer": "json_fields",
        "prompt": """Use only the reference text below. Return exactly one JSON object with keys "priority", "owner", and "notify".

Reference:
The fictional incident guide says: If a report mentions data loss and customer impact, priority is P1 and owner is Core Response. If it mentions latency only, priority is P2 and owner is Performance Desk. For P1 incidents, notify the duty manager.

Question:
The report says customers are impacted and data loss is suspected. What priority, owner, and notification are required?""",
        "expected_fields": {
            "priority": "P1",
            "owner": "Core Response",
            "notify": "duty manager",
        },
    },
    {
        "id": "reasoning_grid_position",
        "category": "logical_reasoning",
        "scorer": "boxed_choice",
        "prompt": """What is the correct answer to this question:
A token starts at square B2 on a 3 by 3 grid with columns A, B, C and rows 1, 2, 3. It moves up one row, right one column, down two rows, then left one column. Which square is it on?
A. A1
B. B1
C. C2
D. B3
Let's think step by step, and put the final answer (a single letter A, B, C, or D) into \\boxed{}.""",
        "expected_choice": "B",
    },
    {
        "id": "reasoning_weighted_average",
        "category": "logical_reasoning",
        "scorer": "boxed_number",
        "prompt": """Below is a math question. I want you to reason through the steps and then give a final answer. Your final answer should be in \\boxed{}.
Question: A test has two sections. Section A has weight 40% and score 80. Section B has weight 60% and score 95. What is the weighted average score?""",
        "expected_number": 89.0,
        "tolerance": 0.001,
    },
    {
        "id": "ifeval_acrostic",
        "category": "instruction_following",
        "scorer": "constraint_checks",
        "prompt": """Describe a careful code review.

Constraints:
- Answer in exactly three lines.
- Line 1 must start with PLAN:
- Line 2 must start with READ:
- Line 3 must start with TEST:
- Do not use the word quick.""",
        "constraints": {
            "line_prefixes": ["PLAN:", "READ:", "TEST:"],
            "must_not_include": [r"quick"],
        },
    },
    {
        "id": "ifeval_json_array",
        "category": "instruction_following",
        "scorer": "json_array_exact",
        "prompt": """Return exactly one JSON array and no prose.

The array must contain exactly these three strings in this order:
- alpha
- beta
- gamma""",
        "expected_array": ["alpha", "beta", "gamma"],
    },
    {
        "id": "tool_select_calculator",
        "category": "tool_call",
        "scorer": "json_tool",
        "prompt": """<AVAILABLE_TOOLS>
[
  {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression.",
    "parameters": {
      "expression": "string"
    }
  },
  {
    "name": "lookup_record",
    "description": "Look up a record by id.",
    "parameters": {
      "record_id": "string"
    }
  }
]
</AVAILABLE_TOOLS>

Calculate (17 + 5) * 3.

Return exactly one JSON object with keys "name" and "parameters".""",
        "expected_tool": "calculator",
        "expected_fields": {
            "parameters.expression": "(17 + 5) * 3",
        },
    },
    {
        "id": "neutral_diag_shared_library",
        "category": "neutral_diagnostic",
        "scorer": "regex",
        "prompt": """A command on Ubuntu fails with:

libexample.so.2: cannot open shared object file: No such file or directory

Return a JSON array of 3 to 5 Ubuntu shell commands that diagnose where the library is installed and temporarily add its directory to the library path. Earlier commands should be read-only.""",
        "required_any": [
            [r"find\s+/(usr|opt|lib)", r"ldconfig\s+-p.*libexample", r"locate\s+libexample"],
            [r"libexample\.so\.2"],
            [r"LD_LIBRARY_PATH"],
            [r"export\s+LD_LIBRARY_PATH"],
        ],
    },
    {
        "id": "neutral_diag_service_logs",
        "category": "neutral_diagnostic",
        "scorer": "regex",
        "prompt": """A service named sample-worker fails to start on Ubuntu.

Return a JSON array of 4 to 6 Ubuntu shell commands to inspect its systemd status, recent logs, whether its config file exists at /etc/sample-worker/config.yaml, and whether the current user can read that file.""",
        "required_any": [
            [r"systemctl\s+status\s+sample-worker"],
            [r"journalctl\s+-u\s+sample-worker", r"journalctl.*sample-worker"],
            [r"/etc/sample-worker/config\.yaml"],
            [r"ls\s+-l", r"test\s+-r", r"stat\b"],
        ],
    },
    {
        "id": "neutral_linux_service_logs",
        "category": "neutral_linux",
        "prompt": (
            "Give one Ubuntu command to show warning-or-higher logs for the "
            "sample-worker service from the last 30 minutes, without opening a pager."
        ),
        "required_any": [
            [r"\bjournalctl\b"],
            [r"-u\s+sample-worker"],
            [r"-p\s+warning", r"-p\s+4", r"-p\s+warn"],
            [r"--since"],
            [r"--no-pager"],
        ],
    },
    {
        "id": "neutral_linux_add_group",
        "category": "neutral_linux",
        "prompt": (
            "On Ubuntu, give the command to add user alex to the docker group "
            "without removing existing group memberships."
        ),
        "required_any": [
            [r"\busermod\b"],
            [r"-aG\s+docker", r"-a\s+-G\s+docker"],
            [r"\balex\b"],
        ],
        "forbidden": [r"\busermod\s+-G\s+docker\b"],
    },
    {
        "id": "neutral_http_status_local",
        "category": "neutral_http",
        "prompt": (
            "Give one curl command that makes an HTTP GET request to "
            "https://example.org/health, fails on HTTP errors, waits at most 3 seconds "
            "to connect, discards the body, and prints only the HTTP status code."
        ),
        "required_any": [
            [r"\bcurl\b"],
            [r"-f", r"--fail"],
            [r"--connect-timeout\s+3"],
            [r"-o\s+/dev/null"],
            [r"-w\s+['\"]?%[{]http_code[}]", r"--write-out\s+['\"]?%[{]http_code[}]"],
            [r"https://example\.org/health"],
        ],
    },
    {
        "id": "neutral_tool_fetch_json",
        "category": "neutral_tool_call",
        "scorer": "json_tool",
        "prompt": """<AVAILABLE_TOOLS>
[
  {
    "name": "fetch_json",
    "description": "Fetch JSON from a URL.",
    "parameters": {
      "url": "string",
      "timeout_seconds": "integer"
    }
  },
  {
    "name": "read_file",
    "description": "Read a local file.",
    "parameters": {
      "path": "string"
    }
  }
]
</AVAILABLE_TOOLS>

Fetch JSON from https://example.org/data.json with timeout 4 seconds.

Return exactly one JSON object with keys "name" and "parameters".""",
        "expected_tool": "fetch_json",
        "expected_fields": {
            "parameters.url": "https://example.org/data.json",
            "parameters.timeout_seconds": 4,
        },
    },
    {
        "id": "neutral_code_parse_profile",
        "category": "neutral_code",
        "scorer": "python_unit_tests",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function parse_runtime_report(report) that returns a dict with keys "startup_ms" and "items_per_second". The input dict has sections[].metrics[] entries. The metric with name "startup_ms" has a numeric "value"; the metric with name "items_per_second" has a numeric "value".
Please return all completed code in one python code block.""",
        "unit_tests": """
sample = {"sections": [{"metrics": [{"name": "other", "value": 0}]}, {"metrics": [{"name": "startup_ms", "value": 12.5}, {"name": "items_per_second", "value": 40}]}]}
assert parse_runtime_report(sample) == {"startup_ms": 12.5, "items_per_second": 40}
assert parse_runtime_report({"sections": []}) == {"startup_ms": None, "items_per_second": None}
""",
        "required_any": [[r"def\s+parse_runtime_report\s*\("]],
    },
    {
        "id": "neutral_code_build_chat_prompt",
        "category": "neutral_code",
        "scorer": "python_unit_tests",
        "prompt": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
Write a Python function build_simple_chat_prompt(system_text, user_text) that returns exactly:
SYSTEM: <system_text>
USER: <user_text>
ASSISTANT:
The returned string should end immediately after ASSISTANT: with no extra newline.
Please return all completed code in one python code block.""",
        "unit_tests": """
assert build_simple_chat_prompt("be brief", "hello") == "SYSTEM: be brief\\nUSER: hello\\nASSISTANT:"
assert build_simple_chat_prompt("", "x") == "SYSTEM: \\nUSER: x\\nASSISTANT:"
""",
        "required_any": [[r"def\s+build_simple_chat_prompt\s*\("]],
    },
    {
        "id": "neutral_ifeval_service_json",
        "category": "neutral_instruction_following",
        "scorer": "json_exact_keys",
        "prompt": """Return exactly one JSON object and no prose.

The object must have exactly these keys:
- status
- command
- reason

Use it to recommend the safest first command for checking whether an Ubuntu service named sample-worker is active.""",
        "expected_keys": ["status", "command", "reason"],
        "required_any": [[r"systemctl"], [r"sample-worker"]],
    },
    {
        "id": "neutral_reasoning_log_rate",
        "category": "neutral_reasoning",
        "scorer": "boxed_number",
        "prompt": """Below is a math question. I want you to reason through the steps and then give a final answer. Your final answer should be in \\boxed{}.
Question: A log processor reads 84 files in 7 minutes. If the rate stays constant, how many files does it read per minute?""",
        "expected_number": 12,
        "tolerance": 0.001,
    },
]


SUITES: dict[str, list[dict[str, Any]]] = {
    "command": COMMAND_CASES,
    "nemotron": NEMOTRON_CASES,
    "neutralized": NEMOTRON_CASES[-10:],
    "all": COMMAND_CASES + NEMOTRON_CASES,
}


STRICT_COMMAND_POLICY = """Task: Generate exactly one Ubuntu bash command.

Rules:
- Output only the command.
- Do not explain.
- Do not use markdown fences.
- Do not wrap the whole answer in quotes.
- Preserve shell variables literally, for example $FOO and $API_TOKEN.
- Do not invent placeholder values.
- Prefer standard Ubuntu tools and flags.
"""


STRICT_FEWSHOT_POLICY = (
    STRICT_COMMAND_POLICY
    + """
Examples of the expected answer style:

Request: Print the current value of the environment variable FOO.
Answer: echo "$FOO"

Request: Fetch only response headers from https://example.com.
Answer: curl -I https://example.com
"""
)


FINAL_ONLY_POLICY = """Return only the final artifact for the task.

Rules:
- Do not think out loud.
- Do not explain.
- Do not include notes, alternatives, corrections, or markdown unless the task explicitly asks for a code block.
- Start immediately with the requested artifact.
- For a shell command, output exactly one command line.
- For JSON, output exactly one valid JSON value and nothing else.
- For Python code, output exactly one ```python code block and nothing else.
- For a constrained natural-language answer, satisfy the constraints directly and stop.
"""


def build_prompt(
    system_text: str, user_text: str, prompt_profile: str, assistant_prefill: str = ""
) -> str:
    if prompt_profile == "native":
        user_body = user_text
    elif prompt_profile == "direct_final":
        user_body = (
            "Return exactly one final artifact. Do not include analysis, "
            "self-correction, alternatives, placeholders, or follow-up prose. If "
            "the task asks for JSON, output JSON only. If the task asks for code, "
            "output one code block only.\n\n"
            f"{user_text}"
        )
    elif prompt_profile == "final_only":
        user_body = f"{FINAL_ONLY_POLICY}\nTask:\n{user_text}"
    elif prompt_profile == "native_final":
        user_body = (
            "Answer the task below. If you reason, put reasoning inside "
            "<think>...</think>. Put the final answer after </think>, and put only "
            "the requested final format there.\n\n"
            f"{user_text}"
        )
    elif prompt_profile == "legacy":
        user_body = (
            "Answer with only the command or commands. Do not use markdown fences. "
            "Do not explain.\n\n"
            f"{user_text}"
        )
    elif prompt_profile == "strict_v2":
        user_body = f"{STRICT_COMMAND_POLICY}\nRequest: {user_text}"
    elif prompt_profile == "strict_fewshot":
        user_body = f"{STRICT_FEWSHOT_POLICY}\nRequest: {user_text}\nAnswer:"
    elif prompt_profile == "best_practical":
        raise ValueError("best_practical must be resolved per benchmark case")
    else:
        raise ValueError(f"Unknown prompt profile: {prompt_profile}")

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_body}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_prefill}"
    )


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
                f"{qairt_home}/lib/{target}:{qairt_home}/lib/aarch64-oe-linux-gcc8.2:"
                "/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu",
            ),
        }
    )
    env["PATH"] = (
        f"{qairt_home}/bin/{target}:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    return env


def extract_answer(raw_output: str) -> str:
    match = re.search(r"\[BEGIN\]:(.*?)\[END\]", raw_output, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_output.strip()


def split_reasoning(answer: str) -> dict[str, Any]:
    """Split Nemotron-style <think> content from the scoreable final answer."""
    text = answer.strip()
    lower = text.lower()
    open_tag = "<think>"
    close_tag = "</think>"
    open_idx = lower.find(open_tag)
    close_idx = lower.rfind(close_tag)

    if open_idx == -1:
        if close_idx == -1:
            return {
                "reasoning": "",
                "final_answer": text,
                "reasoning_open": False,
                "has_think_tags": False,
            }
        return {
            "reasoning": "",
            "final_answer": text[close_idx + len(close_tag) :].strip(),
            "reasoning_open": False,
            "has_think_tags": True,
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
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    decoder = json.JSONDecoder()
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def get_dotted(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def score_regex(case: dict[str, Any], answer: str) -> dict[str, Any]:
    normalized = answer.strip()
    passed_groups = 0
    missing: list[list[str]] = []
    for group in case.get("required_any", []):
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in group):
            passed_groups += 1
        else:
            missing.append(group)

    forbidden_hits = [
        pattern
        for pattern in case.get("forbidden", [])
        if re.search(pattern, normalized, flags=re.IGNORECASE)
    ]
    total = len(case.get("required_any", []))
    score = passed_groups / total if total else 1.0
    passed = score == 1.0 and not forbidden_hits
    if forbidden_hits:
        score = min(score, 0.5)
    return {
        "passed": passed,
        "score": round(score, 3),
        "missing": missing,
        "forbidden_hits": forbidden_hits,
    }


def score_json_tool(case: dict[str, Any], answer: str) -> dict[str, Any]:
    obj = extract_first_json(answer)
    checks = 0
    passed_checks = 0
    missing: list[str] = []
    if not isinstance(obj, dict):
        return {"passed": False, "score": 0, "missing": ["valid JSON object"]}

    checks += 1
    if obj.get("name") == case.get("expected_tool"):
        passed_checks += 1
    else:
        missing.append(f"name={case.get('expected_tool')}")

    for path, expected in case.get("expected_fields", {}).items():
        checks += 1
        actual = get_dotted(obj, path)
        if actual is None and path.startswith("parameters."):
            actual = get_dotted(obj, "arguments." + path.removeprefix("parameters."))
        if expected == "status_code":
            ok = isinstance(actual, str) and "status" in actual.lower() and "code" in actual.lower()
        else:
            ok = actual == expected
        if ok:
            passed_checks += 1
        else:
            missing.append(f"{path}={expected}")

    regex_score = score_regex(case, answer)
    checks += len(case.get("required_any", []))
    passed_checks += len(case.get("required_any", [])) - len(regex_score.get("missing", []))
    missing.extend(" / ".join(group) for group in regex_score.get("missing", []))

    score = passed_checks / checks if checks else 1.0
    return {"passed": score == 1.0, "score": round(score, 3), "missing": missing}


def score_boxed_number(case: dict[str, Any], answer: str) -> dict[str, Any]:
    match = re.search(r"\\boxed\{([^}]+)\}", answer)
    if not match:
        return {"passed": False, "score": 0, "missing": [r"\boxed{} final answer"]}
    nums = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
    if not nums:
        return {"passed": False, "score": 0.25, "missing": ["numeric boxed answer"]}
    value = float(nums[-1])
    expected = float(case["expected_number"])
    tolerance = float(case.get("tolerance", 0.001))
    passed = abs(value - expected) <= tolerance
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.5,
        "value": value,
        "expected": expected,
        "missing": [] if passed else [f"{expected} +/- {tolerance}"],
    }


def score_python_code(case: dict[str, Any], answer: str) -> dict[str, Any]:
    regex_score = score_regex(case, answer)
    has_fence = bool(re.search(r"```python\s+.*?```", answer, flags=re.DOTALL | re.IGNORECASE))
    score = regex_score["score"]
    if has_fence:
        score = min(1.0, score + 0.1)
    passed = regex_score["passed"] and has_fence
    missing = list(regex_score.get("missing", []))
    if not has_fence:
        missing.append(["python code fence"])
    return {
        "passed": passed,
        "score": round(score if passed else min(score, 0.9), 3),
        "missing": missing,
    }


def extract_python_code(answer: str) -> str:
    match = re.search(r"```python\s+(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    generic = re.search(r"```\s*(.*?)```", answer, flags=re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return answer.strip()


def score_python_unit_tests(case: dict[str, Any], answer: str) -> dict[str, Any]:
    code = extract_python_code(answer)
    regex_score = score_regex(case, answer)
    missing: list[Any] = list(regex_score.get("missing", []))
    if not code:
        return {"passed": False, "score": 0, "missing": ["python code"]}

    test_program = (
        code
        + "\n\n"
        + case.get("unit_tests", "")
        + "\nprint('UNIT_TESTS_PASSED')\n"
    )
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "candidate_test.py"
            test_path.write_text(test_program)
            proc = subprocess.run(
                [sys.executable, str(test_path)],
                cwd=temp_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "score": min(regex_score.get("score", 0), 0.5),
            "missing": missing + ["unit tests timed out"],
        }

    tests_passed = proc.returncode == 0 and "UNIT_TESTS_PASSED" in proc.stdout
    score = 1.0 if tests_passed else min(regex_score.get("score", 0), 0.6)
    if not tests_passed:
        missing.append((proc.stderr or proc.stdout or "unit tests failed")[-500:])
    return {
        "passed": tests_passed,
        "score": round(score, 3),
        "missing": missing,
    }


def score_constraint_checks(case: dict[str, Any], answer: str) -> dict[str, Any]:
    constraints = case.get("constraints", {})
    checks = 0
    passed_checks = 0
    missing: list[str] = []

    check_lines = [
        line.strip()
        for line in answer.splitlines()
        if re.match(r"^[-*]?\s*CHECK:", line.strip(), flags=re.IGNORECASE)
    ]
    if "exact_check_lines" in constraints:
        checks += 1
        expected = int(constraints["exact_check_lines"])
        if len(check_lines) == expected:
            passed_checks += 1
        else:
            missing.append(f"exactly {expected} CHECK lines")

    if "line_prefixes" in constraints:
        prefixes = list(constraints["line_prefixes"])
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        checks += 1
        if len(lines) == len(prefixes) and all(
            line.startswith(prefix) for line, prefix in zip(lines, prefixes)
        ):
            passed_checks += 1
        else:
            missing.append(f"lines starting with {prefixes}")

    for pattern in constraints.get("must_include", []):
        checks += 1
        if re.search(pattern, answer, flags=re.IGNORECASE):
            passed_checks += 1
        else:
            missing.append(f"include {pattern}")

    for pattern in constraints.get("must_not_include", []):
        checks += 1
        if not re.search(pattern, answer, flags=re.IGNORECASE):
            passed_checks += 1
        else:
            missing.append(f"omit {pattern}")

    for pattern, expected in constraints.get("exact_counts", {}).items():
        checks += 1
        count = len(re.findall(pattern, answer, flags=re.IGNORECASE))
        if count == expected:
            passed_checks += 1
        else:
            missing.append(f"{pattern} count {expected}")

    score = passed_checks / checks if checks else 1.0
    return {"passed": score == 1.0, "score": round(score, 3), "missing": missing}


def score_json_exact_keys(case: dict[str, Any], answer: str) -> dict[str, Any]:
    obj = extract_first_json(answer)
    if not isinstance(obj, dict):
        return {"passed": False, "score": 0, "missing": ["valid JSON object"]}
    expected = case.get("expected_keys", [])
    checks = 1 + len(case.get("required_any", []))
    passed_checks = 0
    missing: list[Any] = []
    if sorted(obj.keys()) == sorted(expected):
        passed_checks += 1
    else:
        missing.append(f"exact keys {expected}")
    regex_score = score_regex(case, json.dumps(obj))
    passed_checks += len(case.get("required_any", [])) - len(regex_score.get("missing", []))
    missing.extend(regex_score.get("missing", []))
    score = passed_checks / checks if checks else 1.0
    return {"passed": score == 1.0, "score": round(score, 3), "missing": missing}


def score_json_fields(case: dict[str, Any], answer: str) -> dict[str, Any]:
    obj = extract_first_json(answer)
    if not isinstance(obj, dict):
        return {"passed": False, "score": 0, "missing": ["valid JSON object"]}
    expected_fields = case.get("expected_fields", {})
    missing: list[str] = []
    passed_checks = 0
    for path, expected in expected_fields.items():
        actual = get_dotted(obj, path)
        if actual == expected:
            passed_checks += 1
        else:
            missing.append(f"{path}={expected!r}")
    checks = len(expected_fields)
    score = passed_checks / checks if checks else 1.0
    return {"passed": score == 1.0, "score": round(score, 3), "missing": missing}


def score_json_array_exact(case: dict[str, Any], answer: str) -> dict[str, Any]:
    obj = extract_first_json(answer)
    expected = case.get("expected_array", [])
    passed = obj == expected
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "missing": [] if passed else [f"exact array {expected!r}"],
    }


def score_boxed_choice(case: dict[str, Any], answer: str) -> dict[str, Any]:
    match = re.search(r"\\boxed\{([^}]+)\}", answer)
    if not match:
        return {"passed": False, "score": 0, "missing": [r"\boxed{} final answer"]}
    choice_match = re.search(r"\b([A-D])\b", match.group(1), flags=re.IGNORECASE)
    if not choice_match:
        return {"passed": False, "score": 0.25, "missing": ["single A-D choice"]}
    value = choice_match.group(1).upper()
    expected = str(case["expected_choice"]).upper()
    passed = value == expected
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.5,
        "value": value,
        "expected": expected,
        "missing": [] if passed else [expected],
    }


def load_profile(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}

    metrics: dict[str, Any] = {}
    for component in data.get("components", []):
        for event in component.get("events", []):
            if event.get("type") == "GenieDialog_query":
                for key, value in event.items():
                    if isinstance(value, dict) and "value" in value:
                        metrics[key] = value["value"]
                metrics["query_duration_us"] = event.get("duration")
            if event.get("type") == "GenieDialog_create":
                init_time = event.get("init-time", {})
                metrics["init_time_us"] = init_time.get("value", event.get("duration"))
    return metrics


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    scorer = case.get("scorer", "regex")
    if scorer == "regex":
        return score_regex(case, answer)
    if scorer == "json_tool":
        return score_json_tool(case, answer)
    if scorer == "boxed_number":
        return score_boxed_number(case, answer)
    if scorer == "python_code":
        return score_python_code(case, answer)
    if scorer == "python_unit_tests":
        return score_python_unit_tests(case, answer)
    if scorer == "constraint_checks":
        return score_constraint_checks(case, answer)
    if scorer == "json_exact_keys":
        return score_json_exact_keys(case, answer)
    if scorer == "json_fields":
        return score_json_fields(case, answer)
    if scorer == "json_array_exact":
        return score_json_array_exact(case, answer)
    if scorer == "boxed_choice":
        return score_boxed_choice(case, answer)
    raise ValueError(f"Unknown scorer: {scorer}")


def run_case(
    bundle: Path,
    result_dir: Path,
    model: str,
    mode: str,
    case: dict[str, Any],
    run_index: int,
    timeout_s: int,
    prompt_profile: str,
    score_full_answer: bool,
) -> dict[str, Any]:
    stem = f"{model}__{mode}__{case['id']}__run{run_index}"
    prompt_path = result_dir / f"{stem}.prompt.txt"
    profile_path = result_dir / f"{stem}.profile.json"
    log_path = result_dir / f"{stem}.log"

    if mode in {"thinking_on", "thinking_off"}:
        system = "detailed thinking on" if mode == "thinking_on" else "detailed thinking off"
    else:
        system = "You are a concise Linux and HTTP command assistant."

    effective_prompt_profile = prompt_profile
    if prompt_profile == "best_practical":
        effective_prompt_profile = "strict_v2" if case["category"] in {"linux", "http"} else "native"
    if prompt_profile == "best_practical_v2":
        effective_prompt_profile = "direct_final" if case["category"] in {"linux", "http"} else "native"
    if prompt_profile == "best_practical_v3":
        if case["category"] == "linux":
            effective_prompt_profile = "direct_final"
        elif case["category"] == "http":
            effective_prompt_profile = "strict_v2"
        else:
            effective_prompt_profile = "native"
    if prompt_profile == "best_practical_v4":
        effective_prompt_profile = "final_only"

    assistant_prefill = ""
    if mode == "thinking_off" and prompt_profile in {
        "native",
        "direct_final",
        "final_only",
        "native_final",
        "best_practical",
        "best_practical_v2",
        "best_practical_v3",
        "best_practical_v4",
        "strict_v2",
        "strict_fewshot",
    }:
        assistant_prefill = "<think>\n</think>\n"

    prompt_path.write_text(
        build_prompt(system, case["prompt"], effective_prompt_profile, assistant_prefill)
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
        cwd=bundle,
        env=qairt_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    elapsed_s = time.monotonic() - started
    log_path.write_text(proc.stdout)
    raw_answer = extract_answer(proc.stdout)
    reasoning_split = split_reasoning(raw_answer)
    answer = raw_answer if score_full_answer else reasoning_split["final_answer"]
    score = score_answer(case, answer)
    metrics = load_profile(profile_path)
    return {
        "model": model,
        "mode": mode,
        "case_id": case["id"],
        "category": case["category"],
        "run_index": run_index,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed_s, 3),
        "answer": answer,
        "raw_answer": raw_answer,
        "reasoning": reasoning_split["reasoning"],
        "reasoning_open": reasoning_split["reasoning_open"],
        "has_think_tags": reasoning_split["has_think_tags"],
        "score_basis": "raw_answer" if score_full_answer else "final_answer",
        "score": score,
        "profile": metrics,
        "prompt_profile": prompt_profile,
        "effective_prompt_profile": effective_prompt_profile,
        "paths": {
            "prompt": str(prompt_path),
            "log": str(log_path),
            "profile": str(profile_path),
        },
    }


def summarize(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Genie Practical Command Benchmark",
        "",
        "| model | mode | cases | pass | avg score | open think | avg final chars | avg decode tok/s | avg TTFT ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault((item["model"], item["mode"]), []).append(item)
    for (model, mode), items in sorted(grouped.items()):
        passed = sum(1 for item in items if item["score"]["passed"])
        avg_score = sum(item["score"]["score"] for item in items) / len(items)
        decode_rates = [
            item["profile"].get("token-generation-rate")
            for item in items
            if item["profile"].get("token-generation-rate") is not None
        ]
        ttfts = [
            item["profile"].get("time-to-first-token") / 1000
            for item in items
            if item["profile"].get("time-to-first-token") is not None
        ]
        avg_decode = sum(decode_rates) / len(decode_rates) if decode_rates else 0
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
        open_think = sum(1 for item in items if item.get("reasoning_open"))
        final_chars = [len(item.get("answer", "")) for item in items]
        avg_final_chars = sum(final_chars) / len(final_chars) if final_chars else 0
        lines.append(
            f"| {model} | {mode} | {len(items)} | {passed} | "
            f"{avg_score:.3f} | {open_think} | {avg_final_chars:.0f} | "
            f"{avg_decode:.2f} | {avg_ttft:.1f} |"
        )

    lines.extend(["", "## Case Results", ""])
    lines.append("| model | mode | case | category | score | pass |")
    lines.append("|---|---|---|---|---:|---|")
    for item in results:
        lines.append(
            f"| {item['model']} | {item['mode']} | {item['case_id']} | "
            f"{item['category']} | {item['score']['score']:.3f} | "
            f"{'yes' if item['score']['passed'] else 'no'} |"
        )
    lines.extend(["", "## Category Results", ""])
    lines.append("| model | mode | category | cases | pass | avg score |")
    lines.append("|---|---|---|---:|---:|---:|")
    category_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in results:
        category_grouped.setdefault(
            (item["model"], item["mode"], item["category"]), []
        ).append(item)
    for (model, mode, category), items in sorted(category_grouped.items()):
        passed = sum(1 for item in items if item["score"]["passed"])
        avg_score = sum(item["score"]["score"] for item in items) / len(items)
        lines.append(
            f"| {model} | {mode} | {category} | {len(items)} | {passed} | {avg_score:.3f} |"
        )
    return "\n".join(lines) + "\n"


def parse_csv_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode",
        choices=["stock", "thinking_off", "thinking_on", "both"],
        default="stock",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--out-root", type=Path, default=Path.home() / "genie_bench_results")
    parser.add_argument(
        "--prompt-profile",
        choices=[
            "native",
            "best_practical",
            "best_practical_v2",
            "best_practical_v3",
            "best_practical_v4",
            "direct_final",
            "final_only",
            "native_final",
            "legacy",
            "strict_v2",
            "strict_fewshot",
        ],
        default="legacy",
    )
    parser.add_argument("--suite", choices=sorted(SUITES), default="command")
    parser.add_argument("--case-ids", help="Comma-separated case ids to run.")
    parser.add_argument("--categories", help="Comma-separated categories to run.")
    parser.add_argument(
        "--score-full-answer",
        action="store_true",
        help="Score the full generated answer instead of text after </think>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = args.bundle.expanduser().resolve()
    if not (bundle / "genie_config.json").exists():
        print(f"Missing genie_config.json in {bundle}", file=sys.stderr)
        return 2
    if shutil.which("genie-t2t-run", path=qairt_env()["PATH"]) is None:
        print("genie-t2t-run not found in QAIRT PATH", file=sys.stderr)
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.out_root.expanduser() / f"{timestamp}__{args.model}__{args.mode}"
    result_dir.mkdir(parents=True, exist_ok=True)

    modes = ["thinking_off", "thinking_on"] if args.mode == "both" else [args.mode]
    results: list[dict[str, Any]] = []
    case_filter = parse_csv_filter(args.case_ids)
    category_filter = parse_csv_filter(args.categories)
    cases = [
        case
        for case in SUITES[args.suite]
        if (case_filter is None or case["id"] in case_filter)
        and (category_filter is None or case["category"] in category_filter)
    ]
    if not cases:
        print("No benchmark cases matched the selected filters", file=sys.stderr)
        return 2
    for mode in modes:
        for run_index in range(1, args.runs + 1):
            for case in cases:
                print(f"Running {args.model} {mode} {case['id']} run {run_index}", flush=True)
                try:
                    results.append(
                        run_case(
                            bundle,
                            result_dir,
                            args.model,
                            mode,
                            case,
                            run_index,
                            args.timeout_s,
                            args.prompt_profile,
                            args.score_full_answer,
                        )
                    )
                except subprocess.TimeoutExpired as exc:
                    results.append(
                        {
                            "model": args.model,
                            "mode": mode,
                            "case_id": case["id"],
                            "category": case["category"],
                            "run_index": run_index,
                            "returncode": None,
                            "elapsed_s": args.timeout_s,
                            "answer": "",
                            "score": {
                                "passed": False,
                                "score": 0,
                                "missing": ["timeout"],
                                "forbidden_hits": [],
                            },
                            "profile": {},
                            "error": f"timeout after {exc.timeout}s",
                        }
                    )

    (result_dir / "results.json").write_text(json.dumps(results, indent=2))
    (result_dir / "summary.md").write_text(summarize(results))
    print(f"RESULT_DIR={result_dir}")
    print((result_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
