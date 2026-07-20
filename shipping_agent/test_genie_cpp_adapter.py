"""Tests for the persistent C++ Genie protocol adapter."""

import json
import tempfile
import unittest
from pathlib import Path

from shipping_agent.genie_cpp_adapter import (
    adapt_upstream_response,
    build_upstream_payload,
)
from shipping_agent.prepare_bundle import IDENTITY_PROMPT, prepare_bundle


class GenieCppAdapterTests(unittest.TestCase):
    def test_upstream_receives_one_native_prompt_and_no_tools(self) -> None:
        payload = {
            "model": "public-model",
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Read the queue."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_queue",
                        "description": "Return the queue.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        upstream, prompt = build_upstream_payload(payload, "bundle-name")

        self.assertEqual(upstream["model"], "bundle-name")
        self.assertNotIn("tools", upstream)
        self.assertEqual(upstream["messages"], [{"role": "user", "content": prompt}])
        self.assertIn("[SYSTEM_PROMPT]Use tools.[/SYSTEM_PROMPT]", prompt)
        self.assertIn("[AVAILABLE_TOOLS]", prompt)
        self.assertIn("[INST]Read the queue.[/INST]", prompt)

    def test_empty_assistant_content_does_not_render_as_null(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "Read the queue."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_queue",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "{\"ok\":true}",
                },
            ]
        }
        _, prompt = build_upstream_payload(payload, "bundle-name")

        self.assertNotIn("null[TOOL_CALLS]", prompt)
        self.assertIn("[TOOL_CALLS]get_queue[ARGS]{}", prompt)


    def test_native_tool_call_becomes_openai_tool_call(self) -> None:
        upstream = {
            "id": "upstream-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '[TOOL_CALLS]schedule_shipment[ARGS]{"carrier_id":"C1","dock_id":"D1"}',
                    }
                }
            ],
        }
        result, status = adapt_upstream_response(upstream, "public-model")

        choice = result["choices"][0]
        self.assertEqual(status, "tool_mistral_native")
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "schedule_shipment")
        self.assertEqual(
            json.loads(call["function"]["arguments"]),
            {"carrier_id": "C1", "dock_id": "D1"},
        )

    def test_plain_text_remains_a_final_answer(self) -> None:
        upstream = {
            "choices": [{"message": {"role": "assistant", "content": "Done."}}]
        }
        result, status = adapt_upstream_response(upstream, "public-model")

        self.assertEqual(status, "final_mistral_native")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["choices"][0]["message"]["content"], "Done.")

    def test_bundle_preparation_writes_identity_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            artifacts = bundle / "artifacts"
            artifacts.mkdir()
            for name in ("tokenizer.json", "backend.json", "model.bin"):
                (artifacts / name).write_text(name)
            config = {
                "dialog": {
                    "sampler": {},
                    "context": {"size": 2048},
                    "tokenizer": {"path": "artifacts/tokenizer.json"},
                    "engine": {
                        "backend": {"extensions": "artifacts/backend.json"},
                        "model": {
                            "binary": {"ctx-bins": ["artifacts/model.bin"]}
                        },
                    },
                }
            }
            (bundle / "genie_config.json").write_text(json.dumps(config))

            config_path, prompt_path = prepare_bundle(bundle, "agent.json")

            prepared = json.loads(config_path.read_text())
            self.assertEqual(prepared["dialog"]["sampler"]["top-k"], 1)
            prompt = json.loads(prompt_path.read_text())
            self.assertEqual(prompt["context_size"], 2048)
            self.assertEqual(prompt["prompt_user"], IDENTITY_PROMPT["prompt_user"])
            for name in ("tokenizer.json", "backend.json", "model.bin"):
                link = bundle / name
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(), (artifacts / name).resolve())


if __name__ == "__main__":
    unittest.main()
