"""Create deterministic configs for the shipping-agent demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IDENTITY_PROMPT = {
    "prompt_system": "",
    "prompt_user": "string",
    "prompt_assistant": "string",
    "prompt_tool": "string",
    "prompt_start": "",
    "context_size": 4096,
}


def prepare_cpp_service_layout(bundle: Path, config: dict) -> list[Path]:
    """Expose nested QAIRT artifacts where GenieAPIService expects basenames."""
    dialog = config["dialog"]
    references = [
        dialog["tokenizer"]["path"],
        dialog["engine"]["backend"]["extensions"],
        *dialog["engine"]["model"]["binary"]["ctx-bins"],
    ]
    links: list[Path] = []
    for reference in references:
        source = bundle / reference
        if not source.exists():
            raise FileNotFoundError(f"Missing referenced artifact {source}")
        link = bundle / source.name
        if link == source:
            continue
        if link.exists() or link.is_symlink():
            if link.resolve() != source.resolve():
                raise FileExistsError(f"Cannot replace unrelated path {link}")
        else:
            link.symlink_to(source.relative_to(bundle))
        links.append(link)
    return links


def prepare_bundle(bundle: Path, output: str) -> tuple[Path, Path]:
    source = bundle / "genie_config.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}")
    config = json.loads(source.read_text())
    sampler = config["dialog"]["sampler"]
    sampler.update({"seed": 42, "temp": 0.0, "top-k": 1, "top-p": 1.0})

    destination = bundle / output
    destination.write_text(json.dumps(config, indent=2) + "\n")

    prompt_destination = bundle / "prompt.json"
    prompt = dict(IDENTITY_PROMPT)
    prompt["context_size"] = config["dialog"]["context"]["size"]
    prompt_destination.write_text(json.dumps(prompt, indent=2) + "\n")
    prepare_cpp_service_layout(bundle, config)
    return destination, prompt_destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Extracted Genie bundle directory")
    parser.add_argument("--output", default="genie_config.agent.json")
    args = parser.parse_args()

    bundle = args.bundle.expanduser().resolve()
    try:
        destination, prompt_destination = prepare_bundle(bundle, args.output)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(destination)
    print(prompt_destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
