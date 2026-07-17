#!/usr/bin/env python3
"""Compile a Ministral GGUF for the IQ9075 HTP and export a Genie bundle."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--container-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--target", default="dsp_arch:v73;soc_model:77;cores:1")
    parser.add_argument("--num-splits", type=int, default=9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gguf = args.gguf.expanduser().resolve()
    if not gguf.is_file():
        raise FileNotFoundError(gguf)

    for path in (args.cache_root, args.container_dir.parent, args.export_dir.parent):
        path.expanduser().mkdir(parents=True, exist_ok=True)
    if args.container_dir.exists():
        raise FileExistsError(f"Container destination already exists: {args.container_dir}")
    if args.export_dir.exists():
        raise FileExistsError(f"Export destination already exists: {args.export_dir}")

    started = time.monotonic()
    print(f"CREATE_BUILDER gguf={gguf}", flush=True)
    builder = GenAIBuilderFactory.create(
        str(gguf),
        "HTP",
        cache_root=str(args.cache_root.expanduser().resolve()),
    )
    builder.set_targets([args.target])
    builder.set_transformation_options(options={"split.num_splits": args.num_splits})

    build_started = time.monotonic()
    print(f"BUILD_START target={args.target}", flush=True)
    container = builder.build()
    build_elapsed = time.monotonic() - build_started
    print(f"BUILD_DONE elapsed_s={build_elapsed:.3f}", flush=True)

    save_started = time.monotonic()
    container.save(args.container_dir.expanduser().resolve())
    save_elapsed = time.monotonic() - save_started
    print(f"SAVE_DONE elapsed_s={save_elapsed:.3f}", flush=True)

    export_started = time.monotonic()
    container.export(args.export_dir.expanduser().resolve())
    export_elapsed = time.monotonic() - export_started
    print(f"EXPORT_DONE elapsed_s={export_elapsed:.3f}", flush=True)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {
        "gguf": str(gguf),
        "target": args.target,
        "num_splits": args.num_splits,
        "build_elapsed_s": round(build_elapsed, 3),
        "save_elapsed_s": round(save_elapsed, 3),
        "export_elapsed_s": round(export_elapsed, 3),
        "total_elapsed_s": round(time.monotonic() - started, 3),
        "max_rss_kb": usage.ru_maxrss,
        "container_dir": str(args.container_dir.expanduser().resolve()),
        "export_dir": str(args.export_dir.expanduser().resolve()),
    }
    print("EXPORT_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
