#!/usr/bin/env python3
"""Build-time: freeze a held-out alpaca-cleaned prompt pool into the image.

Pulls ``unsloth/alpaca-cleaned`` at a pinned revision, keeps pure instructions (empty
``input`` field) within a length band, dedupes, deterministically shuffles with a fixed
build seed, takes ``pool_size`` items, and writes ``data/prompt_pool.jsonl`` plus a
``data/prompt_pool.sha256`` integrity file. This runs once during ``docker build``;
``datasets`` is therefore a build-only dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

# Fixed build seed — changing this re-freezes a different (still deterministic) pool.
BUILD_SEED = 20260601
MIN_LEN = 16
MAX_LEN = 600


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unsloth/alpaca-cleaned")
    ap.add_argument("--revision", default=None, help="pinned dataset commit SHA")
    ap.add_argument("--pool-size", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join("data", "prompt_pool.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train", revision=args.revision)

    seen: set[str] = set()
    candidates: list[str] = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        has_input = bool((row.get("input") or "").strip())
        if has_input:
            continue
        if not (MIN_LEN <= len(instruction) <= MAX_LEN):
            continue
        key = instruction.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(instruction)

    random.Random(BUILD_SEED).shuffle(candidates)
    pool = candidates[: args.pool_size]
    if len(pool) < args.pool_size:
        raise SystemExit(
            f"only {len(pool)} candidates after filtering; need {args.pool_size}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = [json.dumps({"id": i, "instruction": text}) for i, text in enumerate(pool)]
    blob = ("\n".join(lines) + "\n").encode("utf-8")
    with open(args.out, "wb") as f:
        f.write(blob)
    sha = hashlib.sha256(blob).hexdigest()
    with open(args.out.replace(".jsonl", ".sha256"), "w") as f:
        f.write(sha + "\n")
    print(f"wrote {len(pool)} prompts to {args.out} (sha256={sha})")


if __name__ == "__main__":
    main()
