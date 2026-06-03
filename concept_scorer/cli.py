"""Command-line interface mirroring the HTTP API.

Subcommands:
  score     one-shot scoring of a submission (loads the warm model for the run)
  validate  load+validate a submission only (no GPU); non-zero exit on reject
  smoke     weather reference smoke test on the real model (GPU)
  info      print the pinned config payload
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .backends import SteeringUnsupported
from .config import get_settings
from .errors import SubmissionError
from .scorer import score_submission
from .submission import load_submission
from .version import MODULE_SCHEMA_VERSION, __version__


def _cmd_validate(args) -> int:
    settings = get_settings()
    try:
        sub = load_submission(args.submission, settings, args.concept)
    except SubmissionError as e:
        print(json.dumps(e.to_dict()))
        return 2
    print(json.dumps({"error_code": "ok", "alpha": sub.alpha, "layer": sub.layer,
                      "concept": sub.concept}))
    return 0


def _load_runtime_and_pool(settings, baseline: bool = False):
    from .backends import build_backend
    from .prompts import load_pool

    pool = load_pool(settings)
    rt = build_backend(settings)
    if baseline and hasattr(rt, "allow_unsteered"):
        rt.allow_unsteered = True
    return rt, pool


def _cmd_score(args) -> int:
    settings = get_settings()
    try:
        sub = load_submission(args.submission, settings, args.concept)
    except SubmissionError as e:
        if args.reject_as_zero:
            print(json.dumps({"score": 0.0, **e.to_dict()}))
            return 0
        print(json.dumps(e.to_dict()), file=sys.stderr)
        return 2

    rt, pool = _load_runtime_and_pool(settings, baseline=getattr(args, "baseline", False))
    try:
        result = score_submission(
            rt, settings, sub, args.concept, args.day_index, args.seed, pool,
            return_completions=not args.no_completions,
        )
    except SteeringUnsupported as e:
        print(json.dumps({"error_code": "steering_unsupported", "message": str(e)}), file=sys.stderr)
        return 3
    payload = {
        "score": result.score,
        "hit_count": result.hit_count,
        "total": result.total,
        "diagnostics": result.diagnostics,
    }
    if not args.no_completions:
        payload["completions"] = [dataclasses.asdict(r) for r in result.per_completion]
    print(json.dumps(payload, indent=None if args.json else 2))
    return 0


def _cmd_smoke(args) -> int:
    import os

    from .weather import WeatherDetector  # noqa: F401  (registers concept semantics)

    settings = get_settings()
    ref = args.reference or os.path.join(
        os.path.dirname(__file__), "weather", "reference_direction.safetensors"
    )
    sub = load_submission(ref, settings, "weather")  # concept check: metadata must say weather
    rt, pool = _load_runtime_and_pool(settings, baseline=getattr(args, "baseline", False))

    prompts = pool.sample_day(args.day_index, args.seed, settings.prompts.per_day)
    instructions = [p.instruction for p in prompts]
    try:
        steered = rt.generate(instructions, sub)
    except SteeringUnsupported as e:
        print(json.dumps({"error_code": "steering_unsupported", "message": str(e)}), file=sys.stderr)
        return 3

    det = WeatherDetector()
    hits = sum(1 for c in steered if det.detect(c).hit)
    hit_rate = hits / len(steered) if steered else 0.0
    print(json.dumps({"weather_hit_rate": hit_rate, "hits": hits, "total": len(steered)}))
    ok = hit_rate >= args.floor
    if not ok:
        print(f"SMOKE FAIL: weather hit_rate {hit_rate:.3f} < floor {args.floor}", file=sys.stderr)
    return 0 if ok else 1


def _cmd_info(args) -> int:
    s = get_settings()
    print(json.dumps({
        "repo_id": s.model.repo_id,
        "model_revision": s.model.revision,
        "hidden_size": s.model.hidden_size,
        "steer_layer": s.model.steer_layer,
        "allowed_concepts": list(s.concepts.active_allowed),
        "detector_versions": dict(s.detectors),
        "scoring": {k: dataclasses.asdict(v) for k, v in s.scoring.items()},
        "module_version": __version__,
        "schema_version": MODULE_SCHEMA_VERSION,
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="concept-scorer")
    sub = p.add_subparsers(dest="command", required=True)
    concepts = list(get_settings().concepts.active_allowed)

    sp = sub.add_parser("score", help="score a submission (GPU)")
    sp.add_argument("--submission", required=True)
    sp.add_argument("--concept", required=True, choices=concepts)
    sp.add_argument("--day-index", type=int, required=True)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--no-completions", action="store_true")
    sp.add_argument("--reject-as-zero", action="store_true")
    sp.add_argument("--json", action="store_true", help="compact single-line JSON")
    sp.add_argument("--baseline", action="store_true",
                    help="allow an UNSTEERED run on a black-box backend (not a valid score)")
    sp.set_defaults(func=_cmd_score)

    vp = sub.add_parser("validate", help="validate a submission (no GPU)")
    vp.add_argument("--submission", required=True)
    vp.add_argument("--concept", required=True, choices=concepts)
    vp.set_defaults(func=_cmd_validate)

    smp = sub.add_parser("smoke", help="weather reference smoke test (GPU)")
    smp.add_argument("--reference", default=None)
    smp.add_argument("--day-index", type=int, default=0)
    smp.add_argument("--seed", type=int, default=1234)
    smp.add_argument("--floor", type=float, default=0.15,
                     help="weather hit-rate PASS threshold (default 0.15, CUDA-NF4-calibrated; SPEC §12)")
    smp.add_argument("--baseline", action="store_true",
                     help="allow an UNSTEERED run on a black-box backend (not a valid score)")
    smp.set_defaults(func=_cmd_smoke)

    ip = sub.add_parser("info", help="print pinned config")
    ip.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
