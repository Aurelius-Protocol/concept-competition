#!/usr/bin/env python3
"""Re-score an existing baseline pool against the current (pinned) detectors — no GPU.

When a detector's lexicon changes (e.g. a ``v2 -> v3`` bump), the per-concept scores baked
into ``data/baseline_pool.jsonl`` go stale, but the **completions** do not: they were
generated at ``alpha=0`` (unsteered) and are concept-independent. Regenerating them on a
non-canonical backend (MPS/CPU) could perturb the text; re-scoring the frozen completions
keeps the exact baseline corpus and only refreshes ``hit``/``score``/``matched``/
``contribution`` under the detectors pinned in the active config.

This reads the four detectors + scoring policy from the active config (point
``CONCEPT_SCORER_CONFIG`` at ``config/competition.yaml``), recomputes every record's
``concepts`` block, and rewrites the sidecar in place (atomically, via a temp file). The
``.meta.json`` provenance is refreshed so ``detectors``/``scoring`` reflect the new pins;
everything tied to the unchanged completions (model, generation, source pool sha256, count)
is preserved verbatim.

Run (no model load needed)::

    CONCEPT_SCORER_CONFIG=config/competition.yaml \
    .venv/bin/python scripts/rescore_baseline_pool.py --baseline data/baseline_pool.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

from concept_scorer.config import get_settings
from concept_scorer.detectors import get_detector


def _contribution(res, sc) -> float:
    """Per-completion contribution to the day-score, matching scorer._aggregate."""
    if sc.mode == "graded":
        score = res.score if res.score is not None else 0.0
        return min(max(score / sc.saturation, 0.0), 1.0)
    return 1.0 if res.hit else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=os.path.join("data", "baseline_pool.jsonl"))
    args = ap.parse_args()

    settings = get_settings()
    concepts = list(settings.concepts.active_allowed)
    detectors = {
        c: (get_detector(c, settings.detectors, threshold=settings.scoring[c].threshold),
            settings.scoring[c])
        for c in concepts
    }

    meta_path = os.path.splitext(args.baseline)[0] + ".meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    tmp_path = args.baseline + ".tmp"
    n = 0
    changed = {c: 0 for c in concepts}
    with open(args.baseline) as src, open(tmp_path, "w") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completion = rec["completion"]
            per_concept = {}
            for concept, (det, sc) in detectors.items():
                res = det.detect(completion)
                new_block = {
                    "hit": bool(res.hit),
                    "score": float(res.score) if res.score is not None else 0.0,
                    "matched": list(res.matched),
                    "contribution": _contribution(res, sc),
                }
                old = rec.get("concepts", {}).get(concept)
                if old is None or old.get("contribution") != new_block["contribution"]:
                    changed[concept] += 1
                per_concept[concept] = new_block
            rec["concepts"] = per_concept
            out.write(json.dumps(rec) + "\n")
            n += 1

    if n != meta.get("count"):
        os.remove(tmp_path)
        raise SystemExit(f"re-scored {n} records but meta count is {meta.get('count')} — aborting")

    os.replace(tmp_path, args.baseline)

    # Refresh only the detector/scoring provenance; the completions (and thus model,
    # generation, source pool) are unchanged.
    meta["detectors"] = dict(settings.detectors)
    meta["scoring"] = {
        c: {"mode": sc.mode, "threshold": sc.threshold, "saturation": sc.saturation}
        for c, sc in settings.scoring.items()
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"re-scored {n} records -> {args.baseline}", flush=True)
    print("contribution changes per concept: "
          + ", ".join(f"{c}={changed[c]}" for c in concepts), flush=True)
    print(f"detectors now: {meta['detectors']}", flush=True)


if __name__ == "__main__":
    main()
