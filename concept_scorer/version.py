"""Module version constants for the concept-scorer."""

__version__ = "0.1.0"

# Bumped when the request/response wire schema changes in a breaking way.
# "2": per-completion `score` + response `scoring_mode`; `score` is now mode-dependent
# (hit_rate fraction or graded mean intensity).
# "3": minimal-intervention reward — `/score` response carries `raw_score`/`push`/`push_scale`/
# `efficiency`, the request accepts an optional `push_scale`, and `score` may be discounted by
# exp(-push/push_scale).
MODULE_SCHEMA_VERSION = "3"
