"""AFINN-111 sentiment lexicon — vendored, sha256-pinned, word-level scorer.

AFINN-111 (Finn Årup Nielsen) rates ~2,476 English words/phrases for valence from -5
(most negative) to +5 (most positive). We vendor the wordlist verbatim
(``data/afinn_111.txt``, tab-separated ``word<TAB>valence``) and verify it against a pinned
sha256 at load time — mirroring how the prompt pool is frozen in ``prompts.py``. A mismatch
means the lexicon moved underfoot (which would silently shift every positive-sentiment
score), so we fail fast.

Scoring is word-level: the text is lowercased, split into alphabetic tokens, and the net
valence is the sum of the valences of the tokens present in the lexicon. The 15 multi-word
AFINN entries (e.g. "cool stuff") are intentionally not matched by the word-level tokenizer
— a documented simplification that keeps scoring deterministic and dependency-free.

AFINN-111 is licensed under the Open Database License (ODbL) v1.0; see
``data/AFINN_NOTICE.txt`` for attribution and citation.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "afinn_111.txt"

# Pinned digest of the vendored AFINN-111 wordlist (afinn/data/AFINN-111.txt @ fnielsen/afinn).
AFINN_111_SHA256 = "4703e14ed5ce7cb73591037cf21a202c0bdf8bdac06392808a1d5606ecf77a06"

_TOKEN_RE = re.compile(r"[a-z]+")


def _load_lexicon(path: Path, expected_sha256: str) -> dict[str, int]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"AFINN lexicon sha256 mismatch at {path}: got {digest}, expected {expected_sha256}"
        )
    lexicon: dict[str, int] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        word, _, score = line.partition("\t")
        if " " in word:  # word-level scorer skips the few multi-word phrases
            continue
        lexicon[word] = int(score)
    return lexicon


@lru_cache(maxsize=1)
def _lexicon() -> dict[str, int]:
    return _load_lexicon(_DATA_PATH, AFINN_111_SHA256)


def score_text(text: str) -> tuple[float, list[str]]:
    """Return ``(net_valence, contributing_words)`` for ``text`` by summing token valences."""
    lex = _lexicon()
    net = 0
    matched: list[str] = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        val = lex.get(tok)
        if val is not None:
            net += val
            matched.append(tok)
    return float(net), matched
