"""AFINN-111 sentiment lexicon — vendored, sha256-pinned, with phrase + negation handling.

AFINN-111 (Finn Årup Nielsen) rates ~2,476 English words/phrases for valence from -5
(most negative) to +5 (most positive). We vendor the wordlist verbatim
(``concept_scorer/detectors/data/afinn_111.txt``, tab-separated ``word<TAB>valence``) and
verify it against a pinned sha256 at load time — mirroring how the prompt pool is frozen in
``prompts.py``.

Scoring follows AFINN's intended use rather than a naive word sum:

* **Multi-word phrases** in the list (e.g. ``not good`` -2, ``no fun`` -3, ``does not work``
  -3, ``cool stuff`` +3) are matched greedily, longest-first — so AFINN's own negated idioms
  score correctly instead of being split into their positive component words.
* **Negation** flips the valence of a sentiment word that follows a negator (``not``, ``no``,
  ``never``, ``n't`` …) within a short window, so ``not great`` / ``isn't wonderful`` score
  negative even though AFINN has no entry for them. A contrastive cue (``but``, ``however`` …)
  resets the window.

The net valence is the sum of the (possibly negated) matched valences. AFINN-111 is licensed
under the Open Database License (ODbL); see ``concept_scorer/detectors/data/AFINN_NOTICE.txt``.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "afinn_111.txt"

# Pinned digest of the vendored AFINN-111 wordlist (afinn/data/AFINN-111.txt @ fnielsen/afinn).
AFINN_111_SHA256 = "4703e14ed5ce7cb73591037cf21a202c0bdf8bdac06392808a1d5606ecf77a06"

# Words (and contractions, via the ``n't`` suffix) that negate the sentiment word(s) after them.
_NEGATORS = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "without", "cannot", "hardly", "scarcely", "barely", "rarely", "seldom",
})
# Contrastive cues that close an open negation window ("not good, but great").
_NEGATION_RESET = frozenset({"but", "however", "yet", "though", "although", "nevertheless"})
# A negator flips sentiment words up to this many tokens ahead (covers "not very good"
# without reaching across a clause into the next sentiment word).
_NEGATION_WINDOW = 2

# Word tokens, keeping internal contraction apostrophes (so "isn't"/"can't" stay intact).
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _is_negator(tok: str) -> bool:
    return tok in _NEGATORS or tok.endswith("n't")


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
        lexicon[word] = int(score)
    return lexicon


@lru_cache(maxsize=1)
def _lexicon() -> tuple[dict[str, int], int]:
    lex = _load_lexicon(_DATA_PATH, AFINN_111_SHA256)
    max_words = max((len(k.split()) for k in lex), default=1)
    return lex, max_words


def score_text(text: str) -> tuple[float, list[str]]:
    """Return ``(net_valence, contributing_keys)`` for ``text``.

    Greedy longest-match over the lexicon (phrases before words), with a negation window
    that flips the sign of sentiment terms following a negator.
    """
    lex, max_words = _lexicon()
    tokens = _TOKEN_RE.findall((text or "").lower())
    n = len(tokens)
    net = 0.0
    matched: list[str] = []
    neg_until = -1
    i = 0
    while i < n:
        val: int | None = None
        mlen = 1
        key = ""
        for length in range(min(max_words, n - i), 0, -1):
            cand = tokens[i] if length == 1 else " ".join(tokens[i : i + length])
            found = lex.get(cand)
            if found is not None:
                val, mlen, key = found, length, cand
                break
        if val is not None:
            net += -val if i <= neg_until else val
            matched.append(key)
            # A single-word negator that also carries valence (e.g. "no") still opens a window.
            if mlen == 1 and _is_negator(key):
                neg_until = i + _NEGATION_WINDOW
            i += mlen
            continue
        tok = tokens[i]
        if tok in _NEGATION_RESET:
            neg_until = -1
        elif _is_negator(tok):
            neg_until = i + _NEGATION_WINDOW
        i += 1
    return net, matched
