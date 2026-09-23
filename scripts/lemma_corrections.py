"""Post-Stanza lemma corrections (Decorator around any lemmatizer).

Why this layer exists:
  Stanza's Russian lemmatizer is a seq2seq model. On forms it has rarely seen
  (ё-initial words, imperatives, genitive plurals, neologisms) it sometimes
  produces non-words ("растекться", "жиль", "гед" for «Лёд") or inconsistent
  ё spelling ("теща" vs "тёща"), which splits one word into several rows.
  Editing extracted_vocabulary_stanza.csv by hand is pointless because CI
  regenerates it on every push, so corrections are applied here, to the token
  stream, before aggregation. Both extract_vocabulary_stanza.py and
  classify_vocabulary.py use this layer so their (lemma, upos) keys agree.

Design:
  - normalize_lexical(): mechanical rules only (no dictionary), so it is
    predictable: drop tokens without Cyrillic letters or with digits (years,
    Roman numerals, Latin brand names, initials), strip a leading hyphen left
    by compound splitting ("-запад" from «юго-западе»), drop dangling
    prefixes ending in a hyphen ("трек-" from «трек-номер»).
  - LemmaCorrector: table-driven fixes from data/lemma_corrections.csv,
    reviewed by a human. Key = (stanza_lemma, stanza_upos, wordform); the
    wordform column is optional and, when filled, beats the generic rule.
    Why a wordform option: one wrong lemma can come from forms that need
    different fixes ("свертывать" from «сворачивается» and «сворачивать»).
  - CorrectedLemmatizer: wraps an object with analyze(text) -> list[dict]
    (Decorator), so the Stanza wrapper itself stays unchanged (SRP).

Known limits:
  - The table was built from the aggregated CSV (one example wordform per
    row), not from every token, because Stanza cannot run in the sandbox
    where the audit was done. Generic (lemma, upos) rules cover all tokens
    that produced that wrong lemma; new passages may surface new errors, so
    re-run scripts/audit_lemmas_pymorphy.py after adding passages.
  - The table fixes lemma/UPOS only; case features are left as Stanza gave.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TABLE = Path(__file__).resolve().parents[1] / "data" / "lemma_corrections.csv"

_CYRILLIC = re.compile(r"[а-яё]")
# Letters (Cyrillic, plus Latin only as part of a Cyrillic compound such as
# "qr-код") joined by single hyphens. Digits, dots, apostrophes are excluded.
_LEXICAL = re.compile(r"^[а-яёa-z]+(?:-[а-яёa-z]+)*$")
_VERBAL_UPOS = {"VERB", "AUX"}
_ACTIONS = {"replace", "drop"}
_COLUMNS = ["stanza_lemma", "stanza_upos", "wordform", "action", "lemma", "upos", "note"]


def normalize_lexical(lemma: str) -> str | None:
    """Return the cleaned lemma, or None if it is not vocabulary."""
    lemma = lemma.strip().lower()
    if lemma.endswith("-"):
        return None
    lemma = lemma.lstrip("-–")
    if not _CYRILLIC.search(lemma) or not _LEXICAL.match(lemma):
        return None
    return lemma


@dataclass(frozen=True)
class Rule:
    action: str
    lemma: str
    upos: str
    note: str = ""


RuleKey = tuple[str, str, str]  # (stanza_lemma, stanza_upos, wordform or "")


def load_corrections(path: Path = DEFAULT_TABLE) -> dict[RuleKey, Rule]:
    """Load and validate the correction table (fails loudly on bad rows)."""
    rules: dict[RuleKey, Rule] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != _COLUMNS:
            raise ValueError(f"unexpected columns in {path}: {reader.fieldnames!r}")
        for n, row in enumerate(reader, start=2):
            key = (row["stanza_lemma"], row["stanza_upos"], row["wordform"].lower())
            action = row["action"]
            if action not in _ACTIONS:
                raise ValueError(f"{path}:{n}: invalid action {action!r}")
            if action == "replace" and not (row["lemma"] and row["upos"]):
                raise ValueError(f"{path}:{n}: replace needs lemma and upos")
            if key in rules:
                raise ValueError(f"{path}:{n}: duplicate key {key!r}")
            rules[key] = Rule(action, row["lemma"], row["upos"], row["note"])
    return rules


class LemmaCorrector:
    def __init__(self, rules: dict[RuleKey, Rule]):
        self._rules = rules

    def correct(self, word: dict) -> dict | None:
        """Return a corrected copy of a token dict, or None to drop it."""
        lemma, upos = word["lemma"].lower(), word["upos"]
        rule = (self._rules.get((lemma, upos, word["wordform"].lower()))
                or self._rules.get((lemma, upos, "")))
        out = dict(word)
        if rule is not None:
            if rule.action == "drop":
                return None
            lemma, upos = rule.lemma, rule.upos
            out["upos"] = upos
            if upos not in _VERBAL_UPOS:
                # Why: VerbForm (Fin/Part/Conv) is meaningless once the token
                # is re-tagged as ADJ/NOUN/ADV and would split rows.
                out["verb_form"] = ""
        cleaned = normalize_lexical(lemma)
        if cleaned is None:
            return None
        out["lemma"] = cleaned
        return out


class CorrectedLemmatizer:
    """Decorator: same analyze() interface, corrected output."""

    def __init__(self, inner, corrector: LemmaCorrector | None = None):
        self._inner = inner
        self._corrector = corrector or LemmaCorrector(load_corrections())

    def analyze(self, text: str) -> list[dict]:
        result = []
        for w in self._inner.analyze(text):
            fixed = self._corrector.correct(w)
            if fixed is not None:
                result.append(fixed)
        return result
