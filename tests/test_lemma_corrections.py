"""Tests for the post-Stanza lemma correction layer.

Why these tests exist: the correction table is hand-maintained and applied
automatically in CI, so a malformed row (chained replacement, duplicate key,
typo in action) would silently corrupt every regenerated vocabulary CSV.
The tests run without Stanza (a fake lemmatizer stands in), because Stanza's
model cannot be downloaded in every environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lemma_corrections import (  # noqa: E402
    CorrectedLemmatizer,
    LemmaCorrector,
    load_corrections,
    normalize_lexical,
)

TABLE = ROOT / "data" / "lemma_corrections.csv"
HEADER = "stanza_lemma,stanza_upos,wordform,action,lemma,upos,note\n"


def word(lemma, upos, wordform, verb_form="", case=""):
    return {"lemma": lemma, "upos": upos, "verb_form": verb_form,
            "case": case, "wordform": wordform}


def write_table(tmp_path, rows):
    p = tmp_path / "t.csv"
    p.write_text(HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return p


# --- mechanical lexical rules -------------------------------------------

@pytest.mark.parametrize("lemma", ["2025", "1,7", "05.03", "xix", "chatgpt", "м.", "b12", "ботвинник2"])
def test_non_lexical_tokens_are_dropped(lemma):
    assert normalize_lexical(lemma) is None


@pytest.mark.parametrize("lemma", ["трек-", "мкб-", "тут-", "арбатско-"])
def test_trailing_hyphen_fragments_are_dropped(lemma):
    assert normalize_lexical(lemma) is None


def test_leading_hyphen_is_stripped():
    assert normalize_lexical("-запад") == "запад"


@pytest.mark.parametrize("lemma", ["дом", "северо-запад", "qr-код", "ёлка"])
def test_ordinary_words_pass_unchanged(lemma):
    assert normalize_lexical(lemma) == lemma


# --- table-driven corrections -------------------------------------------

def test_generic_replacement(tmp_path):
    c = LemmaCorrector(load_corrections(write_table(tmp_path, ["жиль,NOUN,,replace,жильё,NOUN,"])))
    assert c.correct(word("жиль", "NOUN", "жильё"))["lemma"] == "жильё"


def test_wordform_specific_rule_wins_over_generic(tmp_path):
    c = LemmaCorrector(load_corrections(write_table(tmp_path, [
        "свертывать,VERB,,replace,сворачивать,VERB,",
        "свертывать,VERB,сворачивается,replace,сворачиваться,VERB,",
    ])))
    assert c.correct(word("свертывать", "VERB", "Сворачивается"))["lemma"] == "сворачиваться"
    assert c.correct(word("свертывать", "VERB", "сворачивать"))["lemma"] == "сворачивать"


def test_drop_action_removes_token(tmp_path):
    c = LemmaCorrector(load_corrections(write_table(tmp_path, ["нон,NOUN,,drop,,,"])))
    assert c.correct(word("нон", "NOUN", "нон")) is None


def test_verb_form_cleared_when_upos_leaves_verb(tmp_path):
    c = LemmaCorrector(load_corrections(write_table(tmp_path, ["кисловатый,VERB,,replace,кисловатый,ADJ,"])))
    out = c.correct(word("кисловатый", "VERB", "кисловат", verb_form="Fin"))
    assert (out["upos"], out["verb_form"]) == ("ADJ", "")


def test_verb_form_kept_for_verb_targets(tmp_path):
    c = LemmaCorrector(load_corrections(write_table(tmp_path, ["вымышить,VERB,,replace,вымыслить,VERB,"])))
    assert c.correct(word("вымышить", "VERB", "вымышлены", verb_form="Part"))["verb_form"] == "Part"


def test_unlisted_word_passes_through_lexical_rules_only(tmp_path):
    c = LemmaCorrector(load_corrections(write_table(tmp_path, [])))
    assert c.correct(word("дом", "NOUN", "дома"))["lemma"] == "дом"
    assert c.correct(word("2025", "ADJ", "2025")) is None


def test_invalid_action_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_corrections(write_table(tmp_path, ["жиль,NOUN,,fix,жильё,NOUN,"]))


def test_duplicate_key_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_corrections(write_table(tmp_path, [
            "жиль,NOUN,,replace,жильё,NOUN,",
            "жиль,NOUN,,replace,жилье,NOUN,",
        ]))


def test_replace_requires_target(tmp_path):
    with pytest.raises(ValueError):
        load_corrections(write_table(tmp_path, ["жиль,NOUN,,replace,,NOUN,"]))


# --- decorator ---------------------------------------------------------

class FakeLemmatizer:
    def __init__(self, words):
        self._words = words

    def analyze(self, text):
        return list(self._words)


def test_corrected_lemmatizer_filters_and_replaces(tmp_path):
    c = LemmaCorrector(load_corrections(write_table(tmp_path, ["жиль,NOUN,,replace,жильё,NOUN,"])))
    lem = CorrectedLemmatizer(FakeLemmatizer([
        word("жиль", "NOUN", "жильё"), word("2025", "ADJ", "2025"), word("дом", "NOUN", "дом"),
    ]), c)
    assert [w["lemma"] for w in lem.analyze("x")] == ["жильё", "дом"]


# --- integrity of the real table ---------------------------------------

def test_repository_table_loads_and_is_consistent():
    rules = load_corrections(TABLE)
    assert rules, "correction table should not be empty"
    sources = {(k[0], k[1]) for k in rules}
    for (src_lemma, src_upos, _wf), rule in rules.items():
        if rule.action == "drop":
            continue
        # Why: targets must survive the lexical filter, otherwise a
        # replacement would be silently dropped afterwards.
        assert normalize_lexical(rule.lemma) == rule.lemma, rule
        # Why: no chains (A->B, B->C); a single lookup must be final.
        assert (rule.lemma, rule.upos) not in sources or (rule.lemma, rule.upos) == (src_lemma, src_upos), rule
