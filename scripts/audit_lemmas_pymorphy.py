#!/usr/bin/env python3
"""Flag suspicious Stanza lemmas for human review (maintenance tool).

Why: data/lemma_corrections.csv is reviewed by hand. After new passages are
added, run this to list candidate errors, decide each one, and add rows to
the table. The OpenCorpora dictionary (pymorphy3) is used only as an
independent second opinion, NOT as ground truth: many flagged "unknown"
words are legitimate neologisms (нейросеть, коворкинг), and pymorphy prefers
some forms UD does not (e.g. «лучший» -> «хороший»). Hence this tool never
edits the table itself.

Not run in CI (needs pymorphy3 + pymorphy3-dicts-ru, which the pipeline
does not otherwise depend on).

Flags:
  unknown          lemma not in the dictionary (typo-like model output?)
  not_normal_form  lemma is a word form, not a dictionary form (жиль, засух)
  yo               lemma lacks ё that the dictionary form has (теща)
  pos_mismatch     dictionary never assigns a compatible POS (чуть/NOUN)
Rows already covered by the correction table are skipped.

Usage: python scripts/audit_lemmas_pymorphy.py [--extracted CSV] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pymorphy3

from lemma_corrections import load_corrections, normalize_lexical

UPOS_TO_OC = {
    "NOUN": {"NOUN"}, "PROPN": {"NOUN"}, "VERB": {"INFN", "VERB"},
    "ADJ": {"ADJF", "ADJS", "PRTF", "PRTS", "COMP", "NUMR", "NPRO"},
    "ADV": {"ADVB", "PRED", "COMP", "GRND", "CONJ", "PRCL", "INTJ"},
}
SKIP_UPOS = {"ADP", "CCONJ", "SCONJ", "PART", "AUX", "INTJ", "NUM", "DET", "PRON"}


def fold(s: str) -> str:
    return s.replace("ё", "е")


def audit_row(m, lemma: str, upos: str, wordform: str) -> tuple[str, str]:
    """Return (flag, dictionary suggestion) or ("", "") if the row looks fine."""
    if upos in SKIP_UPOS:
        return "", ""
    parses = m.parse(lemma)
    if not m.word_is_known(lemma):
        if upos == "PROPN":
            return "", ""
        wf = wordform.lower()
        sugg = {p.normal_form for p in m.parse(wf)} if m.word_is_known(wf) else set()
        return "unknown", ";".join(sorted(sugg))
    if upos == "PROPN":
        return "", ""
    if not any(p.normal_form == lemma for p in parses):
        yo = [p.normal_form for p in parses if fold(p.normal_form) == fold(lemma)]
        if yo:
            return "yo", yo[0]
        return "not_normal_form", ";".join(sorted({p.normal_form for p in parses}))
    allowed = UPOS_TO_OC.get(upos)
    if allowed and not any(p.tag.POS in allowed for p in parses if fold(p.normal_form) == fold(lemma)):
        return "pos_mismatch", ",".join(sorted({p.tag.POS or "?" for p in parses}))
    return "", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extracted", type=Path, default=Path("extracted_vocabulary_stanza.csv"))
    ap.add_argument("--out", type=Path, default=Path("lemma_audit.csv"))
    args = ap.parse_args()

    m = pymorphy3.MorphAnalyzer()
    covered = {(k[0], k[1]) for k in load_corrections()}
    out = []
    with args.extracted.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            lemma, upos = r["lemma"], r["upos"]
            if (lemma, upos) in covered or normalize_lexical(lemma) is None:
                continue  # already decided, or dropped by the mechanical rules
            flag, sugg = audit_row(m, lemma, upos, r["example_wordform"])
            if flag:
                out.append([flag, lemma, upos, r["verb_form"], r["example_wordform"], r["count"], sugg])
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["flag", "lemma", "upos", "verb_form", "example_wordform", "count", "dict_suggestion"])
        w.writerows(out)
    print(f"wrote {args.out}: {len(out)} rows to review")


if __name__ == "__main__":
    main()
