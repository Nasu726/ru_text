#!/usr/bin/env python3
"""Classify extracted Russian vocabulary into a deliberately small set of labels.

The classification is for corpus organization, not for estimating word importance
or CEFR level.  Each (lemma, UPOS) pair receives exactly one lexical class:

- everyday: concrete vocabulary strongly tied to ordinary-life contexts
- general: broadly usable vocabulary, including function words
- specialized: vocabulary strongly tied to academic/specialist contexts

The initial label is heuristic.  Existing labels are preserved on later runs so
that human corrections are not overwritten; only newly discovered (lemma, UPOS)
pairs are auto-classified.  Rows which no longer exist in the extracted
vocabulary are removed.

The heuristic combines:
1. the kinds of topic categories where the word actually occurred, and
2. wordfreq's Russian Zipf frequency only as a guard against classifying common
   words such as "север" as specialist merely because they happened to occur in
   a specialist topic.

No frequency score is written to the output CSV.  The output deliberately stays
small: lemma, upos, lexical_class.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from wordfreq import zipf_frequency

from extract_vocabulary_stanza import StanzaLemmatizer


VALID_CLASSES = {"everyday", "general", "specialized"}

# Categories designed mainly around concrete situations and ordinary objects.
EVERYDAY_CATEGORIES = {
    "日常生活・家庭",
    "買い物・消費",
    "交通・移動",
    "住居・生活インフラ",
    "旅行・観光",
    "友人・人間関係",
    "天気・季節・自然",
    "娯楽・文化",
    "スポーツ",
    "食文化",
    "祝日・伝統行事",
    "服装・外見",
    "人物描写・性格・感情",
    "家族・子育て・介護",
    "手続き・公共サービス",
    "緊急・安全・トラブル",
    "動植物・自然",
    "趣味・余暇",
    "コミュニケーション・言語行為",
    "ロシアの暮らしと文化",
}

# Academic/general-education categories are the clearest source of terminology.
SPECIALIZED_CATEGORIES = {
    "心理学(一般教養)",
    "歴史学(一般教養)",
    "考古学・人類学(一般教養)",
    "哲学・倫理学(一般教養)",
    "宗教学・神話学(一般教養)",
    "社会学(一般教養)",
    "政治学・国際関係(一般教養)",
    "経済学(一般教養)",
    "言語学(一般教養)",
    "法学・くらしの法律知識(一般教養)",
    "地理学(一般教養)",
    "天文学・宇宙科学(一般教養)",
    "自然科学トリビア(一般教養)",
    "美術史・音楽学(一般教養)",
    "演劇・映画学(一般教養)",
    "文学・出版文化(一般教養)",
}

# These are grammatical/function categories rather than topical vocabulary.
# Labeling them "general" avoids pretending that an adposition or conjunction
# belongs to a specialist topic just because of where it happened to occur.
FUNCTION_UPOS = {
    "ADP",
    "AUX",
    "CCONJ",
    "DET",
    "PART",
    "PRON",
    "SCONJ",
}


def load_manifest(path: Path) -> dict[str, str]:
    """Return filename -> category and reject malformed/duplicate filenames."""
    result: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"id", "category", "topic", "register", "filename"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"unexpected manifest columns: {reader.fieldnames!r}")
        for row in reader:
            filename = row["filename"]
            if filename in result:
                raise ValueError(f"duplicate filename in manifest: {filename}")
            result[filename] = row["category"]
    return result


def load_extracted_keys(path: Path) -> set[tuple[str, str]]:
    """Return the exact (lemma, UPOS) universe that must be classified."""
    keys: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row["lemma"], row["upos"]))
    return keys


def load_existing(path: Path) -> dict[tuple[str, str], str]:
    """Load human-reviewed labels if the classification file already exists."""
    if not path.exists():
        return {}

    labels: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = ["lemma", "upos", "lexical_class"]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected classification columns: {reader.fieldnames!r}")
        for row in reader:
            lexical_class = row["lexical_class"]
            if lexical_class not in VALID_CLASSES:
                raise ValueError(
                    f"invalid lexical_class {lexical_class!r} for "
                    f"{row['lemma']!r}/{row['upos']!r}"
                )
            labels[(row["lemma"], row["upos"])] = lexical_class
    return labels


def category_bucket(category: str) -> str:
    if category in EVERYDAY_CATEGORIES:
        return "everyday"
    if category in SPECIALIZED_CATEGORIES:
        return "specialized"
    return "general"


def auto_classify(
    lemma: str,
    upos: str,
    bucket_counts: Counter[str],
    category_count: int,
) -> str:
    """Assign one coarse lexical class; this is intentionally conservative."""
    if upos in FUNCTION_UPOS:
        return "general"

    total = sum(bucket_counts.values())
    if total == 0:
        return "general"

    everyday_share = bucket_counts["everyday"] / total
    specialized_share = bucket_counts["specialized"] / total

    # wordfreq is not used as an importance score.  It is only a sanity check
    # against treating a common general word as technical because it appeared
    # in one academic passage.
    zipf = zipf_frequency(lemma, "ru")

    # Strong, concentrated ordinary-life usage is the clearest everyday signal.
    if everyday_share >= 0.70 and specialized_share <= 0.15:
        return "everyday"

    # Technical vocabulary should be both strongly specialist-context-bound and
    # not already a very common Russian word.
    if specialized_share >= 0.70 and zipf < 4.0:
        return "specialized"

    # Breadth across several topic categories is evidence for general vocabulary.
    if category_count >= 3 or zipf >= 4.0:
        return "general"

    # Softer fallbacks for words with only a few observations.
    if specialized_share >= 0.55 and zipf < 3.5:
        return "specialized"
    if everyday_share >= 0.55:
        return "everyday"
    return "general"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("passages_dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated_texts/manifest.csv"),
    )
    parser.add_argument(
        "--extracted",
        type=Path,
        default=Path("extracted_vocabulary_stanza.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("vocabulary_classification.csv"),
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    extracted_keys = load_extracted_keys(args.extracted)
    existing = load_existing(args.out)

    # Keep token counts by broad context bucket, plus the actual category set.
    bucket_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    categories: dict[tuple[str, str], set[str]] = defaultdict(set)

    lemmatizer = StanzaLemmatizer()
    passage_paths = sorted(args.passages_dir.glob("*.txt"))
    for i, path in enumerate(passage_paths, start=1):
        if path.name not in manifest:
            raise ValueError(f"{path.name} is missing from manifest")
        category = manifest[path.name]
        bucket = category_bucket(category)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for word in lemmatizer.analyze(text):
            key = (word["lemma"], word["upos"])
            bucket_counts[key][bucket] += 1
            categories[key].add(category)
        if i % 50 == 0:
            print(f"[classification] {i}/{len(passage_paths)} passages processed")

    labels: dict[tuple[str, str], str] = {}
    preserved = 0
    generated = 0
    for key in extracted_keys:
        if key in existing:
            labels[key] = existing[key]
            preserved += 1
            continue
        lemma, upos = key
        labels[key] = auto_classify(
            lemma,
            upos,
            bucket_counts[key],
            len(categories[key]),
        )
        generated += 1

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "upos", "lexical_class"])
        for lemma, upos in sorted(labels):
            writer.writerow([lemma, upos, labels[(lemma, upos)]])

    summary = Counter(labels.values())
    print(
        f"wrote {args.out}: {len(labels)} rows "
        f"(preserved={preserved}, auto-classified={generated}); "
        f"everyday={summary['everyday']}, "
        f"general={summary['general']}, "
        f"specialized={summary['specialized']}"
    )


if __name__ == "__main__":
    main()
