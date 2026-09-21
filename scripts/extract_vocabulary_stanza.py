#!/usr/bin/env python3
"""
extract_vocabulary_stanza.py

Why (この用途):
  トピック集(data/topics/)を元に生成したロシア語文章の集合から、語彙を
  抽出・レンマ化するためのスクリプト。辞書ベースの形態素解析器
  (pymorphy等)は未知語(新しい借用語、固有名詞、生成文章特有の言い回し)に
  弱いため、PyTorchベースで文脈を見て推論する Stanza を使う。

  Why this cannot run in the current sandbox (確認済みの事実、推測ではない):
    stanza.resources.common.download() のソースを直接確認したところ、
    ロシア語モデル本体 (default.zip) のダウンロードURLは
    "https://huggingface.co/stanfordnlp/stanza-{lang}/resolve/..." に
    テンプレート化されており、huggingface.co 以外からの配布経路が無い。
    huggingface.co はこのサンドボックスの組織ポリシーでCONNECT自体が
    403拒否されることを `curl` で直接確認済み(OpenRussianのAPIと同種の
    制約)。したがって pip install stanza 自体は成功するが、モデルの
    ダウンロードだけがサンドボックス内で不可能。
    ユーザーが用意するGitHubリポジトリ側(ネットワーク制限のない環境、
    GitHub Actions等)での実行を想定する。

  Why cache (ユーザー指定の「cache利用推奨」への対応):
    stanza.download() はデフォルトで `~/.cache/stanza/` にモデルを保存し、
    2回目以降は再ダウンロードしない。GitHub Actions上で実行する場合は
    このディレクトリを actions/cache でキャッシュすることを強く推奨する
    (ロシア語モデルは数百MB規模になりうり、PRのたびに毎回落とすのは
    非現実的)。ローカル実行の場合は何もしなくても2回目以降は自動的に
    キャッシュが効く。

設計方針 (デザインパターン):
  単一責任(SRP)で分割する:
    - load_passages(): 入力テキストの読み込みのみ
    - StanzaLemmatizer: Stanzaパイプラインの初期化とレンマ化のみを担当する
      薄いラッパー(将来 spaCy 等に差し替える場合もこのクラスだけ変更すればよい
      = Strategyパターンの単純な適用)
    - main(): 入出力とサマリー出力のオーケストレーションのみ

失敗しそうな部分 (事前の注記):
  1. 本スクリプトはCEFRレベルを判定するためのものではない。主目的は、
     大量生成した一般向け文章から語彙候補を抽出・集計することにある。
     一方、形動詞・副動詞や名詞・形容詞・数詞の格変化はB1〜B2付近の
     学習で重要である。形動詞・副動詞は verb_form ごとに行を分けて集計し、
     格は同じlemmaを分裂させないよう case_counts に格別頻度をまとめる。
  2. Stanzaは文単位でパイプラインを回すため、大量の文章を処理する際は
     GPUが無い環境だとCPU推論がそれなりに遅い可能性がある。まずは
     少量(パイロット規模)で実行時間を確認してから本番スケールに
     進めることを推奨する。
  3. 入力は `data/generated_texts/ru_generated_texts.csv`
     (列: category,topic,register,text) を前提とする。生成文の由来
     (どのトピックから出たか)を後から追えるようにするため、単なる
     プレーンテキストではなくCSVにしている。
  4. VerbForm=Conv/Part の判定は、SynTagRus(ロシア語UDツリーバンク)から
     学習した統計的タガーによる推論であり、辞書引きやルールベースの
     決め打ちではない。そのため以下のような境界事例で誤判定しうる:
     形容詞化した形動詞(例: "блестящий"が独立した形容詞的な意味で
     使われる場合)、副詞化した副動詞(例: "молча")、述語的に使われる
     短語尾形動詞と述語形容詞の混同、そしてAI生成文の言い回しが
     SynTagRusの学習データ(新聞・小説等の伝統的な書き言葉)の分布から
     大きく外れている場合の精度低下。大量生成したコーパスに対しては、
     verb_form=Conv/Partでタグ付けされた語をいくつか無作為抽出して
     目視確認することを推奨する。

Why (lemma, upos, verb_form) 単位で集計し、caseは集約するのか:
  деепричастие(副動詞)やпричастие(形動詞)はUD(Universal Dependencies)の
  レンマ化では定動詞と同じレンマに正規化される(例: "читая"(副動詞)も
  "читать"(不定形)もlemmaは"читать")。そのため verb_form は集計キーに残し、
  副動詞・形動詞が実際に出現したという事実を統計上明確にする。

  一方、Caseまで集計キーにすると "новый / нового / новому" のような
  単なる格変化が同じlemmaの複数行に分裂し、語彙リストとして扱いにくくなる。
  そこでCaseはキーから外し、各 (lemma, upos, verb_form) 行の case_counts に
  格別の出現回数を集約する。これにより語彙の重複を避けつつ、名詞・形容詞・
  数詞などの格変化がどの程度現れたかは統計として保持できる。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import stanza


def load_passages(path: Path) -> list[str]:
    """`data/generated_texts/ru_generated_texts.csv` 形式のCSVから
    text列(生成文章本文)だけを取り出してリストで返す。

    Why CSVから読むのか: モジュールdocstring(失敗しそうな部分 3)参照。
    category/topic/register列は本抽出処理では使わないが、ファイル自体には
    残しておく(生成文の由来を追えるようにするため)。ここではtextだけを
    使う。
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        return [row["text"].strip() for row in csv.DictReader(f) if row["text"].strip()]


def parse_feats(feats: str | None) -> dict[str, str]:
    """UD素性文字列("Case=Gen|Number=Sing|...")を辞書に分解する。

    Why: Stanzaは feats を1本のパイプ区切り文字列で返すため、
    verb_form(=VerbForm)やcase(=Case)だけを個別列として取り出すために
    毎回パースする。feats が None または空文字列の場合は空辞書を返す
    (助詞・接続詞など、格や態を持たない品詞では feats が空になる)。
    """
    if not feats:
        return {}
    result = {}
    for pair in feats.split("|"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key] = value
    return result


class StanzaLemmatizer:
    """Stanzaパイプラインの初期化とレンマ化のみを担当する薄いラッパー。"""

    def __init__(self) -> None:
        # Why download_method=REUSE_RESOURCES: 既にキャッシュがあれば
        # 再ダウンロードせず即座に使う(ユーザー指定の「cache利用推奨」)。
        # 初回のみ実際のダウンロードが発生する。
        self._pipeline = stanza.Pipeline(
            lang="ru",
            processors="tokenize,pos,lemma",
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
            verbose=False,
        )

    def analyze(self, text: str) -> list[dict]:
        """1語ごとに lemma/upos/verb_form/case/wordform を持つ辞書のリストを返す。
        句読点・記号・未分類トークンはスキップする。

        Why verb_form/case を個別に取り出すか: モジュールdocstring参照
        (副動詞・形動詞はlemmaだけでは判別できないため)。
        """
        doc = self._pipeline(text)
        result = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos in ("PUNCT", "SYM", "X"):
                    continue
                feats = parse_feats(word.feats)
                result.append({
                    "lemma": word.lemma.lower(),
                    "upos": word.upos,
                    "verb_form": feats.get("VerbForm", ""),
                    "case": feats.get("Case", ""),
                    "wordform": word.text,
                })
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_text", type=Path,
        help="生成文章CSV(列: category,topic,register,text)。"
             "既定の置き場所は data/generated_texts/ru_generated_texts.csv",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("extracted_vocabulary_stanza.csv"),
        help="出力CSVのパス",
    )
    args = parser.parse_args()

    print("[setup] Stanzaパイプラインを初期化中(初回はモデルダウンロードが発生します)...", file=sys.stderr)
    lemmatizer = StanzaLemmatizer()
    passages = load_passages(args.input_text)
    print(f"[input] {len(passages)} パッセージを読み込みました", file=sys.stderr)

    # verb_form は副動詞・形動詞を統計上分離するためキーに残す。
    # case は同じlemmaを格ごとに別行へ分裂させないため、行内のcase_countsに集約する。
    agg: dict[tuple[str, str, str], dict] = {}
    for i, passage in enumerate(passages, start=1):
        for w in lemmatizer.analyze(passage):
            key = (w["lemma"], w["upos"], w["verb_form"])
            if key not in agg:
                agg[key] = {
                    "count": 0,
                    "case_counts": {},
                    "example_wordform": w["wordform"],
                }
            agg[key]["count"] += 1
            if w["case"]:
                case_counts = agg[key]["case_counts"]
                case_counts[w["case"]] = case_counts.get(w["case"], 0) + 1
        if i % 50 == 0:
            print(f"[progress] {i}/{len(passages)} パッセージ処理済み", file=sys.stderr)

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "upos", "verb_form", "case_counts", "example_wordform", "count"])
        for (lemma, upos, verb_form), d in sorted(agg.items(), key=lambda kv: -kv[1]["count"]):
            case_counts = "|".join(
                f"{case}:{count}"
                for case, count in sorted(d["case_counts"].items())
            )
            writer.writerow([
                lemma,
                upos,
                verb_form,
                case_counts,
                d["example_wordform"],
                d["count"],
            ])

    distinct_lemmas = len(set(k[0] for k in agg))
    gerunds = sum(1 for k in agg if k[2] == "Conv")
    participles = sum(1 for k in agg if k[2] == "Part")
    print(f"\n総抽出語数(異なりlemma数): {distinct_lemmas}")
    print(f"副動詞(деепричастие, VerbForm=Conv)として出現したlemma×形の組: {gerunds}")
    print(f"形動詞(причастие, VerbForm=Part)として出現したlemma×形の組: {participles}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
