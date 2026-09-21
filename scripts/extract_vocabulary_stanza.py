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
  1. Stanzaの品詞タグ(Universal POS, 例: NOUN/VERB/ADJ)は「参考情報」として
     出力するのみで、CEFRレベルの判定には使わない(判定はClaudeの教育的
     判断で別途行う想定、README参照)。
  2. Stanzaは文単位でパイプラインを回すため、大量の文章を処理する際は
     GPUが無い環境だとCPU推論がそれなりに遅い可能性がある。まずは
     少量(パイロット規模)で実行時間を確認してから本番スケールに
     進めることを推奨する。
  3. 入力テキストファイルの想定形式は「1行1パッセージ」のプレーンテキスト。
     実際に大量生成した文章をどう保存するかはまだ決まっていないため、
     このスクリプトの load_passages() を実際のファイル形式に合わせて
     書き換える必要がある可能性が高い。

Why lemma単位ではなく (lemma, upos, verb_form, case) 単位で集計するのか:
  ユーザー指摘により追加。деепричастие(副動詞)やпричастие(形動詞)は
  UD(Universal Dependencies)のレンマ化では定動詞と同じレンマに正規化される
  (例: "читая"(副動詞)も"читать"(不定形)もlemmaは"читать")。そのため
  lemmaだけで集計すると、副動詞・形動詞が実際に出現したという事実が
  消えてしまう。Stanzaは`word.feats`にUD素性文字列(例: "VerbForm=Conv"が
  副動詞、"VerbForm=Part"が形動詞、"Case=Gen"が生格など)を返すため、
  これを解析して verb_form / case を別列として残し、名詞・数詞の格変化や
  動詞の格支配(前置詞+格)がどの程度出現したかを後から集計・確認できる
  ようにする。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import stanza


def load_passages(path: Path) -> list[str]:
    """1行1パッセージのプレーンテキストを読む。
    Why: 生成文章の保存形式が未確定のため、まずは最も単純な形式を仮定する。
    実際の生成パイプラインが固まったら、この関数だけを差し替えればよい。
    """
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
    parser.add_argument("input_text", type=Path, help="1行1パッセージのプレーンテキストファイル")
    parser.add_argument(
        "--out", type=Path, default=Path("extracted_vocabulary_stanza.csv"),
        help="出力CSVのパス",
    )
    args = parser.parse_args()

    print("[setup] Stanzaパイプラインを初期化中(初回はモデルダウンロードが発生します)...", file=sys.stderr)
    lemmatizer = StanzaLemmatizer()
    passages = load_passages(args.input_text)
    print(f"[input] {len(passages)} パッセージを読み込みました", file=sys.stderr)

    # Why (lemma, upos, verb_form, case) をキーにするか: モジュールdocstring
    # 参照。同じlemmaでも「定動詞として」「副動詞として」「形動詞として」出た
    # 回数を別々に数えられるようにする。
    agg: dict[tuple[str, str, str, str], dict] = {}
    for i, passage in enumerate(passages, start=1):
        for w in lemmatizer.analyze(passage):
            key = (w["lemma"], w["upos"], w["verb_form"], w["case"])
            if key not in agg:
                agg[key] = {"count": 0, "example_wordform": w["wordform"]}
            agg[key]["count"] += 1
        if i % 50 == 0:
            print(f"[progress] {i}/{len(passages)} パッセージ処理済み", file=sys.stderr)

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "upos", "verb_form", "case", "example_wordform", "count"])
        for (lemma, upos, verb_form, case), d in sorted(agg.items(), key=lambda kv: -kv[1]["count"]):
            writer.writerow([lemma, upos, verb_form, case, d["example_wordform"], d["count"]])

    distinct_lemmas = len(set(k[0] for k in agg))
    gerunds = sum(1 for k in agg if k[2] == "Conv")
    participles = sum(1 for k in agg if k[2] == "Part")
    print(f"\n総抽出語数(異なりlemma数): {distinct_lemmas}")
    print(f"副動詞(деепричастие, VerbForm=Conv)として出現したlemma×形の組: {gerunds}")
    print(f"形動詞(причастие, VerbForm=Part)として出現したlemma×形の組: {participles}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
