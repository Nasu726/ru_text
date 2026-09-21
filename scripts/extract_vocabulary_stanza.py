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

    def lemmatize(self, text: str) -> list[tuple[str, str]]:
        """(lemma, upos) のリストを返す。句読点等はスキップする。"""
        doc = self._pipeline(text)
        result = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos in ("PUNCT", "SYM", "X"):
                    continue
                result.append((word.lemma.lower(), word.upos))
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

    lemma_data: dict[str, dict] = {}
    for i, passage in enumerate(passages, start=1):
        for lemma, upos in lemmatizer.lemmatize(passage):
            if lemma not in lemma_data:
                lemma_data[lemma] = {"upos": upos, "count": 0}
            lemma_data[lemma]["count"] += 1
        if i % 50 == 0:
            print(f"[progress] {i}/{len(passages)} パッセージ処理済み", file=sys.stderr)

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "upos", "count"])
        for lemma, d in sorted(lemma_data.items(), key=lambda kv: -kv[1]["count"]):
            writer.writerow([lemma, d["upos"], d["count"]])

    print(f"\n総抽出語数(異なり語数): {len(lemma_data)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
