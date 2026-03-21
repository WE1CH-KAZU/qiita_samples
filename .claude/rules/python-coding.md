# Pythonコーディングルール

このプロジェクトでPythonコードを書く際は以下のルールに従うこと。

## 環境ルール

- パッケージのインストールは必ず `uv add <package>` を使う（`pip install` は使わない）
- スクリプトの実行は `uv run python <script.py>` を使う
- 検証用スクリプトは記事ディレクトリ（`<article-slug>/`）内に配置する

## コードスタイル

- スクリプトの冒頭にはモジュール docstring でそのスクリプトの内容・目的を明記する

```python
"""価格弾力性モデルの検証スクリプト。

対数リンク関数を用いたGLMで価格弾力性を推定し、
係数の解釈と予測結果を確認する。
"""
```

- インデントはスペース4つ
- 変数名・関数名はスネークケース（例: `price_elasticity`）
- 定数はアッパースネークケース（例: `MAX_ITER`）
- 関数の中で定義するネストされた関数には冒頭に `_` を付けて区別する（例: `_helper()`）
- 1行の長さは120文字以内
- 型ヒントは関数の引数・戻り値に付ける
- 関数・クラスには Google style docstrings を記入する

```python
def calc_elasticity(price: float, quantity: float) -> float:
    """価格弾力性を計算する。

    Args:
        price: 価格の変化率
        quantity: 数量の変化率

    Returns:
        価格弾力性の値
    """

    def _normalize(value: float) -> float:
        """内部処理: 値を正規化する。"""
        return value / 100.0

    return _normalize(quantity) / _normalize(price)
```

## コメントルール

- 処理の「何を」ではなく「なぜ」を説明するコメントを入れる
- 読者向けの `.py` ファイルには、各ブロックの意図を説明するコメントを必ず入れる
- マジックナンバーには必ずコメントで意味を説明する（例: `alpha = 0.05  # 有意水準5%`）

## 動作確認ルール

- 記事から参照する `.py` ファイルはすべて以下の手順で確認する
  1. `uv run ruff check <script.py>` でリントを通す
  2. `uv run ruff format <script.py>` でフォーマットを適用する
  3. `uv run python <script.py>` で実行して出力を確認する
- 実行エラーが出た場合は原因を特定して `.py` ファイルを修正してから記事に反映する
- 実行中に警告が出た場合は、フィルター（`grep -v`・`-W ignore`・`warnings.filterwarnings` など）で握りつぶさず、一つずつ原因を特定して対処する
  - 警告は根本的な問題のシグナルであることが多い（例: `PerfectSeparationWarning` → データ生成の欠陥）
  - 警告が多すぎて出力が読めない場合は、まず警告の種類と件数を把握し（`2>&1 | sort | uniq -c`）、原因を調べてから対処する
- 乱数を使う場合はシード値を28に固定して再現性を確保する（例: `np.random.seed(28)`）
- 数値計算の結果は目視で妥当性を確認する（オーダー感が正しいか）

## 依存パッケージルール

- 使用パッケージは `pyproject.toml` の `dependencies` に記録される（`uv add` で自動追加）
- 標準ライブラリで代替できる場合は外部パッケージを追加しない
- バージョン固定が必要な場合は `uv add "package==x.y.z"` で指定する
