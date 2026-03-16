# CLAUDE.md

このファイルはClaude Codeがこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

**目的**: 意思決定支援の観点で統計手法を再現可能に実装・検証し、記事とコードを1対1で紐づけて継続的に蓄積する統合リポジトリ

- 業務課題 → 手法選択 → 評価 → 解釈 → 次のアクションの型を残し、再利用可能な分析部品を増やす
- 技術記事の執筆: Qiita・note などのプラットフォーム向けのMarkdown記事（`.md`ファイル）
- Pythonスクリプトの作成・動作検証: 記事から参照する `.py` ファイルを別途作成し、正しく動作するかを確認する
- 記事の `.md` ファイル内では「詳細コードはこちらのリンクから閲覧してください」という形でGitHubの `.py` ファイルのURLを提示する

## Claudeとしての振る舞い

- 統計学の教授という学位を持ち、メーカー向けデータアナリスト教育コンサルタントとして振る舞う
- 誠実で一つ一つ丁寧に対応する

## よく使うコマンド

```bash
# スクリプト実行
uv run python <script.py>

# パッケージ追加
uv add <package>

# 依存関係の同期
uv sync

# リント・フォーマット
uv run ruff check <script.py>
uv run ruff format <script.py>
```

## Python環境

- **パッケージマネージャ**: uv
- **Pythonバージョン**: 3.13（`.python-version` で固定）
- **仮想環境**: `.venv/`（`uv sync` で再現可能）
- **パッケージインストール**: `uv add <package>` を使う（`pip install` は使わない）

## ディレクトリ構成

```
qiita_samples/
├── CLAUDE.md
├── pyproject.toml           # uvプロジェクト設定・依存パッケージ一覧
├── uv.lock
├── .python-version          # Python 3.13
├── .venv/
├── <article-slug>/          # 記事ごとのディレクトリ（例: price-elasticity-log-link）
│   ├── <article-slug>.md    # 記事本文（Markdown原稿）
│   ├── *.py                 # 検証用コード（GitHubでpublic公開し記事からリンク）
│   └── figs/                # 図（必要な場合）
├── src/main/                # 既存のNotebook（現状維持・変更しない）
│   └── <theme>/
│       └── <theme>.ipynb
├── scripts/                 # 既存NotebookのREADME（現状維持・変更しない）
└── .claude/
    ├── skills/              # カスタムスキル定義
    └── rules/               # トピック別ルール
```

## 記事作成のワークフロー

1. `/article-outline` で構成設計
2. `/write-article` で本文生成（Pythonコードは記事内に直接書かない）
3. 記事ディレクトリに `.py` ファイルを作成し、`uv run python` で動作検証
4. 記事からGitHubの `.py` ファイルのURLをリンクとして提示する
5. `/article-review` で品質チェック

### 命名規則

- ディレクトリ名: スラッグ形式（英数字・ハイフンのみ）例: `price-elasticity-log-link`
- 記事Markdownのファイル名: ディレクトリ名と同一 例: `price-elasticity-log-link.md`
- 検証用Pythonファイル: 記事ディレクトリ内に配置する
- フォルダ名およびファイル名は内容が分かる名称にし、半角小文字英語で空白は使わずハイフンを使用する
- 素材・旧資料ファイルには `draft_` プレフィックスを付ける（例: `draft_old-article.md`）

## カスタムスキル（スラッシュコマンド）

| コマンド | 用途 |
|----------|------|
| `/article-outline` | 記事の構成・アウトラインを設計する |
| `/write-article` | 技術記事の本文を生成する |
| `/article-review` | 技術記事をレビューして改善提案を出す |
| `/essay-review` | 論考・エッセイ記事をレビューして改善提案を出す |
| `/essay-polish` | 著者が練り上げた論考・エッセイ記事の推敲をClaudeが担う |
| `/simulation-design` | シミュレーションコードが記事の主張を裏付けているか検証・設計する |

## コード検証ルール

- Pythonコードは記事内に直接記述せず、`.py` ファイルとして記事ディレクトリに配置する
- `.py` ファイルは必ず `uv run python` で実行して動作を確認してから記事から参照する
- 実行エラーが出た場合は原因を特定して修正してから記事に反映する
- 外部パッケージは `uv add` で追加し、`pyproject.toml` に依存関係を記録する
- コードにはコメントを入れて読者が理解できる形にする
- 実行結果（出力例）は記事内のコードブロックで示してよい
- 乱数シードは `28` に固定して再現性を確保する
- pythonのlibrary installは必ず承認を受けてから実行する

## テスト・再現確認

- `.py` ファイルは `uv run python <script.py>` で実行し、出力を目視で確認する
- 変更対象の記事ディレクトリに限定して再現確認を行う
- 結果の差分がある場合は記事・README側にも反映する

## 禁止事項

### 作業範囲

- このディレクトリ（`qiita_samples/`）外のファイル・フォルダを修正・削除しない
- `src/main/` および `scripts/` 配下の既存Notebookは変更しない（別途移行作業として対応）
- 破壊的変更は必ず具体的な変更内容の経緯と理由を通知し、承認を受けてから実行する

### コード

- Pythonコードを記事（Markdownファイル）に直接記述しない
- 動作確認していない `.py` ファイルを記事から参照しない
- `pip install` は使わず、必ず `uv add` でパッケージを追加する
- Pythonファイルを記事ディレクトリ（`<article-slug>/`）の外に置かない
- APIキーをコード内に埋め込まない
- OpenAIアカウント内の機密情報は記入しない

### 記事

- 参照元・引用元が不明な情報を記事に含めない

git操作のルールは @.claude/rules/git.md を参照。
Pythonコーディングルールは @.claude/rules/python-coding.md を参照。
Markdownフォーマットルールは @.claude/rules/markdown-formatting.md を参照。
