# 意思決定支援のための統計プレイブック

本リポジトリは、社内外の記事（例：Qiita）で紹介した**統計手法**を、意思決定支援の観点で再現可能に実装・検証し、継続的に蓄積するための「統合リポジトリ」です。

## 目的 / Purpose

- 統計手法の「紹介」だけでなく、**業務課題 → 手法選択 → 評価 → 解釈 → 次のアクション**までの型を残す
- 記事とコードを1対1で紐づけ、検証可能な形でナレッジを蓄積する
- 同種の課題に対して再利用できる"分析部品"を増やす（前処理・CV・可視化・解釈テンプレ等）

## プロジェクト一覧 / Project Catalog

以下は記事・コードの対応表です（新規追加時はこの表を更新します）。

| ID | テーマ | 記事 | コード | 設計書 | 主要手法 | 位置づけ（いつ使うか） | Status |
|---:|---|---|---|---|---|---|---|
| 001 | PLS回帰で多重共線性＋少サンプルのKPI分析 | [Qiita](https://qiita.com/WE1CH-KAZU/items/8ed9f1d3d950611b477c) | [pls_regression.py](pls-regression/pls_regression.py) | [readme](pls-regression/pls_regression_readme.md) | PLS / PCR / Ridge / Lasso | KPIが束で動き、pが大きくnが小さい | stable |
| 002 | MLP回帰で非線形KPI分析 | [Qiita](https://qiita.com/WE1CH-KAZU/items/d4f4b5f7c9ab8ed779f2) | [mlp_regression.py](mlp-regression/mlp_regression.py) | [readme](mlp-regression/mlp_regression_readme.md) | MLP / Ridge | 非線形性が疑われる小〜中規模データ | stable |
| 003 | 価格弾力性をlogリンク関数でモデリング | 未出稿 | [price_elasticity_glm.py](price-elasticity-log-link/price_elasticity_glm.py) | - | GLM / Poisson / Negative Binomial | 需要予測・価格感応度の推定 | draft |
| 004 | 動的価格弾力性モデル | 未出稿 | [verify.py](dynamic-price-elasticity/verify.py) | - | ローリングGLM / 状態空間モデル | 時変する価格感応度の推定 | draft |
| 005 | データアナリストの知識とAI | [note](https://note.com/_we1ch_/n/n20e367ee6aaf) | - | - | エッセイ | - | stable |

> 運用ルール：Status は `draft / stable / deprecated` のいずれかを付与します。

## ディレクトリ構成 / Repository Structure

```
qiita_samples/
├── <article-slug>/          # 1記事 = 1フォルダ
│   ├── <article-slug>.md    # 記事原稿（Qiita向けMarkdown）
│   ├── *.py                 # 検証用コード（GitHubでpublic公開し記事からリンク）
│   ├── figs/                # 図（必要な場合）
│   └── *_readme.md          # モックデータ設計書（必要な場合）
└── .claude/                 # Claude Code 向けルール・スキル定義
```

## クイックスタート / Quickstart

Python 3.13 と [uv](https://docs.astral.sh/uv/) が必要。

```bash
uv sync
uv run python <article-slug>/<script>.py
```

## License

MIT License
