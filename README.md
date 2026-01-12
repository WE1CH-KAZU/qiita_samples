# 意思決定支援のための統計プレイブック

本リポジトリは、社内外の記事（例：Qiita）で紹介した **統計手法**を、意思決定支援の観点で再現可能に実装・検証し、継続的に蓄積するための「統合リポジトリ」です。  


## 目的 / Purpose
- 統計手法の「紹介」だけでなく、**業務課題 → 手法選択 → 評価 → 解釈 → 次のアクション**までの型を残す
- 記事とコードを1対1で紐づけ、検証可能な形でナレッジを蓄積する
- 同種の課題に対して再利用できる“分析部品”を増やす（前処理・CV・可視化・解釈テンプレ等）


## プロジェクト一覧 / Project Catalog
以下は記事・コードの対応表です（新規追加時はこの表を更新します）。

| ID | テーマ | 記事 | コード | 主要手法 | Readme | 位置づけ（いつ使うか） | Status |
|---:|---|---|---|---|---|---|---|
| 001 | PLS回帰で多重共線性＋少サンプルのKPI分析 | <Qiita URL> | `projects/001_pls_kpi_demo/` | PLS / PCR / Ridge / Lasso | `script/pls_regression.md` | KPIが束で動き、pが大きくnが小さい | stable |
| ... | ... | ... | ... | ... | ... | ... | draft |

> 運用ルール：Status は `draft / stable / deprecated` のいずれかを付与します。

## 全体ディレクトリ構成 / Repository Structure
複数プロジェクトを1つのリポジトリで運用するため、以下の規約を採用します（必要に応じて調整してください）。

- `src/` : 記事に関連するnotebook(1テーマ=1フォルダ)
- `scripts/` : 記事に関するReadme
- `settings/` : 記事に関する環境テキストファイル

## クイックスタート / Quickstart
### 1) 環境
- Python: 3.11+ 推奨
- 主要パッケージ: numpy / pandas / scikit-learn / matplotlib など

## License
- MIT License
