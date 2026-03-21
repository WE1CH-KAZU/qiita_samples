# git操作ルール

## ブランチ戦略

- **main**: リリース済みの安定版。直接コミットしない
- **stage**: mainへのマージ前の最終確認ブランチ
- **develop**: 日常的な作業ブランチ

### 作業フロー

```
develop → stage → main
```

1. 作業は `develop` ブランチで行う
2. 公開前の最終確認は `stage` ブランチにマージして行う
3. 確認が取れたら `stage` から `main` にマージする

## 基本方針

- コミットはユーザーから明示的に依頼された場合のみ実行する
- リモートへのプッシュはユーザーから明示的に依頼された場合のみ実行する
- ブランチの作成・削除はユーザーから明示的に依頼された場合のみ実行する
- `main` ブランチに直接コミット・プッシュしない

## コミットメッセージ

[Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に準拠し、**description は必ず英語で記述する（日本語不可）**。

```
<type>(<scope>): <description>
```

description に日本語を使うのは誤り。type・scope は英数字・記号のみのため、英語ルールは description に適用される。

```
# 良い例
feat: add problem-connection flow to article-outline skill
docs: update MLP regression article link and status to stable
fix(price-elasticity-log-link): correct GLM coefficient interpretation

# 悪い例（description が日本語になっている）
feat: article-outlineスキルに課題接続フローを追加
docs: MLP回帰記事のQiitaリンクとステータスをstableに更新
```

- scope には変更対象を示す識別子を入れる
  - 記事ディレクトリに関する変更: 記事スラッグ（例: `price-elasticity-log-link`）
  - プロジェクト全体の設定・ルール変更: scope を省略してよい（例: `chore: ...`）

このプロジェクトで使用するtype:

- `docs:` 記事ファイル（.md）の追加・更新
- `feat:` 新しい記事ディレクトリ・スキル・設定の追加
- `fix:` 記事の誤記修正・コードのバグ修正
- `chore:` 依存パッケージの追加・環境設定の変更

## ステージングのルール

- `git add` はコミット対象のファイルを個別に指定する（`git add .` や `git add -A` は使わない）

## 禁止操作

- `git push --force` は実行しない
- `git reset --hard` はユーザーに確認してから実行する
- `git commit --amend` はユーザーに確認してから実行する
- `.env` など機密情報を含む可能性があるファイルはコミットしない
