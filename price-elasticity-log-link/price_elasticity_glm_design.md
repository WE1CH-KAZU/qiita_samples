# price_elasticity_glm.py 設計書

## 概要

| 項目 | 内容 |
|------|------|
| ファイル名 | `price_elasticity_glm.py` |
| 対応記事 | `price-elasticity-log-link.md` |
| 目的 | GLM（Poisson + logリンク）による価格弾力性の推定と、OLSとの比較検証 |
| 実行コマンド | `uv run python price-elasticity-log-link/price_elasticity_glm.py` |

---

## モジュール構成

スクリプトは記事の章立てに対応した5つのセクションで構成される。

```
price_elasticity_glm.py
├── [3.1] データ準備
├── [3.2] GLM推定（Poisson + logリンク）
├── [3.3] OLSとの比較
├── [3.4] 結果の可視化
├── [3.5] 過分散の確認とNegative Binomialへの切り替え
├── [4.1] 複数変数への拡張
└── [4.3] 弾力性を使った最適価格の試算
```

---

## データフロー

```
乱数シード固定（seed=28）
        ↓
価格配列生成（Uniform[100, 500]、n=500）
        ↓
log-log関係でPoisson期待値μを計算
  log(μ) = log(1000) + (-1.5) * (log(P) - log(100))
        ↓
Poisson乱数で需要量を生成
        ↓
DataFrameに格納（price / demand / log_price）
        ↓
        ├── GLM推定（Poisson + log）→ 弾力性の点推定・区間推定
        ├── OLS推定              → 弾力性の点推定（平均点近似）
        ├── 可視化               → price_elasticity.png
        ├── Negative Binomial推定 → 過分散時の代替
        └── 拡張モデル推定       → プロモーション + 月次ダミー追加
```

---

## セクション別設計

### [3.1] データ準備

**目的**: 真の弾力性 $\beta = -1.5$ を既知とするシミュレーションデータを生成する。

| パラメータ | 値 | 意味 |
|-----------|-----|------|
| `n` | 500 | サンプルサイズ |
| `true_elasticity` | -1.5 | グラウンドトゥルース（推定後と比較する） |
| `baseline_demand` | 1000 | 基準価格100円での期待需要量 |
| `baseline_price` | 100 | スケール固定のための基準価格（円） |
| `price` | Uniform[100, 500] | 説明変数の価格配列 |

**データ生成式**:

```
log(μ) = log(baseline_demand) + true_elasticity * (log(P) - log(baseline_price))
demand ~ Poisson(μ)
```

基準価格100円のとき `E[demand] = 1000` になるようにスケールを固定する。`np.log(baseline_price)` を引くことで切片の解釈が `log(baseline_demand)` と一致する。

**出力**: `df.describe()` による基本統計の表示

---

### [3.2] GLM推定（Poisson + logリンク）

**目的**: Poisson GLMで価格弾力性 $\beta$ を推定し、真値との誤差を確認する。

**モデル式**:

```
demand ~ log_price
family = Poisson(link=Log())
```

`log_price` の係数が直接、価格弾力性 $\beta$ として解釈できる（log-logモデルの性質）。

**出力**:

- `model_glm.summary()`: statsmodelsの推定結果サマリ
- 点推定値 `elasticity_est`
- 95%信頼区間 `ci`

---

### [3.3] OLSとの比較

**目的**: 同一データにOLSを当てはめ、弾力性推定の精度を対比する。

**モデル式**:

```
demand ~ log_price
family = OLS（Gaussian）
```

OLSの `log_price` 係数は需要量の絶対値変化（$dQ/d(\log P)$）であり、弾力性そのものではない。弾力性への換算には以下の平均点近似を使う。

```
弾力性 = β_OLS / mean(demand)
```

**導出根拠**:

```
Q = α + β * log(P)
dQ/d(log P) = β
dQ/dP = β / P
E = dQ/dP * P/Q = β/Q  →  平均点での近似: β / mean(Q)
```

**出力**: OLS係数と弾力性の換算値

---

### [3.4] 結果の可視化

**目的**: GLMとOLSの当てはまりを視覚的に比較し、log-log関係の直線性を示す。

**出力ファイル**: `price-elasticity-log-link/price_elasticity.png`（dpi=150）

| パネル | 軸 | 内容 |
|--------|-----|------|
| 左（原スケール） | x: 価格（円）、y: 需要量 | 観測値の散布 + GLM曲線（べき乗）+ OLS直線 |
| 右（log-logスケール） | x: log(価格)、y: log(需要量) | 観測値の散布 + GLM直線（傾き = 弾力性） |

**フォント設定**: `Hiragino Sans` → `AppleGothic` → `DejaVu Sans` の順でフォールバック。GUI非表示環境のため `plt.switch_backend("Agg")` を使用。

---

### [3.5] 過分散の確認とNegative Binomialへの切り替え

**目的**: Poisson適合の妥当性を定量指標で確認し、過分散時の代替モデルを示す。

**過分散の判定指標**:

```
Pearson χ² / 残差自由度 ≈ 1  → Poisson 適合
Pearson χ² / 残差自由度 >> 1 → 過分散（NegativeBinomial を検討）
```

`model_glm.pearson_chi2` と `model_glm.df_resid` から計算。生データの `分散/平均` は価格差による変動を含むため、適切な過分散指標にはならない（記事でも注記）。

**Negative Binomialモデル**:

```
family = NegativeBinomial(alpha=1.0, link=Log())
```

`alpha` は過分散パラメータの初期値。実データでは推定値（`model_nb.params['alpha']`）を確認する。

**出力**: Pearson χ²/df の値と NB 推定弾力性

---

### [4.1] 複数変数への拡張

**目的**: 実務的なモデルとして、価格以外の変数を制御した純粋な価格弾力性を推定する。

**追加変数**:

| 変数 | 型 | 内容 |
|------|-----|------|
| `promotion` | 0/1 バイナリ | プロモーション実施フラグ（乱数生成） |
| `month` | 1〜12 整数 | 月次ダミー（季節性の代理変数） |

**モデル式**:

```
demand ~ log_price + promotion + C(month)
family = Poisson(link=Log())
```

`C(month)` で月をカテゴリ変数として扱い、11個のダミー変数に展開する。`log_price` の係数が「プロモーション・季節性を制御した上での」価格弾力性。

---

### [4.3] 弾力性を使った最適価格の試算

**目的**: 推定弾力性から収益最大化価格を試算する。

**関数**: `revenue_simulation(price_range, baseline_price, baseline_demand, elasticity) -> pd.DataFrame`

| 引数 | 型 | 説明 |
|------|-----|------|
| `price_range` | `np.ndarray` | 試算する価格の配列（円） |
| `baseline_price` | `float` | 基準価格（円） |
| `baseline_demand` | `float` | 基準価格での需要量（個） |
| `elasticity` | `float` | 推定価格弾力性（負の値） |

**需要予測式**（定弾力性モデル）:

```
demand(P) = baseline_demand * (P / baseline_price) ** elasticity
revenue(P) = P * demand(P)
```

**出力**: `sim.loc[sim["revenue"].idxmax()]` による収益最大化価格・需要・収益の表示

> **注意**: 弾力的（$|E| > 1$）な商品では収益関数は価格について単調減少するため、試算価格範囲の下限が最適価格として選ばれる。実務では限界費用・在庫制約との組み合わせが必要。

---

## 依存パッケージ

| パッケージ | 用途 |
|-----------|------|
| `numpy` | 乱数生成・数値計算 |
| `pandas` | DataFrame操作・統計表示 |
| `statsmodels` | GLM・OLS推定（`smf.glm`, `smf.ols`） |
| `matplotlib` | 散布図・回帰曲線の描画 |

---

## 実行時の確認事項

1. `uv run ruff check price-elasticity-log-link/price_elasticity_glm.py` でリントを通す
1. `uv run ruff format price-elasticity-log-link/price_elasticity_glm.py` でフォーマットを適用する
1. `uv run python price-elasticity-log-link/price_elasticity_glm.py` で実行し、以下を目視確認する

| 確認項目 | 期待値 |
|---------|--------|
| 推定弾力性 | ≈ −1.49（真値 −1.50 と誤差 1%未満） |
| 95%信頼区間 | 真値 −1.5 を含む |
| OLS弾力性 | ≈ −1.62（8%程度のバイアス） |
| Pearson χ²/df | ≈ 1.0（Poisson 適合） |
| NB推定弾力性 | ≈ −1.49（Poisson と近似一致） |
| 調整済み弾力性 | ≈ −1.49（単変数と近似一致） |
| 画像出力 | `price_elasticity.png` が生成される |

---

## 設計上の判断

### なぜ `log_price` を前処理してDataFrameに持つか

`smf.glm(formula="demand ~ log_price", ...)` は数式文字列で変数を参照する。
`np.log(price)` を `formula="demand ~ np.log(price)"` のように数式中に書くことも可能だが、以下の理由で明示的な列として準備する。

- 可視化（x軸）での再利用が容易
- `model_glm.params["log_price"]` でパラメータ名が明確になる

### なぜ `baseline_price` を引数に切り出すか

データ生成とシミュレーション関数の両方で同じ基準価格（100円）を参照するため、マジックナンバーの重複を避けるために定数として切り出す。

### なぜ `plt.switch_backend("Agg")` を冒頭で呼ぶか

macOS の対話的セッション以外（CIや `uv run` 経由）では GUI が利用できない。
`savefig()` だけを使うので `Agg` バックエンドで十分。
