"""価格弾力性モデルの検証スクリプト。

対数リンク関数を用いたGLM（Poisson回帰）で価格弾力性を推定し、
係数の解釈・予測結果・OLSとの比較を確認する。
記事 price-elasticity-log-link.md の掲載コードと一致している。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# GUIなしで画像保存するためのバックエンドに切り替え
plt.switch_backend("Agg")
# macOS で利用可能な日本語フォントを優先指定（なければ DejaVu Sans にフォールバック）
plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]

# ──────────────────────────────────────────────
# 3.1 データの準備
# ──────────────────────────────────────────────

# 再現性のためにシード固定
np.random.seed(28)
n = 500

# 価格: 100〜500円の一様乱数
price = np.random.uniform(100, 500, n)

# 真の価格弾力性
true_elasticity = -1.5
baseline_demand = 1000  # 基準価格（100円）における需要量
baseline_price = 100  # 基準価格（円）

# log(Q) = log(baseline) + β * (log(P) - log(P0)) というlog-log関係でデータ生成
# → P=P0 のとき E[Q]=baseline_demand になるようにスケールを固定する
log_mu = np.log(baseline_demand) + true_elasticity * (np.log(price) - np.log(baseline_price))
# Poisson分布に従う需要量を生成
demand = np.random.poisson(lam=np.exp(log_mu))

df = pd.DataFrame(
    {
        "price": price,
        "demand": demand,
        "log_price": np.log(price),
    }
)

print("=== データの基本統計 ===")
print(df.describe().round(2))

# ──────────────────────────────────────────────
# 3.2 GLMの推定（Poisson + logリンク）
# ──────────────────────────────────────────────

# GLM: Poisson分布 + logリンク関数
model_glm = smf.glm(
    formula="demand ~ log_price",
    data=df,
    family=sm.families.Poisson(  # 誤差分布: Poisson分布（カウントデータ向け）
        link=sm.families.links.Log()  # リンク関数: log（非負制約 + 弾力性の直読み）
    ),
).fit()

print(model_glm.summary())

# 弾力性の確認
elasticity_est = model_glm.params["log_price"]
ci = model_glm.conf_int().loc["log_price"]

print("\n=== 価格弾力性の推定結果 ===")
print(f"推定値:       {elasticity_est:.4f}")
print(f"95%信頼区間:  [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"真の弾力性:   {true_elasticity}")

# ──────────────────────────────────────────────
# 3.3 OLSとの比較
# ──────────────────────────────────────────────

# 比較用: OLS（線形回帰）
model_ols = smf.ols(
    formula="demand ~ log_price",
    data=df,
).fit()

print(f"OLS 推定係数（log_price）: {model_ols.params['log_price']:.4f}")
print("※OLSのlog_price係数は弾力性そのものではなく解釈が難しい")

# OLSモデル（demand ~ log_price）における弾力性の導出:
#   Q = α + β·log(P)  →  dQ/d(log P) = β  →  dQ/dP = β/P
#   弾力性 E = dQ/dP · P/Q = β/Q
#   平均点での近似: E ≈ β / mean(Q)
mean_demand = df["demand"].mean()
ols_elasticity = model_ols.params["log_price"] / mean_demand

print(f"OLS から計算した弾力性（平均点での近似）: {ols_elasticity:.4f}")

# ──────────────────────────────────────────────
# 3.4 結果の可視化
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- 左: 価格 vs 需要量（原スケール）
prices_range = np.linspace(100, 500, 200)
log_prices_range = np.log(prices_range)

pred_glm = np.exp(model_glm.params["Intercept"] + model_glm.params["log_price"] * log_prices_range)
pred_ols = model_ols.params["Intercept"] + model_ols.params["log_price"] * log_prices_range

axes[0].scatter(df["price"], df["demand"], alpha=0.2, s=8, color="steelblue", label="観測値")
axes[0].plot(prices_range, pred_glm, "r-", lw=2, label=f"GLM (E={elasticity_est:.2f})")
axes[0].plot(prices_range, pred_ols, "g--", lw=2, label="OLS")
axes[0].set_xlabel("価格（円）")
axes[0].set_ylabel("需要量")
axes[0].set_title("価格 vs 需要量")
axes[0].legend()

# --- 右: log-log スケール（直線になるはず）
axes[1].scatter(np.log(df["price"]), np.log(df["demand"]), alpha=0.2, s=8, color="steelblue", label="観測値")
axes[1].plot(log_prices_range, np.log(pred_glm), "r-", lw=2, label="GLM (log-log)")
axes[1].set_xlabel("log(価格)")
axes[1].set_ylabel("log(需要量)")
axes[1].set_title("log(価格) vs log(需要量)")
axes[1].legend()

plt.tight_layout()
plt.savefig("price-elasticity-log-link/price_elasticity.png", dpi=150)
# Agg バックエンド（非インタラクティブ）のため plt.show() は省略

# ──────────────────────────────────────────────
# 3.5 過分散の確認とNegative Binomialへの切り替え
# ──────────────────────────────────────────────

# 過分散のチェック: Pearson χ² / 残差自由度 を確認する
# Poisson が適合していれば ≈ 1、大きければ過分散
pearson_chi2 = model_glm.pearson_chi2
df_resid = model_glm.df_resid
print(f"Pearson χ²: {pearson_chi2:.1f}")
print(f"残差自由度: {df_resid}")
print(f"Pearson χ² / df（分散比）: {pearson_chi2 / df_resid:.2f}")
# ≈ 1 → Poisson が適合; >> 1 → 過分散（Negative Binomial を検討）

# 過分散がある場合は Negative Binomial を使う
model_nb = smf.glm(
    formula="demand ~ log_price",
    data=df,
    family=sm.families.NegativeBinomial(
        alpha=1.0,  # 過分散パラメータ（初期値; 実データでは推定値を参照）
        link=sm.families.links.Log(),
    ),
).fit()

print(f"\nNegative Binomial 推定弾力性: {model_nb.params['log_price']:.4f}")

# ──────────────────────────────────────────────
# 4.1 複数変数への拡張（ダミーデータ付き）
# ──────────────────────────────────────────────

# 実務で使う変数（プロモーション・月）を追加した拡張データを準備する
np.random.seed(28)
df_full = df.copy()
df_full["promotion"] = np.random.randint(0, 2, n)  # 0: 通常, 1: プロモーション中
df_full["month"] = np.random.randint(1, 13, n)  # 1〜12月

# 価格 + プロモーション + 季節性を含むモデル
model_full = smf.glm(
    formula="demand ~ log_price + promotion + C(month)",
    data=df_full,
    family=sm.families.Poisson(link=sm.families.links.Log()),
).fit()

# log_price の係数が「他の変数を制御した上での」価格弾力性
print(f"調整済み価格弾力性: {model_full.params['log_price']:.4f}")

# ──────────────────────────────────────────────
# 4.3 弾力性を使った最適価格の試算
# ──────────────────────────────────────────────


def revenue_simulation(
    price_range: np.ndarray,
    baseline_price: float,
    baseline_demand: float,
    elasticity: float,
) -> pd.DataFrame:
    """価格シミュレーションで収益を計算する。

    Args:
        price_range: シミュレーションする価格の配列（円）
        baseline_price: 基準価格（円）
        baseline_demand: 基準価格での需要量（個）
        elasticity: 価格弾力性（通常は負の値）

    Returns:
        price, demand, revenue の列を持つDataFrame
    """
    results = []
    for p in price_range:
        # 定弾力性モデルによる需要予測
        demand_pred = baseline_demand * (p / baseline_price) ** elasticity
        revenue = p * demand_pred
        results.append({"price": p, "demand": demand_pred, "revenue": revenue})
    return pd.DataFrame(results)


sim = revenue_simulation(
    price_range=np.arange(100, 600, 10),
    baseline_price=100,  # 基準価格（データ生成時の baseline_price に合わせる）
    baseline_demand=baseline_demand,
    elasticity=elasticity_est,
)

optimal = sim.loc[sim["revenue"].idxmax()]
print(f"収益最大化価格: {optimal['price']:.0f}円")
print(f"そのときの推定需要: {optimal['demand']:.0f}個")
print(f"そのときの推定収益: {optimal['revenue']:.0f}円")
