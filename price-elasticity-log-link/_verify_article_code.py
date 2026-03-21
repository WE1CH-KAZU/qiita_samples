"""記事掲載コードの検証スクリプト。

price-elasticity-log-link.md に掲載されているPythonコードが
記事の主張と合致しているかを検証する。

検証項目:
1. データ生成：baseline_demand=1000, price=100-500 のパラメータが適切か
2. GLM推定：真の弾力性-1.5を再現できるか
3. OLS弾力性の計算式：正しいか
4. 過分散チェック：分散/平均が適切な値か
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ========================
# 検証1: データ生成の妥当性
# ========================
print("=" * 60)
print("検証1: データ生成の妥当性チェック")
print("=" * 60)

# 各価格点での Poisson の期待値λを確認する
baseline_demand = 1000
true_elasticity = -1.5
price_points = [100, 200, 300, 400, 500]

print("\n各価格点でのPoisson期待値 λ (= baseline * price^elasticity):")
for p in price_points:
    lam = baseline_demand * (p**true_elasticity)
    print(f"  price={p:4d}: λ = {lam:.4f}  → Poisson(λ) のほぼ全サンプルは 0 または 1")

# 記事のコードで実際にデータ生成（シード42）
np.random.seed(42)
n = 500
price = np.random.uniform(100, 500, n)
log_mu = np.log(baseline_demand) + true_elasticity * np.log(price)
demand_raw = np.random.poisson(lam=np.exp(log_mu))
demand_clipped = np.maximum(demand_raw, 1)

print("\n--- 生成されたデータの統計 ---")
print(f"demand（クリップ前）: mean={demand_raw.mean():.4f}, min={demand_raw.min()}, max={demand_raw.max()}")
print(f"demand（クリップ後）: mean={demand_clipped.mean():.4f}, min={demand_clipped.min()}, max={demand_clipped.max()}")
print(f"demand_raw == 0 の割合: {(demand_raw == 0).mean():.1%}  ← クリップでほぼ1に固定される割合")
print(f"demand_clipped == 1 の割合: {(demand_clipped == 1).mean():.1%}")

df = pd.DataFrame(
    {
        "price": price,
        "demand": demand_clipped,
        "log_price": np.log(price),
    }
)

print("\n[判定] ほぼ全需要量が1に固定されているため、価格変動と需要の関係を")
print("       正しくモデリングできない可能性が高い。")

# ========================
# 検証2: GLM推定
# ========================
print("\n" + "=" * 60)
print("検証2: GLM推定（真の弾力性 -1.5 が再現されるか）")
print("=" * 60)

model_glm = smf.glm(
    formula="demand ~ log_price",
    data=df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
).fit(disp=0)

elasticity_est = model_glm.params["log_price"]
ci = model_glm.conf_int().loc["log_price"]

print(f"\n推定弾力性:     {elasticity_est:.4f}  （記事の出力例: -1.4973）")
print(f"95%信頼区間:    [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"真の弾力性:     {true_elasticity}")
is_ok = abs(elasticity_est - true_elasticity) < 0.1
result_msg = "✓ YES" if is_ok else "✗ NO（出力例と一致しない可能性）"
print(f"記事の主張通りか: {result_msg}")

# ========================
# 検証3: OLS弾力性の計算式
# ========================
print("\n" + "=" * 60)
print("検証3: OLS弾力性計算式の正確性チェック")
print("=" * 60)

model_ols = smf.ols(formula="demand ~ log_price", data=df).fit()
mean_price = df["price"].mean()
mean_demand = df["demand"].mean()
beta_log_price = model_ols.params["log_price"]

print(f"\nOLS係数 β (log_price): {beta_log_price:.4f}")
print(f"mean_price: {mean_price:.2f}")
print(f"mean_demand: {mean_demand:.4f}")

# 記事の計算式
ols_elasticity_article = beta_log_price / mean_demand * mean_price
# 正しい計算式（demand ~ log_price モデルの場合）
ols_elasticity_correct = beta_log_price / mean_demand

print(f"\n記事の式 (β * P̄ / Q̄):   {ols_elasticity_article:.4f}")
print(f"正しい式 (β / Q̄):         {ols_elasticity_correct:.4f}")
print(f"誤差倍率:                  {mean_price:.1f}倍（mean_price 分だけずれている）")
print("\n[理由] モデルが demand ~ log_price なので β = dQ/d(log P) = P * dQ/dP")
print("       弾力性 E = dQ/dP * P/Q = β/Q  →  平均点での近似は β/Q̄")
print("       記事の式は demand ~ price（log なし）のモデルに対応しており、")
print("       今回のモデル指定と矛盾している。")

# ========================
# 検証4: 過分散チェック
# ========================
print("\n" + "=" * 60)
print("検証4: 過分散チェック")
print("=" * 60)

variance = df["demand"].var()
mean_val = df["demand"].mean()
ratio = variance / mean_val
print(f"\n平均: {mean_val:.4f}")
print(f"分散: {variance:.4f}")
print(f"分散/平均（過分散比）: {ratio:.2f}")
print("\n[注意] データはPoisson分布で生成しているため、理論上は比 ≈ 1 のはず。")
print("       ただし np.maximum(demand, 1) のクリップにより分布が歪んでいる。")

# ========================
# 参考: 適切なパラメータでの再現
# ========================
print("\n" + "=" * 60)
print("参考: 適切なbaseline_demandでのデータ生成")
print("=" * 60)

# mean_price ≈ 300 で mean_demand ≈ 500 になるbaseline_demand
target_mean_demand = 500
target_price = 300
baseline_demand_fixed = target_mean_demand * (target_price**-true_elasticity)
print(f"\n価格300円で平均需要500になるbaseline_demand: {baseline_demand_fixed:.0f}")

np.random.seed(28)  # コーディングルールに従いシード28
price_ok = np.random.uniform(100, 500, n)
log_mu_ok = np.log(baseline_demand_fixed) + true_elasticity * np.log(price_ok)
demand_ok = np.random.poisson(lam=np.exp(log_mu_ok))
demand_ok = np.maximum(demand_ok, 1)

df_ok = pd.DataFrame(
    {
        "price": price_ok,
        "demand": demand_ok,
        "log_price": np.log(price_ok),
    }
)
print(f"demand統計: mean={demand_ok.mean():.1f}, min={demand_ok.min()}, max={demand_ok.max()}")
print(f"demand == 0（クリップ前）の割合: {(np.random.poisson(lam=np.exp(log_mu_ok)) == 0).mean():.1%}")

model_glm_ok = smf.glm(
    formula="demand ~ log_price",
    data=df_ok,
    family=sm.families.Poisson(link=sm.families.links.Log()),
).fit(disp=0)
elasticity_ok = model_glm_ok.params["log_price"]
ci_ok = model_glm_ok.conf_int().loc["log_price"]
print(f"\n修正後の推定弾力性: {elasticity_ok:.4f}")
print(f"95%信頼区間: [{ci_ok[0]:.4f}, {ci_ok[1]:.4f}]")
print(f"真の弾力性: {true_elasticity}")
print(f"主張通りに弾力性を回復できているか: {'✓ YES' if abs(elasticity_ok - true_elasticity) < 0.05 else '✗ NO'}")
