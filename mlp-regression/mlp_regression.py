"""MLP回帰モック解析スクリプト。

小サンプル・多数説明変数環境でのMLP回帰の実装・検証を行う。
前処理・交差検証・ハイパーパラメータ探索・汎化性能評価・解釈性のための
Permutation Importance を含む。Ridge との比較を通じて非線形モデルの優位性を検証する。
乱数シードは 28 に固定して再現性を担保する。
"""

from __future__ import annotations

import warnings
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# スクリプトの場所を基準に figs/ ディレクトリを解決する
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 28  # 乱数シード（再現性確保）

# FutureWarning を無視
warnings.simplefilter(action="ignore", category=FutureWarning)
# RuntimeWarning を無視
warnings.simplefilter(action="ignore", category=RuntimeWarning)


# ── データ生成 ───────────────────────────────────────────────────────────────


def make_stronger_nonlinear_data(n: int = 200, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.Series]:
    """強い非線形性を持つモックデータを生成する。

    3つの潜在因子（digital, event_sales, price_cond）から説明変数を生成し、
    目的変数に非線形変換（sin / cos / tanh / 3乗など）を組み合わせて
    MLPが線形モデルより優位になるよう設計する。

    Args:
        n: サンプル数。
        seed: 乱数シード。

    Returns:
        以下のタプルを返す。
            - X: 特徴量DataFrame（形状: [n, 40]）。
            - y: 目的変数Series（0〜1に正規化済み）。
    """
    rng = np.random.default_rng(seed)
    digital = rng.normal(0, 1, n)
    event_sales = rng.normal(0, 1, n)
    price_cond = rng.normal(0, 1, n)
    t = np.arange(n)
    season = np.sin(2 * np.pi * t / 12.0)

    # 潜在因子ごとに説明変数を生成（各8変数 × 3因子 + ノイズ16変数 = 40変数）
    X = pd.DataFrame({f"digital_{i}": 50 + 10 * digital + 3 * season + rng.normal(0, 3, n) for i in range(8)})
    X = pd.concat(
        [X, pd.DataFrame({f"event_{i}": 30 + 8 * event_sales + rng.normal(0, 3, n) for i in range(8)})], axis=1
    )
    X = pd.concat(
        [X, pd.DataFrame({f"price_{i}": 5 + 2 * price_cond + rng.normal(0, 1, n) for i in range(8)})], axis=1
    )
    X = pd.concat([X, pd.DataFrame({f"noise_{i}": rng.normal(0, 1, n) for i in range(16)})], axis=1)

    # 強い非線形・相互作用を組み合わせた目的変数
    y_cont = (
        1.0 * np.sin(1.5 * digital) * np.abs(digital)
        + 0.9 * (np.cos(1.2 * event_sales) ** 2) * event_sales
        - 0.6 * price_cond
        + 1.2 * (digital * event_sales) ** 2 / (1 + np.abs(digital * event_sales))
        + 0.7 * np.tanh(1.4 * digital * price_cond)
        + 0.9 * (digital**3) * np.sign(event_sales)
        + 0.02 * season
        + rng.normal(0, 0.02, n)
    )
    # 0〜1 に正規化
    y = (y_cont - y_cont.min()) / (y_cont.max() - y_cont.min())
    return X, pd.Series(y, name="target_mlp")


# ── ユーティリティ関数 ────────────────────────────────────────────────────────


def _rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """RMSE を計算する。

    Args:
        y_true: 実測値。
        y_pred: 予測値。

    Returns:
        float: RMSE。
    """
    return sqrt(mean_squared_error(y_true, y_pred))


def plot_top_importances(res: object, feature_names: pd.Index, model_name: str, top_n: int = 10) -> None:
    """Permutation Importance の上位特徴量を棒グラフで描画する。

    Args:
        res: permutation_importance の戻り値（Bunch）。
        feature_names: 特徴量名のインデックス。
        model_name: モデル名（グラフタイトルに使用）。
        top_n: 上位何特徴量を表示するか。
    """
    importances = res.importances_mean
    idx_sorted = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 4))
    plt.bar(range(top_n), importances[idx_sorted])
    plt.xticks(range(top_n), feature_names[idx_sorted], rotation=45, ha="right")
    plt.title(f"Top {top_n} permutation importances: {model_name}")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"perm_importance_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── メイン処理 ────────────────────────────────────────────────────────────────

# データ生成・分割
X, y = make_stronger_nonlinear_data(n=200)

n_test = 40  # テストサンプル数
X_train, X_test = X.iloc[:-n_test, :], X.iloc[-n_test:, :]
y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

# 交差検証の設定（時系列性を考慮してシャッフルしない）
cv = KFold(n_splits=4, shuffle=False)

print(f"Data shapes: X={X.shape}, y={y.shape}; train={X_train.shape}, test={X_test.shape}")
print(f"Random seed: {RANDOM_STATE}")

# 目的変数の時系列プロット
plt.figure(figsize=(10, 4))
plt.plot(y.to_numpy(dtype=float), marker="o", label="target_mlp")
plt.title("目的変数の時系列")
plt.xlabel("time")
plt.ylabel("normalized value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / "target_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# 説明変数グループ平均の時系列プロット
groups = {
    "digital": [c for c in X.columns if c.startswith("digital_")],
    "event": [c for c in X.columns if c.startswith("event_")],
    "price": [c for c in X.columns if c.startswith("price_")],
    "noise": [c for c in X.columns if c.startswith("noise_")],
}
group_means = {k: X[v].mean(axis=1) for k, v in groups.items()}

plt.figure(figsize=(10, 4))
for name, series in group_means.items():
    plt.plot(series.to_numpy(dtype=float), label=f"{name} mean")
plt.title("Explanation variable time series (group means)")
plt.xlabel("time")
plt.ylabel("value (mean of group)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / "feature_group_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# MLP: solver='lbfgs' は小〜中規模データで安定して収束するため採用
pipeline_mlp = Pipeline(
    [("scaler", StandardScaler()), ("mlp", MLPRegressor(random_state=RANDOM_STATE, solver="lbfgs", max_iter=2000))]
)
param_grid_mlp = {
    "mlp__hidden_layer_sizes": [(10,), (20,), (20, 10), (50,)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-4, 1e-3, 1e-2],
}
search_mlp = GridSearchCV(pipeline_mlp, param_grid_mlp, scoring="neg_mean_squared_error", cv=cv, n_jobs=1)
search_mlp.fit(X_train, y_train)

# Ridge ベースライン
pipeline_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(random_state=RANDOM_STATE))])
param_grid_ridge = {"ridge__alpha": [1e-3, 1e-1, 1.0, 10.0]}
search_ridge = GridSearchCV(pipeline_ridge, param_grid_ridge, scoring="neg_mean_squared_error", cv=cv, n_jobs=1)
search_ridge.fit(X_train, y_train)

# CV / テスト評価
cv_rmse_mlp = np.sqrt(-search_mlp.best_score_)
cv_rmse_ridge = np.sqrt(-search_ridge.best_score_)

mlp_best = search_mlp.best_estimator_
ridge_best = search_ridge.best_estimator_

y_pred_mlp = mlp_best.predict(X_test)
y_pred_ridge = ridge_best.predict(X_test)

rmse_mlp = _rmse(y_test, y_pred_mlp)
rmse_ridge = _rmse(y_test, y_pred_ridge)

print("MLP best params:", search_mlp.best_params_)
print("Ridge best params:", search_ridge.best_params_)
print(f"CV RMSE  MLP: {cv_rmse_mlp:.4f}")
print(f"CV RMSE  Ridge: {cv_rmse_ridge:.4f}")
print(f"Test RMSE MLP: {rmse_mlp:.4f}")
print(f"Test RMSE Ridge: {rmse_ridge:.4f}")

# Permutation Importance
res_mlp = permutation_importance(mlp_best, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, n_jobs=1)
res_ridge = permutation_importance(ridge_best, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, n_jobs=1)

plot_top_importances(res_mlp, X_test.columns, "MLP")
plot_top_importances(res_ridge, X_test.columns, "Ridge")

# 予測 vs 実測（テストセット）
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_mlp, label="MLP", alpha=0.7)
plt.scatter(y_test, y_pred_ridge, label="Ridge", alpha=0.7)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("True target")
plt.ylabel("Predicted")
plt.legend()
plt.title("Predicted vs True (test set)")
plt.tight_layout()
plt.savefig(FIGS_DIR / "predicted_vs_true.png", dpi=150, bbox_inches="tight")
plt.show()

print("Note: MLP used solver='lbfgs' for more stable convergence on this sample size.")
