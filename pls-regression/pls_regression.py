"""PLS回帰モック解析スクリプト。

間接部門のKPI分析を想定し、多重共線性・少サンプル環境でのPLS回帰の実装・検証を行う。
PLS / PCR / Ridge / Lasso の4手法を交差検証で比較し、意思決定に繋がる係数解釈を示す。
乱数シードは 28 に固定して再現性を担保する。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

# スクリプトの場所を基準に figs/ ディレクトリを解決する
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

RANDOM_STATE = 28

# FutureWarning を無視
warnings.simplefilter(action="ignore", category=FutureWarning)
# RuntimeWarning を無視
warnings.simplefilter(action="ignore", category=RuntimeWarning)


# ── データ生成 ───────────────────────────────────────────────────────────────


def make_mock_indirect_kpi_data(
    n: int = 36,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """間接部門を想定したKPIのモックデータを生成する。

    観測KPIに多重共線性が生じるよう、3つの潜在要因（業務の「束」）からデータを生成する。

    - digital: デジタル施策系
    - event_sales: イベント／ウェビナー／営業訪問系
    - price_cond: 価格・競合圧力束（成約率に負に効く想定）

    Args:
        n: 観測数（例：月次データの月数）。正の整数。
        seed: 乱数シード。再現性確保のために使用する。

    Returns:
        以下のタプルを返す。
            - X: 観測されたKPIからなる特徴量行列（形状: [n, p]）。
            - y: 成約率を表す目的変数（0〜1の範囲）。

    Raises:
        ValueError: n が正の整数でない場合。
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    rng = np.random.default_rng(seed)

    def _clip01(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    # 潜在要因（業務の"束"）
    digital = rng.normal(0, 1, n)
    event_sales = rng.normal(0, 1, n)
    price_cond = rng.normal(0, 1, n)

    # 緩い季節性（全体に少し相関を作る）
    t = np.arange(n)
    season = np.sin(2 * np.pi * t / 12.0)

    X = pd.DataFrame(
        {
            # digital束 16変数
            "mail_sends": np.clip(50 + 10 * digital + 3 * season + rng.normal(0, 3, n), 0, None),
            "mail_delivered": np.clip(48 + 9.5 * digital + 3 * season + rng.normal(0, 3, n), 0, None),
            "open_rate": _clip01(0.25 + 0.05 * digital + 0.01 * season + rng.normal(0, 0.02, n)),
            "click_rate": _clip01(0.04 + 0.02 * digital + 0.005 * season + rng.normal(0, 0.01, n)),
            "web_visits": np.clip(200 + 40 * digital + 10 * season + rng.normal(0, 15, n), 0, None),
            "unique_visitors": np.clip(140 + 30 * digital + 8 * season + rng.normal(0, 12, n), 0, None),
            "content_views": np.clip(500 + 90 * digital + 20 * season + rng.normal(0, 30, n), 0, None),
            "whitepaper_dl": np.clip(20 + 6 * digital + 2 * season + rng.normal(0, 2, n), 0, None),
            "avg_session_time_sec": np.clip(120 + 25 * digital + 5 * season + rng.normal(0, 10, n), 10, None),
            "bounce_rate": _clip01(0.55 - 0.08 * digital - 0.01 * season + rng.normal(0, 0.03, n)),
            "seo_rank_score": np.clip(60 + 8 * digital + 2 * season + rng.normal(0, 3, n), 0, None),
            "paid_search_impressions": np.clip(4000 + 700 * digital + 150 * season + rng.normal(0, 250, n), 0, None),
            "paid_search_clicks": np.clip(120 + 30 * digital + 10 * season + rng.normal(0, 10, n), 0, None),
            "cpc_yen": np.clip(180 - 20 * digital + rng.normal(0, 8, n), 30, None),  # 品質改善でCPC低下
            "retargeting_impressions": np.clip(2500 + 500 * digital + 120 * season + rng.normal(0, 200, n), 0, None),
            "mql_count": np.clip(35 + 7.5 * digital + 2 * season + rng.normal(0, 3, n), 0, None),
            # event_sales束 16変数
            "event_registrations": np.clip(45 + 10 * event_sales + 4 * season + rng.normal(0, 4, n), 0, None),
            "event_attend": np.clip(30 + 8 * event_sales + 3 * season + rng.normal(0, 3, n), 0, None),
            "webinar_registrations": np.clip(40 + 9 * event_sales + 4 * season + rng.normal(0, 4, n), 0, None),
            "webinar_join": np.clip(25 + 7 * event_sales + 3 * season + rng.normal(0, 3, n), 0, None),
            "webinar_questions": np.clip(12 + 3.5 * event_sales + rng.normal(0, 2, n), 0, None),
            "sales_visits": np.clip(40 + 9 * event_sales + 2 * season + rng.normal(0, 4, n), 0, None),
            "sales_calls": np.clip(80 + 15 * event_sales + 3 * season + rng.normal(0, 7, n), 0, None),
            "demo_requests": np.clip(18 + 5.5 * event_sales + 1 * season + rng.normal(0, 2.5, n), 0, None),
            "tech_inquiry": np.clip(10 + 4 * event_sales + rng.normal(0, 2, n), 0, None),
            "sample_shipments": np.clip(8 + 2.8 * event_sales + rng.normal(0, 1.5, n), 0, None),
            "proposal_sent": np.clip(16 + 4.5 * event_sales + rng.normal(0, 2, n), 0, None),
            "proposal_value_million_yen": np.clip(22 + 6.0 * event_sales + rng.normal(0, 4, n), 0, None),
            "followup_emails": np.clip(60 + 12 * event_sales + rng.normal(0, 6, n), 0, None),
            "meeting_minutes": np.clip(900 + 180 * event_sales + rng.normal(0, 120, n), 0, None),
            "partner_referrals": np.clip(6 + 2.0 * event_sales + rng.normal(0, 1.2, n), 0, None),
            "mql_to_sql_rate": _clip01(0.25 + 0.06 * event_sales + rng.normal(0, 0.03, n)),
            # price_cond束（負に効く想定）13変数
            "price_up_flag": (price_cond > 0.3).astype(int),
            "discount_rate": _clip01(0.08 + 0.03 * price_cond + rng.normal(0, 0.01, n)),
            "competitor_pressure": np.clip(50 + 12 * price_cond + 2 * season + rng.normal(0, 4, n), 0, None),
            "competitor_price_index": np.clip(100 + 6 * price_cond + rng.normal(0, 2.5, n), 70, None),
            "bid_lost_rate": _clip01(0.35 + 0.07 * price_cond + rng.normal(0, 0.03, n)),
            "customer_budget_tightness": np.clip(50 + 10 * price_cond + rng.normal(0, 5, n), 0, None),
            "fx_volatility_index": np.clip(15 + 3.5 * price_cond + rng.normal(0, 1.8, n), 0, None),
            "raw_material_cost_index": np.clip(100 + 7 * price_cond + rng.normal(0, 3, n), 70, None),
            "margin_pressure_score": np.clip(40 + 11 * price_cond + rng.normal(0, 4, n), 0, None),
            "procurement_lead_time_days": np.clip(30 + 6 * price_cond + rng.normal(0, 3, n), 5, None),
            "delivery_delay_flag": (price_cond + rng.normal(0, 0.6, n) > 0.8).astype(int),
            "stockout_risk_score": np.clip(35 + 9 * price_cond + rng.normal(0, 4, n), 0, None),
            "price_sensitivity_score": np.clip(50 + 10 * price_cond + rng.normal(0, 5, n), 0, None),
        }
    )
    # ロジスティック変換で目的変数を0〜1に収める
    y_latent = 0.9 * digital + 1.1 * event_sales - 1.2 * price_cond + rng.normal(0, 0.7, n)
    y = 1 / (1 + np.exp(-y_latent))

    return X, pd.Series(y, name="deal_conversion_rate")


# ── ユーティリティ関数 ────────────────────────────────────────────────────────


def rmse(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """RMSE（平方平均二乗誤差の平方根）を計算する。

    Args:
        y_true: 実測値。
        y_pred: 予測値。y_true と同じ順序・長さである必要がある。

    Returns:
        RMSE を表す浮動小数点数。
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def aic_bic_from_rss(n: int, rss: float, k: int) -> tuple[float, float]:
    """RSS から AIC と BIC を計算する。

    Args:
        n: 観測数。
        rss: 残差平方和。
        k: モデルのパラメータ数（複雑度の代理指標）。

    Returns:
        (AIC, BIC) のタプル。

    Raises:
        ValueError: n <= 0、k <= 0、または rss <= 0 の場合。
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if rss <= 0:
        raise ValueError("rss must be positive to compute log(rss / n).")

    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return float(aic), float(bic)


def eval_components(
    kind: Literal["PLS", "PCR"],
    X: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
    max_k: int = 10,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], int]:
    """PLS または PCR の成分数を変化させて性能を評価する。

    Args:
        kind: "PLS" または "PCR"。
        X: 特徴量行列。
        y: 目的変数。
        cv: 交差検証の分割方法。
        max_k: 試行する最大成分数。

    Returns:
        (df, preds, best_k) のタプル。
            - df: 成分数ごとの評価結果（列: ["k", "cv_rmse", "AIC", "BIC"]）。
            - preds: 成分数 k をキー、CV予測値を値とする辞書。
            - best_k: CV RMSE が最小となる成分数。

    Raises:
        ValueError: kind が不正、または max_k < 1 の場合。
    """
    if max_k < 1:
        raise ValueError("max_k must be >= 1.")
    if kind not in ("PLS", "PCR"):
        raise ValueError('kind must be either "PLS" or "PCR".')

    rows: list[tuple[int, float, float, float]] = []
    preds: dict[int, np.ndarray] = {}

    for k in range(1, max_k + 1):
        if kind == "PLS":
            model: Pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("m", PLSRegression(n_components=k)),
                ]
            )
        else:  # PCR
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=k)),
                    ("m", LinearRegression()),
                ]
            )
        y_pred = cross_val_predict(model, X, y, cv=cv)
        preds[k] = y_pred
        cv_rmse = rmse(y, y_pred)

        model.fit(X, y)
        y_fit = model.predict(X).ravel()

        rss = float(np.sum((y.to_numpy() - y_fit) ** 2))
        # 成分数 k に切片相当 (+1) を加えて複雑度の代理指標とする
        aic, bic = aic_bic_from_rss(len(y), rss, k + 1)

        rows.append((k, cv_rmse, aic, bic))

    df = pd.DataFrame(rows, columns=["k", "cv_rmse", "AIC", "BIC"])
    best_k = int(df.sort_values("cv_rmse").iloc[0]["k"])
    return df, preds, best_k


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """多重共線性の評価として VIF を計算する。

    Args:
        X: 特徴量行列。

    Returns:
        VIF値を含むDataFrame（列: ["feature", "VIF"]）。VIF降順でソート済み。
    """
    X_with_const = add_constant(X, prepend=False)

    vif_values = []
    feature_names = []
    for i in range(len(X.columns)):
        vif = variance_inflation_factor(X_with_const.values, i)
        vif_values.append(vif)
        feature_names.append(X.columns[i])

    return pd.DataFrame({"feature": feature_names, "VIF": vif_values}).sort_values("VIF", ascending=False)


# ── メイン処理 ────────────────────────────────────────────────────────────────

# データ生成（36行 × 45列）
X, y = make_mock_indirect_kpi_data()
print(X.shape)
print(y.head())

# VIF確認（多重共線性が強いことを示す）
vif_result = calculate_vif(X)
print(vif_result.head(10))

# 交差検証の設定（月次データを4半期単位で分割）
n_split = len(X) // 4  # 四半期単位の分割数
cv = KFold(n_splits=n_split, shuffle=False)

# PLS / PCR: 成分数探索
pls_df, pls_pred, best_pls = eval_components("PLS", X, y, cv=cv)
pcr_df, pcr_pred, best_pcr = eval_components("PCR", X, y, cv=cv)

# Ridge: 正則化強度探索
ridge_rows: list[tuple[float, float]] = []
for alpha in [0.1, 1, 10, 100, 1000]:
    model = Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=alpha))])
    y_pred = cross_val_predict(model, X, y, cv=cv)
    ridge_rows.append((float(alpha), rmse(y, y_pred)))
ridge_df = pd.DataFrame(ridge_rows, columns=["alpha", "cv_rmse"])
best_alpha = float(ridge_df.sort_values("cv_rmse").iloc[0]["alpha"])
ridge_pred = cross_val_predict(
    Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=best_alpha))]), X, y, cv=cv
)

# Lasso: 正則化強度探索
lasso_rows: list[tuple[float, float]] = []
max_iter = 50000  # 収束しないことがあるため max_iter を増やす
for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=alpha, max_iter=max_iter))])
    y_pred = cross_val_predict(model, X, y, cv=cv)
    lasso_rows.append((float(alpha), rmse(y, y_pred)))
lasso_df = pd.DataFrame(lasso_rows, columns=["alpha", "cv_rmse"])
best_lasso_alpha = float(lasso_df.sort_values("cv_rmse").iloc[0]["alpha"])
lasso_pred = cross_val_predict(
    Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=best_lasso_alpha, max_iter=max_iter))]), X, y, cv=cv
)

print("PLS\n", pls_df, "\nbest_k:", best_pls)
print("\nPCR\n", pcr_df, "\nbest_k:", best_pcr)
print("\nRidge\n", ridge_df, "\nbest_alpha:", best_alpha)
print("\nLasso\n", lasso_df, "\nbest_alpha:", best_lasso_alpha)

# 最適パラメータでモデルを再学習し係数を抽出
pls_best = Pipeline([("scaler", StandardScaler()), ("m", PLSRegression(n_components=best_pls))]).fit(X, y)
pcr_best = Pipeline(
    [("scaler", StandardScaler()), ("pca", PCA(n_components=best_pcr)), ("m", LinearRegression())]
).fit(X, y)
ridge_best = Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=best_alpha))]).fit(X, y)
lasso_best = Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=best_lasso_alpha, max_iter=max_iter))]).fit(X, y)

pls_coef = pd.Series(pls_best.named_steps["m"].coef_.ravel(), index=X.columns).sort_values()
pcr_coef = pd.Series(
    pcr_best.named_steps["m"].coef_.ravel(), index=[f"PC{i}" for i in range(best_pcr)]
).sort_values()
ridge_coef = pd.Series(ridge_best.named_steps["m"].coef_.ravel(), index=X.columns).sort_values()
lasso_coef = pd.Series(lasso_best.named_steps["m"].coef_.ravel(), index=X.columns).sort_values()

# PLS重み（ヒートマップ）
pls = pls_best.named_steps["m"]
weights = pd.DataFrame(
    pls.x_weights_, index=X.columns, columns=[f"comp{i}" for i in range(best_pls)]
)
plt.figure(figsize=(10, max(8, int(len(weights.index) * 0.2))))
sns.heatmap(weights, annot=True, fmt=".3f", cmap="RdBu_r", center=0, cbar_kws={"label": "Weight"})
plt.title(f"PLS x_weights_ (n_components={best_pls})\nHow original features are assigned to latent factors")
plt.xlabel("Latent Factor")
plt.ylabel("Original Features")
plt.tight_layout()
plt.savefig(FIGS_DIR / "pls_weights_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# PLS負荷量散布図
digital_features = [
    "mail_sends", "mail_delivered", "open_rate", "click_rate", "web_visits",
    "unique_visitors", "content_views", "whitepaper_dl", "avg_session_time_sec",
    "bounce_rate", "seo_rank_score", "paid_search_impressions", "paid_search_clicks",
    "cpc_yen", "retargeting_impressions", "mql_count",
]
event_sales_features = [
    "event_registrations", "event_attend", "webinar_registrations", "webinar_join",
    "webinar_questions", "sales_visits", "sales_calls", "demo_requests", "tech_inquiry",
    "sample_shipments", "proposal_sent", "proposal_value_million_yen", "followup_emails",
    "meeting_minutes", "partner_referrals", "mql_to_sql_rate",
]
price_cond_features = [
    "price_up_flag", "discount_rate", "competitor_pressure", "competitor_price_index",
    "bid_lost_rate", "customer_budget_tightness", "fx_volatility_index",
    "raw_material_cost_index", "margin_pressure_score", "procurement_lead_time_days",
    "delivery_delay_flag", "stockout_risk_score", "price_sensitivity_score",
]
kpi_groups = {"Marketing": digital_features, "Event_sales": event_sales_features, "Price_conditions": price_cond_features}
feature2group = {feat: group for group, feats in kpi_groups.items() for feat in feats}

loadings = pd.DataFrame(pls.x_loadings_, index=X.columns, columns=[f"comp{i}" for i in range(best_pls)])
feature_groups = X.columns.map(feature2group)
groups = feature_groups.unique()
markers = ["o", "s", "^"]
colors = sns.color_palette("Set2", n_colors=len(groups))

plt.figure(figsize=(10, 8))
for i, group in enumerate(groups):
    idx = feature_groups == group
    plt.scatter(
        loadings.loc[idx, "comp0"], loadings.loc[idx, "comp1"],
        label=group, marker=markers[i % len(markers)], s=120, c=[colors[i]], edgecolors="k",
    )
    for feat in loadings.index[idx]:
        plt.text(loadings.loc[feat, "comp0"], loadings.loc[feat, "comp1"], feat, fontsize=8, alpha=0.7)
plt.axhline(0, color="grey", linestyle="--", linewidth=1)
plt.axvline(0, color="grey", linestyle="--", linewidth=1)
plt.xlabel("PLS Loading comp0")
plt.ylabel("PLS Loading comp1")
plt.title("PLS Feature Loadings (comp0 vs comp1)\nGrouped by Latent KPI Factor")
plt.legend(title="Latent Factor Group", loc="best")
plt.tight_layout()
plt.savefig(FIGS_DIR / "pls_loadings_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# PCR主成分ヒートマップ
pca = pcr_best.named_steps["pca"]
pcr_components = pd.DataFrame(
    pca.components_.T, index=X.columns, columns=[f"PC{i}" for i in range(best_pcr)]
)
plt.figure(figsize=(10, max(8, int(len(weights.index) * 0.2))))
sns.heatmap(pcr_components, annot=True, fmt=".3f", cmap="RdBu_r", center=0, cbar_kws={"label": "Component Weight"})
plt.title(f"PCR PCA components_ (n_components={best_pcr})\nHow original features are compressed into principal components")
plt.xlabel("Principal Component")
plt.ylabel("Original Features")
plt.tight_layout()
plt.savefig(FIGS_DIR / "pcr_components_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# 予測値 vs 実測値
plt.figure()
plt.scatter(y, pls_pred[best_pls], label=f"PLS k={best_pls}", color="#222222")
plt.scatter(y, pcr_pred[best_pcr], label=f"PCR k={best_pcr}", color="#555555")
plt.scatter(y, ridge_pred, label=f"Ridge alpha={best_alpha}", color="white", edgecolor="#000000", alpha=0.75, marker="^", linewidths=1.0)
plt.scatter(y, lasso_pred, label=f"Lasso alpha={best_lasso_alpha}", color="white", edgecolor="#000000", alpha=0.75, marker="s", linewidths=1.0)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Actual y")
plt.ylabel("CV Predicted y")
plt.title("Actual vs CV Predicted")
plt.legend()
plt.savefig(FIGS_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

# 各モデルの回帰係数
for coef, label, color in [
    (pls_coef, f"PLS (k={best_pls})", "gray"),
    (ridge_coef, f"Ridge (alpha={best_alpha:g})", "steelblue"),
    (lasso_coef, f"Lasso (alpha={best_lasso_alpha:g})", "orange"),
]:
    plt.figure(figsize=(10, max(8, int(len(coef) * 0.2))))
    plt.barh(coef.index, coef.values, color=color)
    plt.xlabel(f"{label} coef (scaled X)")
    plt.title(f"{label} coefficients")
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{label.split()[0].lower()}_coef.png", dpi=150, bbox_inches="tight")
    plt.show()

# Lasso のスパース性を確認
n_nonzero_lasso = (lasso_coef != 0).sum()
n_total = len(lasso_coef)
print(f"Lasso regression: {n_nonzero_lasso}/{n_total} features are used (sparsity)")
