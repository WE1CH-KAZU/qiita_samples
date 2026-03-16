"""動的価格弾力性モデルの動作検証スクリプト。

記事「動的価格弾力性モデル：時変する価格感応度をPythonで推定する」に掲載する
ローリングウィンドウGLM・状態空間モデル（カルマンフィルタ）のコードを検証し、
グラフを figs/ に保存する。
"""

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.mlemodel import MLEModel

# 日本語フォント設定（macOS）
matplotlib.rcParams["font.family"] = "Hiragino Sans"

# グラフ出力先ディレクトリ
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


# =============================================================================
# 3.1 シミュレーションデータの生成
# =============================================================================

np.random.seed(28)  # シード値28に固定して再現性を確保

# 週次データ 3年分（156週）
n_weeks = 156
dates = pd.date_range(start="2021-01-04", periods=n_weeks, freq="W-MON")

# 真の弾力性: ドリフト付きランダムウォークで生成
# 大局的には競合環境悪化により緩やかに弾力的な方向へドリフトしつつ、
# 週次で不規則な変動を伴う（より現実的なシナリオ）
BETA0 = -1.0  # 初期弾力性
DRIFT = -0.001  # 週次ドリフト（3年間で約 -0.16 の変化）
SIGMA_STATE = 0.05  # 週次の弾力性ノイズ（状態ノイズ）

state_noise = np.random.normal(DRIFT, SIGMA_STATE, n_weeks - 1)
true_elasticity = np.concatenate([[BETA0], BETA0 + np.cumsum(state_noise)])

# 価格（週ごとにランダムに変動、200〜400円）
price = np.random.uniform(200, 400, n_weeks)

# 対数価格の中心化: 基準価格 300円（範囲の中央値）からの対数乖離に変換
# → log_price = 0 が「価格が基準価格と等しい」状態に対応し、
#   alpha が「基準価格での期待需要量」を直接表す
PRICE_BASE = 300.0
log_price = np.log(price / PRICE_BASE)  # ≈ -0.4 〜 +0.3

# 基準価格 300円での期待需要量を 1000 個に設定
# NOTE: 原式 alpha = log(5000) と log_price = log(price) の組み合わせは
#       elasticity=-2.5 時に log_demand ≈ -5.75 → demand ≈ 0 → クリップ問題が発生するため修正
alpha = np.log(1000)

# 対数需要量: log(E[Q_t]) = alpha + beta_t * log(P_t / P_base)
log_demand_true = alpha + true_elasticity * log_price
# 観測ノイズ（σ=0.02）: 教科書的なクリーンデータの設定
log_demand_obs = log_demand_true + np.random.normal(0, 0.02, n_weeks)
demand = np.round(np.exp(log_demand_obs)).astype(int)
demand = np.maximum(demand, 1)

df = pd.DataFrame(
    {
        "date": dates,
        "price": price,
        "demand": demand,
        "log_price": log_price,
        "log_demand": np.log(demand),
        "true_elasticity": true_elasticity,
    }
).set_index("date")

print(df.head())
print(f"\n弾力性の変化: {true_elasticity[0]:.1f} → {true_elasticity[-1]:.1f}")


# =============================================================================
# 3.2 ローリングウィンドウGLM
# =============================================================================


def rolling_glm_elasticity(df: pd.DataFrame, window_size: int = 26) -> pd.DataFrame:
    """ローリングウィンドウGLMで時変弾力性を推定する。

    Args:
        df: log_price, demand 列を含む DataFrame
        window_size: ウィンドウ幅（週数）

    Returns:
        各時点の弾力性推定値と信頼区間を格納した DataFrame
    """
    results = []

    for i in range(window_size, len(df) + 1):
        window = df.iloc[i - window_size : i]

        model = smf.glm(
            formula="demand ~ log_price",
            data=window,
            family=sm.families.Poisson(  # 誤差分布: Poisson分布
                link=sm.families.links.Log()  # リンク関数: log
            ),
        ).fit(disp=False)

        ci = model.conf_int().loc["log_price"]
        results.append(
            {
                "date": df.index[i - 1],
                "elasticity": model.params["log_price"],
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }
        )

    return pd.DataFrame(results).set_index("date")


# ウィンドウサイズ 26週（半年）と 52週（1年）で比較
rolling_26 = rolling_glm_elasticity(df, window_size=26)
rolling_52 = rolling_glm_elasticity(df, window_size=52)

print(f"\n推定期間: {rolling_26.index[0].date()} 〜 {rolling_26.index[-1].date()}")
print(rolling_26.tail())


# =============================================================================
# 3.3 状態空間モデル（カルマンフィルタ）
# =============================================================================


class TimeVaryingElasticity(MLEModel):
    """時変係数モデル（状態空間表現）。

    観測方程式: y_t = α_t + β_t * x_t + ε_t
    状態方程式: [α_t, β_t]' = [α_{t-1}, β_{t-1}]' + η_t

    状態ベクトル: [α_t, β_t]
    """

    def __init__(self, endog: np.ndarray, exog: np.ndarray) -> None:
        # 状態の数: 2（切片 α_t + 弾力性 β_t）
        super().__init__(endog, k_states=2, k_posdef=2)
        self.exog = np.array(exog)

        # 時変観測行列を時系列長 T 分で事前確保（形状: 1 × 2 × T）
        # NOTE: デフォルトは (1, 2, 1) のため t > 0 で IndexError になる
        self["design"] = np.zeros((1, 2, self.nobs))

        # 状態遷移行列（ランダムウォーク: I_2）
        self["transition"] = np.eye(2)

        # 状態誤差の選択行列
        self["selection"] = np.eye(2)

        # 初期状態の共分散（弱情報事前分布）
        self.initialize_approximate_diffuse()

    @property
    def param_names(self) -> list[str]:
        return ["sigma2_obs", "sigma2_alpha", "sigma2_beta"]

    @property
    def start_params(self) -> list[float]:
        return [0.01, 0.01, 0.01]

    def transform_params(self, params: np.ndarray) -> np.ndarray:
        # パラメータを正値に制約（分散は非負）
        return params**2

    def untransform_params(self, params: np.ndarray) -> np.ndarray:
        return params**0.5

    def update(self, params: np.ndarray, **kwargs) -> np.ndarray:
        params = super().update(params, **kwargs)

        sigma2_obs, sigma2_alpha, sigma2_beta = params

        # 観測行列: Z_t = [1, x_t]（時変）
        # （各時点で log_price を設定）
        for t in range(self.nobs):
            self["design", 0, :, t] = [1.0, self.exog[t]]

        # 観測誤差の分散
        self["obs_cov", 0, 0] = sigma2_obs

        # 状態誤差の共分散（対角行列）
        self["state_cov"] = np.diag([sigma2_alpha, sigma2_beta])

        return params  # NOTE: 記事のコードに return 文が欠落していたため追加


# モデルの推定
y = df["log_demand"].values
x = df["log_price"].values

ssm = TimeVaryingElasticity(endog=y, exog=x)
result = ssm.fit(method="lbfgs", maxiter=500, disp=False)

print(result.summary())

# スムージング済み状態（全期間のデータを使った後方平滑化）を取得
# カルマンスムーザー: フィルタ（片側・過去のみ）と異なり、全データを活用して
# 各時点の弾力性を最も精度高く推定する。回顧的分析に適する。
smoothed_states = result.smoothed_state  # shape: (2, T)
smoothed_cov = result.smoothed_state_cov  # shape: (2, 2, T)

elasticity_ssm = pd.Series(
    smoothed_states[1, :],  # β_t の系列
    index=df.index,
    name="elasticity_ssm",
)

# 95%信頼区間（状態の事後標準偏差から計算）
std_beta = np.sqrt(smoothed_cov[1, 1, :])
ci_ssm = pd.DataFrame(
    {
        "elasticity": smoothed_states[1, :],
        "ci_lower": smoothed_states[1, :] - 1.96 * std_beta,
        "ci_upper": smoothed_states[1, :] + 1.96 * std_beta,
    },
    index=df.index,
)


# =============================================================================
# 3.4 結果の可視化と比較
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

# --- 共通の設定 ---
true_series = df["true_elasticity"]

for ax in axes:
    ax.axhline(y=0, color="k", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.plot(true_series, "k-", lw=1.5, label="真の弾力性", alpha=0.6)
    ax.set_ylabel("価格弾力性")
    ax.set_ylim(-3.5, 0.5)
    ax.legend(loc="lower left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# --- ローリングウィンドウ 26週 ---
axes[0].plot(rolling_26["elasticity"], "b-", lw=1.5, label="ローリングGLM (W=26週)")
axes[0].fill_between(
    rolling_26.index,
    rolling_26["ci_lower"],
    rolling_26["ci_upper"],
    alpha=0.2,
    color="blue",
    label="95%信頼区間",
)
axes[0].set_title("ローリングウィンドウGLM（W=26週）", fontsize=11)
axes[0].legend(loc="lower left", fontsize=9)

# --- ローリングウィンドウ 52週 ---
axes[1].plot(rolling_52["elasticity"], "g-", lw=1.5, label="ローリングGLM (W=52週)")
axes[1].fill_between(
    rolling_52.index,
    rolling_52["ci_lower"],
    rolling_52["ci_upper"],
    alpha=0.2,
    color="green",
    label="95%信頼区間",
)
axes[1].set_title("ローリングウィンドウGLM（W=52週）", fontsize=11)
axes[1].legend(loc="lower left", fontsize=9)

# --- 状態空間モデル ---
axes[2].plot(ci_ssm["elasticity"], "r-", lw=1.5, label="状態空間モデル（カルマンフィルタ）")
axes[2].fill_between(
    ci_ssm.index,
    ci_ssm["ci_lower"],
    ci_ssm["ci_upper"],
    alpha=0.2,
    color="red",
    label="95%信頼区間",
)
axes[2].set_title("状態空間モデル（カルマンフィルタ）", fontsize=11)
axes[2].legend(loc="lower left", fontsize=9)
axes[2].set_xlabel("日付")

plt.suptitle("動的価格弾力性の推定比較", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIGS_DIR / "dynamic_elasticity_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nグラフを保存しました: {FIGS_DIR / 'dynamic_elasticity_comparison.png'}")
plt.close()


# =============================================================================
# 3.5 推定精度の定量比較
# =============================================================================

# 各モデルの RMSE（真の弾力性との比較）

# ローリングGLM 26週: 推定開始は26週目以降
common_idx_26 = rolling_26.index
true_26 = df.loc[common_idx_26, "true_elasticity"]
rmse_26 = np.sqrt(np.mean((rolling_26["elasticity"] - true_26) ** 2))

# ローリングGLM 52週
common_idx_52 = rolling_52.index
true_52 = df.loc[common_idx_52, "true_elasticity"]
rmse_52 = np.sqrt(np.mean((rolling_52["elasticity"] - true_52) ** 2))

# 状態空間モデル（全期間）
rmse_ssm = np.sqrt(np.mean((ci_ssm["elasticity"] - df["true_elasticity"]) ** 2))

print("\n=== 推定精度比較（RMSE）===")
print(f"ローリングGLM (W=26週): {rmse_26:.4f}")
print(f"ローリングGLM (W=52週): {rmse_52:.4f}")
print(f"状態空間モデル:          {rmse_ssm:.4f}")


# =============================================================================
# 4.1 クロスバリデーションによるウィンドウサイズ選択
# =============================================================================

window_candidates = [13, 26, 39, 52]  # 13週=約3ヶ月
cv_scores: dict[int, float] = {}

tscv = TimeSeriesSplit(n_splits=5)

for w in window_candidates:
    errors = []
    for train_idx, test_idx in tscv.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        if len(train) < w:
            continue

        model = smf.glm(
            "demand ~ log_price",
            data=train.iloc[-w:],  # 直近 w 週で学習
            family=sm.families.Poisson(link=sm.families.links.Log()),
        ).fit(disp=False)

        # テスト期間の予測誤差（対数需要の RMSE）
        pred_log = model.predict(test)
        err = np.sqrt(np.mean((np.log(test["demand"]) - np.log(pred_log)) ** 2))
        errors.append(err)

    cv_scores[w] = np.mean(errors)
    print(f"W={w:3d}週: CV-RMSE = {cv_scores[w]:.4f}")

best_w = min(cv_scores, key=cv_scores.get)
print(f"\n最適ウィンドウサイズ: {best_w}週")


# =============================================================================
# 4.2 値付け判断のアラート
# =============================================================================

ELASTICITY_THRESHOLD = -1.5  # 弾力性がこれを下回ったら要注意

latest_elasticity = ci_ssm["elasticity"].iloc[-1]
latest_ci_lower = ci_ssm["ci_lower"].iloc[-1]

if latest_ci_lower < ELASTICITY_THRESHOLD:
    print(f"\n⚠️  警告: 推定弾力性 {latest_elasticity:.2f}")
    print(f"   95%CI下限が {ELASTICITY_THRESHOLD} を下回っています。")
    print("   値上げは慎重に検討してください。")
else:
    print(f"\n✅  弾力性 {latest_elasticity:.2f} — 非弾力的な状態を維持しています。")


# =============================================================================
# 4.2 Lerner指数による理論的最適価格
# =============================================================================


def optimal_price_markup(elasticity: float, cost: float) -> tuple[float, float] | None:
    """Lerner指数による理論的最適価格を計算する。

    Lerner条件: (P - MC)/P = -1/E → P* = MC / (1 + 1/E)
    E < -1 の場合のみ有効。

    Args:
        elasticity: 価格弾力性（負の値）
        cost: 原価

    Returns:
        (最適価格, マークアップ率) のタプル。E >= -1 の場合は None
    """
    if elasticity >= -1:
        return None  # 弾力性が -1 以上だと最適価格が存在しない
    optimal_price = cost / (1 + 1 / elasticity)  # E=-2 → cost/0.5 = 2*cost
    markup_rate = (optimal_price - cost) / optimal_price  # = 1/|E|
    return optimal_price, markup_rate


cost = 150  # 原価
e = ci_ssm["elasticity"].iloc[-1]

result_price = optimal_price_markup(e, cost)
if result_price:
    p_opt, markup = result_price
    print(f"\n弾力性: {e:.2f}")
    print(f"理論最適価格: {p_opt:.0f}円（マークアップ率: {markup:.1%}）")


# =============================================================================
# 4.3 ruptures による変化点検出
# =============================================================================

signal = ci_ssm["elasticity"].values.reshape(-1, 1)

# Pelt アルゴリズム: コスト関数 rbf + ペナルティ pen で変化点数を自動決定
algo = rpt.Pelt(model="rbf").fit(signal)
breakpoints = algo.predict(pen=3)

# breakpoints は変化点のインデックスリスト（末尾は系列長を含む）
change_indices = [bp for bp in breakpoints if bp < len(ci_ssm)]
change_dates = ci_ssm.index[change_indices] if change_indices else []

print(f"\n検出された変化点数: {len(change_indices)}")
for d in change_dates:
    e_at_change = ci_ssm.loc[d, "elasticity"]
    print(f"  {d.date()}  弾力性: {e_at_change:.2f}")

print("\n✅ 動作確認完了")
