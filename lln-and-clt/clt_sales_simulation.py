"""売上データへの中心極限定理の適用を示すシミュレーションスクリプト。

個々の受注額が右に歪んだ分布（対数正規分布）に従っていても、
月次の受注合計を多数集めると正規分布に近づくことを可視化する。
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 日本語フォントの設定（macOS）
plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

# 乱数シード固定（再現性確保）
np.random.seed(28)

# --- シミュレーション設定 ---
N_ORDERS_PER_MONTH = 10  # 1ヶ月あたりの受注件数
N_MONTHS = 1000  # シミュレートする月数

# 個々の受注額：対数正規分布（単位：万円）
# 多くは100万円前後、稀に500万円超の大口が入る右歪み分布を想定
# 対数正規分布のパラメータ: 中央値 ≈ exp(mu) = 100万円
MU = np.log(100)  # 対数スケールの平均
SIGMA = 0.8  # 対数スケールの標準偏差（大きいほど右裾が長くなる）

# --- シミュレーション ---
# 個々の受注額を大量生成 shape: (N_MONTHS, N_ORDERS_PER_MONTH)
all_orders = np.random.lognormal(mean=MU, sigma=SIGMA, size=(N_MONTHS, N_ORDERS_PER_MONTH))

# 月次売上合計（各月の受注額を合算）
monthly_totals = all_orders.sum(axis=1)  # shape: (N_MONTHS,)

# 個々の受注額（可視化用にサンプルを取り出す）
individual_orders = all_orders.flatten()

# --- 描画（1行2列のサブプロット）---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ---- 左パネル：個々の受注額の分布（右歪み）----
ax_left = axes[0]
ax_left.hist(
    individual_orders,
    bins=60,
    range=(0, 600),  # 600万円までを表示（外れ値を除外して見やすく）
    density=True,
    color="#4C72B0",
    alpha=0.75,
    edgecolor="white",
)
ax_left.set_title("個々の受注額の分布\n（元データ：右に歪んだ分布）", fontsize=12)
ax_left.set_xlabel("受注額（万円）", fontsize=11)
ax_left.set_ylabel("確率密度", fontsize=11)
ax_left.grid(axis="y", linestyle=":", alpha=0.6)

# ---- 右パネル：月次売上合計の分布（正規分布に近づく）----
ax_right = axes[1]
ax_right.hist(
    monthly_totals,
    bins=40,
    density=True,
    color="#4C72B0",
    alpha=0.75,
    edgecolor="white",
    label="月次売上合計のヒストグラム",
)

# 中心極限定理が予測する正規分布を重ねる
# 月次合計の理論的な平均・標準偏差
lognorm_mean = np.exp(MU + SIGMA**2 / 2)  # 対数正規分布の期待値
lognorm_var = (np.exp(SIGMA**2) - 1) * np.exp(2 * MU + SIGMA**2)  # 分散
total_mean = N_ORDERS_PER_MONTH * lognorm_mean  # 月次合計の期待値
total_std = np.sqrt(N_ORDERS_PER_MONTH * lognorm_var)  # 月次合計の標準偏差

x = np.linspace(monthly_totals.min(), monthly_totals.max(), 300)
ax_right.plot(
    x,
    stats.norm.pdf(x, loc=total_mean, scale=total_std),
    color="#DD4444",
    linewidth=2.0,
    linestyle="--",
    label=f"正規分布（理論値）\n平均≈{total_mean:.0f}万円",
)

ax_right.set_title(f"月次売上合計の分布\n（{N_ORDERS_PER_MONTH}件×{N_MONTHS}ヶ月分：正規分布に近づく）", fontsize=12)
ax_right.set_xlabel("月次売上合計（万円）", fontsize=11)
ax_right.set_ylabel("確率密度", fontsize=11)
ax_right.legend(fontsize=9, loc="upper right")
ax_right.grid(axis="y", linestyle=":", alpha=0.6)

fig.suptitle("中心極限定理：元データが偏っていても、月次合計は正規分布に従う", fontsize=13, y=1.02)

plt.tight_layout()

# --- 保存 ---
fig_dir = pathlib.Path(__file__).parent / "fig"
fig_dir.mkdir(exist_ok=True)

output_path = fig_dir / "clt_sales_simulation.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"saved: {output_path}")
