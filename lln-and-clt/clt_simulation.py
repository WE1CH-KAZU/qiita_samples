"""中心極限定理のシミュレーションスクリプト。

サイコロを10回振って出た目の平均を1セットとし、
セット数を10・100・1000と増やすにつれて
その平均値の分布が正規分布に近づく様子を可視化する。
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
N_DICE_PER_SET = 10  # 1セットあたりのサイコロを振る回数
N_SETS_MAX = 1000  # 最大セット数
STAGES = [10, 100, 1000]  # 可視化する段階

# サイコロの理論的な平均と標準偏差
# 一様分布 {1,2,3,4,5,6} の平均 = 3.5、分散 = 35/12
THEORETICAL_MEAN = 3.5
THEORETICAL_STD = np.sqrt(35 / 12)

# 標本平均の標準偏差（標準誤差）= 母標準偏差 / √サンプルサイズ
# 中心極限定理が予測する正規分布の標準偏差
SE = THEORETICAL_STD / np.sqrt(N_DICE_PER_SET)

# --- N_SETS_MAX セット分まとめてシミュレーション ---
# shape: (N_SETS_MAX, N_DICE_PER_SET)
all_rolls = np.random.randint(1, 7, size=(N_SETS_MAX, N_DICE_PER_SET))
# 各セットの平均値（shape: N_SETS_MAX,）
all_means = all_rolls.mean(axis=1)

# --- 描画（1行3列のサブプロット）---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

for ax, n in zip(axes, STAGES):
    means = all_means[:n]  # 最初の n セット分の平均値を使う

    # ヒストグラム（density=True で確率密度に正規化）
    ax.hist(
        means,
        bins=min(n // 3, 30),  # セット数が少ない場合はビン数も絞る
        density=True,
        color="#4C72B0",
        alpha=0.7,
        edgecolor="white",
        label="標本平均のヒストグラム",
    )

    # 中心極限定理が予測する正規分布の曲線
    x = np.linspace(1, 6, 300)
    ax.plot(
        x,
        stats.norm.pdf(x, loc=THEORETICAL_MEAN, scale=SE),
        color="#DD4444",
        linewidth=1.8,
        linestyle="--",
        label=f"正規分布（理論値）\n平均={THEORETICAL_MEAN}, SE≈{SE:.2f}",
    )

    ax.set_title(f"セット数：{n} 回", fontsize=12)
    ax.set_xlabel("出た目の平均値", fontsize=11)
    ax.set_xlim(1, 6)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # 凡例は右端のパネルのみ表示
    if n == STAGES[-1]:
        ax.legend(fontsize=9, loc="upper right")

axes[0].set_ylabel("確率密度", fontsize=11)

fig.suptitle(
    f"中心極限定理：セット数を増やすほど平均値の分布は正規分布に近づく（1セット＝サイコロ{N_DICE_PER_SET}回）",
    fontsize=12,
    y=1.02,
)

plt.tight_layout()

# --- 保存 ---
fig_dir = pathlib.Path(__file__).parent / "fig"
fig_dir.mkdir(exist_ok=True)

output_path = fig_dir / "clt_simulation.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"saved: {output_path}")
