"""大数の法則のシミュレーションスクリプト。

サイコロを繰り返し振り、「1の目が出る確率」が試行回数の増加とともに
理論値（1/6）に収束していく様子を 10・100・1000 回の3段階で可視化する。
"""

import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 日本語フォントの設定（macOS）
plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

# 乱数シード固定（再現性確保）
np.random.seed(28)

# --- シミュレーション設定 ---
N_MAX = 1000  # 最大試行回数
THEORETICAL = 1 / 6  # 理論値（1の目が出る確率）
STAGES = [10, 100, 1000]  # 可視化する段階

# --- サイコロを N_MAX 回まとめて振る ---
rolls = np.random.randint(1, 7, size=N_MAX)
is_one = (rolls == 1).astype(float)  # 1の目 → 1、それ以外 → 0
cumulative_prob = np.cumsum(is_one) / np.arange(1, N_MAX + 1)

# --- 描画（1行3列のサブプロット）---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for ax, n in zip(axes, STAGES):
    trial_axis = np.arange(1, n + 1)

    # 実測の累積確率
    ax.plot(
        trial_axis,
        cumulative_prob[:n],
        color="#4C72B0",
        linewidth=1.5,
        label="実測の累積確率",
    )

    # 理論値の水平線
    ax.axhline(
        y=THEORETICAL,
        color="#DD4444",
        linewidth=1.2,
        linestyle="--",
        label=f"理論値 1/6 ≈ {THEORETICAL:.3f}",
    )

    ax.set_title(f"試行回数：{n} 回", fontsize=12)
    ax.set_xlabel("試行回数", fontsize=11)
    ax.set_xlim(1, n)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # 凡例は右端のパネルのみ表示
    if n == STAGES[-1]:
        ax.legend(fontsize=10, loc="upper right")

# 左端のパネルにのみ y 軸ラベルを付ける
axes[0].set_ylabel("1の目が出た確率", fontsize=11)

fig.suptitle("大数の法則：試行回数を増やすほど確率は理論値（1/6）に近づく", fontsize=13, y=1.02)

plt.tight_layout()

# --- 保存 ---
fig_dir = pathlib.Path(__file__).parent / "fig"
fig_dir.mkdir(exist_ok=True)

output_path = fig_dir / "lln_simulation.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"saved: {output_path}")
