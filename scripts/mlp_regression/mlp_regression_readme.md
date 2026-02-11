# MLP Regression Demo (多層パーセプトロン回帰デモ)

## Problem Setting / 問題設定
- Small sample (e.g., monthly 24–36 observations) and many KPIs (30–50+).
  - 観測数が少ない（月次24〜36等）かつ説明変数が多数（30〜50以上）。
- Explanatory variables are often correlated (bundle-like behavior).
  - 説明変数が互いに相関し、多重共線性の問題を生む。

## Goal / このモックデータのGoal
- Evaluate whether nonlinear models (MLP) improve predictive performance compared to linear baselines while keeping interpretability sufficient for decision making.
- 非線形モデル（MLP）が線形モデルに比べて予測で改善するかを評価しつつ、意思決定へ落とせる説明性を確保する。

## Approach / Goalへのアプローチ方法
- Generate mock data with latent factors and incorporate **strong nonlinear dependence and interactions** in the target (e.g., sigmoid, sin, tanh, products) so that nonlinear models like MLP can show advantage over linear baselines.
- Use time-aware cross-validation (no shuffle) and GridSearchCV to select MLP hyperparameters (hidden layers, alpha, learning rate).
- Compare with linear baseline (Ridge). Use StandardScaler inside CV pipeline to avoid leakage.

## 数式 / Theory
- MLP with one hidden layer (example):

$$
z = W^{(1)} x + b^{(1)}, \quad h = \phi(z), \quad \hat{y} = W^{(2)} h + b^{(2)}
$$

- Loss: MSE with L2 regularization:

$$
\min_{\theta} \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i(\theta))^2 + \alpha \|\theta\|_2^2
$$

## Variables / 変数
- X: 多数の説明変数（digital系, event系, price系, noise）
- y: 正規化された目的変数（0~1）

## モック用データ作成のTips / Tips for mock data
- Create a few latent factors that drive many observed covariates to simulate multicollinearity.
  - 潜在因子から複数の説明変数を作ることで多重共線性を再現する。
- Inject nonlinear transformations (sigmoid, power) into target to assess MLP effectiveness.
  - 目的変数にシグモイドや二乗の項を入れて非線形性を作る。
- Keep n small (24–36) to replicate practical difficulty and to highlight overfitting risks.
  - 実務課題に近づけるためサンプル数を小さく保つ。これにより過学習対策の重要性が明確になる。

## Practical Recommendations / 実務上の推奨
- Always standardize features inside CV.
- Use early_stopping and L2 regularization (alpha) for MLP.
- Provide feature importance (Permutation / SHAP) to stakeholders to maintain trust.

---

このドキュメントは `src/main/mlp_regression/mlp_regression.ipynb` の実装に対応しています。