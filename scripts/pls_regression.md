# PLS Regression Demo for Decision Support (with PCR / Ridge / Lasso baselines) 
This repository provides a reproducible demo pipeline for **Partial Least Squares (PLS) regression** in small-sample, multicollinear KPI settings (typical in corporate indirect functions), including comparisons with **PCR / Ridge / Lasso** and cross-validation-based model selection.  
本リポジトリは、間接部門で典型的な「サンプル数が少ない」「KPI間の多重共線性が強い」状況を想定し、**PLS回帰（Partial Least Squares）** を再現可能に試せるデモパイプラインを提供します。**PCR / Ridge / Lasso** との比較と、交差検証に基づくモデル選択（成分数・正則化強度の決定）も含みます。

## Associated Article
- Qiita: （ここに記事URL）  
  - Why PLS is practical under multicollinearity + small n  
    多重共線性＋少サンプルでPLSが実務的に有効な理由  
  - How to select #components via CV  
    交差検証で成分数（n_components）を選ぶ方法  
  - How to interpret components (weights / loadings / score correlations)  
    成分解釈（weights / loadings / 成分スコアとXの相関）の見方  

## Problem Setting
In indirect business functions (marketing / PR / planning), we often face:  
間接部門（マーケティング／広報／企画など）では、次の状況が頻発します。  
- Small number of observations (e.g., monthly data for 2–3 years)  
  観測数が少ない（例：月次データで2〜3年）  
- Many KPIs (30–50+)  
  KPIが多い（30〜50以上）  
- Strong multicollinearity among KPIs (KPIs move in bundles)  
  KPI間の多重共線性が強い（KPIが“束”で動く）  

Goal:  
目的：  
- Predict a business outcome (e.g., deal conversion rate)  
  事業成果（例：商談化率）を予測する  
- Provide stable, update-resistant explanations for decision-making  
  意思決定に使える、更新してもブレにくい説明を提供する  

## Approach (Why PLS)
- **OLS** becomes unstable under strong multicollinearity and p≈n.  
  **OLS（重回帰）** は、多重共線性が強く p≈n の状況で不安定になりやすいです。  
- **PCR** reduces X via PCA, but PCA optimizes variance in X (not necessarily aligned with y).  
  **PCR（主成分回帰）** はPCAでXを圧縮しますが、PCAはXの分散最大化を優先するため、yに効く方向と一致するとは限りません。  
- **Ridge/Lasso** improves generalization, but coefficient shrinkage/selection can be harder to translate into KPI governance.  
  **Ridge/Lasso** は汎化性能を上げやすい一方、係数の縮小・選択の理由をKPI運用（追う指標の合意形成）に翻訳しにくい場合があります。  
- **PLS** compresses X into components aligned with covariance between X and y, often producing more decision-friendly “KPI bundles”.  
  **PLS** はXとyの共分散に整合する方向でXを成分に圧縮するため、意思決定で扱いやすい「KPIの束」を作りやすいことが多いです。  

## Method Details
### Preprocessing
- Standardize X using `StandardScaler` (fit on training folds only)  
  `StandardScaler` でXを標準化します（学習foldのみにfitし、リークを避けます）。  

### Cross-Validation Design
- K-fold with non-shuffled splits to mimic time-ordered evaluation (quarter-based folds):  
  時系列順の評価を模すため、シャッフルしないK-fold（四半期単位のfold）を用います。  
  - `n_splits = len(X) // 4`  
  - `KFold(n_splits=n_splits, shuffle=False)`  

### Model Selection / Metrics
- PLS/PCR: select `n_components` by minimizing CV RMSE  
  PLS/PCR：CV RMSEが最小となる `n_components` を選びます。  
- Ridge/Lasso: select `alpha` by minimizing CV RMSE  
  Ridge/Lasso：CV RMSEが最小となる `alpha` を選びます。  
- Metric: RMSE (primary), optional: AIC/BIC (as a reference only)  
  指標：RMSEを主指標とし、AIC/BICは参考（補助）として扱います。  

## Repository Structure of this Demo
- `src/main/pls_regression` : notebook (All included)


## Requirements
- Python 3.11+ (recommended)  
  Python 3.11以上（推奨）  
- Key packages: numpy, pandas, scikit-learn, matplotlib  
  主要パッケージ：numpy, pandas, scikit-learn, matplotlib  
