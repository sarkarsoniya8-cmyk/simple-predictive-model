# Findings Report

## Problem
Predict diabetes disease progression (a continuous score) from 10 baseline patient features. Regression task.

## Data preparation
- Source: `sklearn.datasets.load_diabetes` (442 rows, 10 features).
- No missing values; all features are numeric and already mean-centred and scaled, so no extra cleaning or scaling was needed.
- Features (X) and target (y) were separated, then split 80/20 (353 train / 89 test) with `random_state=42` for reproducibility.

## Model selection
Linear Regression was chosen because the target is continuous and it is a simple, interpretable baseline whose coefficients show how each feature relates to the target. A model that always predicts the training mean was added as a reference point.

## Evaluation (test set)
| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Mean baseline | 5361.5 | 73.2 | 64.0 | -0.012 |
| Linear Regression | 2900.2 | 53.9 | 42.8 | 0.453 |

Train R² was 0.528, close to the test R² of 0.453.

## What the metrics indicate
- Test MSE drops by about 46% against the baseline, so the model learns real signal.
- R² of 0.45 means about 45% of the variance in the target is explained. Predictions are off by roughly 54 points (RMSE) on average, which is sizeable relative to the target's spread.
- The small gap between train and test R² suggests the model is underfitting rather than overfitting.
- Residual plots (see `images/evaluation_plots.png`) show errors roughly centred on zero and roughly bell-shaped, with no obvious funnel or curve. The wide scatter in actual vs predicted, though, shows that many predictions are still far off.
- Largest coefficients by size: s1 (negative), s5, BMI, s2 and BP. s1 and s2 are serum measures that are likely correlated with each other, so their individual coefficients are less reliable.

## Viability and next steps
Linear Regression is a reasonable baseline but not accurate enough for real clinical use. Next steps: regularised models (Ridge, Lasso), non-linear models (random forest, gradient boosting), feature engineering, and k-fold cross-validation for more stable estimates.
