# Data Learnability Review

- Generated: 2026-07-07 08:50:17
- Data latest date: 2026-07-06
- Scope: data/label learnability only; no model training; no stock candidates.
- Formal output: unchanged by this review.

## 白話結論

三份核心資料加已批准候選特徵內，有跨 train/development/holdout 方向穩定的 +3% 成功、回撤風險旁支與報酬排序訊號。

- Review status: `learnable_signal_present`
- Recommended next step: `feature_screen_then_retrain`

## Holdout Snapshot

- Primary +3% success rate: 70.64%
- Hard risk-adjusted comparison success rate: 38.94%
- Success with -3% drawdown side risk: 31.70%
- Drawdown side risk among successes: 44.88%
- Average max adverse return: -8.19%
- Average realized rule return: 5.26%

## Learnable Signal Counts

- Stable success features: 70
- Stable risk-filter features: 102
- Stable return-ranking features: 85

## Top Feature Clues

These are distinguishability clues, not buy reasons; a stable negative direction can still be useful for filtering risk.

- Success target clues: market_margin_5, theme_軟體／電信／雲平台, market_breadth, theme_avg_ret_5, 融資餘額, 加權指數收盤_ret_3, theme_avg_close_vs_ma_10, close_ret_5
- Risk-filter clues: theme_軟體／電信／雲平台, market_breadth, theme_avg_ret_5, 融資餘額, 加權指數收盤_ret_3, theme_avg_close_vs_ma_10, close_ret_5, 融券餘額
- Return-ranking clues: market_margin_5, theme_軟體／電信／雲平台, market_breadth, 融資餘額, 加權指數收盤_ret_3, 融券餘額, volatility_20, 電子指數收盤_ret_3

## Boundary

- This is not a probability model.
- This does not update formal candidates.
- This does not add a new model branch.
- It only decides whether the current core CSV inputs and approved candidate feature inputs contain stable enough signal for the +3% target and drawdown side-risk labels.

## Outputs

- `validation_layer\data_learnability_feature_signal.csv`
- `validation_layer\data_learnability_failure_profile.csv`
- `decision_layer\data_learnability_decision.json`
