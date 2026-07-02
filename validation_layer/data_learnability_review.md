# Data Learnability Review

- Generated: 2026-07-03 00:06:49
- Data latest date: 2026-06-30
- Scope: data/label learnability only; no model training; no stock candidates.
- Formal output: unchanged by this review.

## 白話結論

三份資料內有跨 train/development/holdout 方向穩定的成功、風險過濾與報酬排序訊號。

- Review status: `learnable_signal_present`
- Recommended next step: `feature_screen_then_retrain`

## Holdout Snapshot

- Risk-adjusted success rate: 39.22%
- Old success but risk-failed rate: 32.72%
- Risk-failed among old successes: 45.48%
- Average max adverse return: -8.05%
- Average realized rule return: -0.61%

## Learnable Signal Counts

- Stable success features: 44
- Stable risk-filter features: 61
- Stable return-ranking features: 51

## Top Feature Clues

These are distinguishability clues, not buy reasons; a stable negative direction can still be useful for filtering risk.

- Success target clues: close_vs_ma_20, close_ret_10, close_vs_ma_10, close_ret_20, close_ret_5, return_vs_electronics_5, return_vs_weighted_5, market_margin_5
- Risk-filter clues: close_vs_ma_20, close_ret_10, close_vs_ma_10, close_ret_20, close_ret_5, return_vs_electronics_5, return_vs_weighted_5, return_vs_electronics_20
- Return-ranking clues: close_vs_ma_20, close_ret_20, market_margin_5, return_vs_electronics_20, return_vs_weighted_20, return_vs_weighted_10, return_vs_electronics_10, same_day_return_rank_3

## Boundary

- This is not a probability model.
- This does not update formal candidates.
- This does not add a new model branch.
- It only decides whether the current three CSV inputs contain stable enough signal for the risk-adjusted target.

## Outputs

- `validation_layer\data_learnability_feature_signal.csv`
- `validation_layer\data_learnability_failure_profile.csv`
- `decision_layer\data_learnability_decision.json`
