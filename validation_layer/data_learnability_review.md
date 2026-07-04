# Data Learnability Review

- Generated: 2026-07-04 10:00:15
- Data latest date: 2026-07-03
- Scope: data/label learnability only; no model training; no stock candidates.
- Formal output: unchanged by this review.

## 白話結論

三份核心資料加已批准候選特徵內，有跨 train/development/holdout 方向穩定的 +3% 成功、回撤風險旁支與報酬排序訊號。

- Review status: `learnable_signal_present`
- Recommended next step: `feature_screen_then_retrain`

## Holdout Snapshot

- Primary +3% success rate: 71.26%
- Hard risk-adjusted comparison success rate: 39.55%
- Success with -3% drawdown side risk: 31.71%
- Drawdown side risk among successes: 44.50%
- Average max adverse return: -8.11%
- Average realized rule return: 5.42%

## Learnable Signal Counts

- Stable success features: 42
- Stable risk-filter features: 71
- Stable return-ranking features: 56

## Top Feature Clues

These are distinguishability clues, not buy reasons; a stable negative direction can still be useful for filtering risk.

- Success target clues: market_margin_5, theme_軟體／電信／雲平台, market_breadth, 融資餘額, 加權指數收盤_ret_3, 融券餘額, volatility_20, close_ret_5
- Risk-filter clues: theme_軟體／電信／雲平台, market_breadth, 融資餘額, 加權指數收盤_ret_3, 融券餘額, volatility_20, close_ret_5, industry_volume_rank_20
- Return-ranking clues: market_margin_5, theme_軟體／電信／雲平台, market_breadth, 融資餘額, 加權指數收盤_ret_3, 融券餘額, volatility_20, volatility_10

## Boundary

- This is not a probability model.
- This does not update formal candidates.
- This does not add a new model branch.
- It only decides whether the current core CSV inputs and approved candidate feature inputs contain stable enough signal for the +3% target and drawdown side-risk labels.

## Outputs

- `validation_layer\data_learnability_feature_signal.csv`
- `validation_layer\data_learnability_failure_profile.csv`
- `decision_layer\data_learnability_decision.json`
