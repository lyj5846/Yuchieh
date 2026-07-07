# Theme Rotation Feature Review

- Generated: 2026-07-07 08:51:14
- Data latest date: 2026-07-06
- Review type: feature learnability review only.
- Formal output: unchanged by this review.
- This does not choose stocks.
- This does not train or promote a model.
- This does not create a second decision layer.
- Holdout columns are audit-only, not used for feature selection.
- research_score is not used and no probability is produced.

## Decision

- Status: `candidate_for_main_model_feature_integration`
- Recommended next step: `plan_theme_rotation_feature_contract`
- Reason: Theme rotation features show stable train/development learnability and can be planned as main-model features.
- Generated theme-rotation features: 50
- Stable candidate features: 37

## Top Feature Signals

| feature | stable metrics | dev success corr | dev return corr | dev same-day rank corr | holdout return corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| theme_strength_rank_20 | 3 | 14.60% | 19.67% | 21.17% | 11.21% |
| theme_ma_position_rank_20 | 3 | 13.18% | 18.58% | 19.93% | 9.54% |
| theme_strength_rank_10 | 3 | 12.19% | 17.69% | 18.86% | 7.73% |
| theme_acceleration_rank_5_20 | 3 | -13.16% | -17.70% | -18.46% | -8.82% |
| theme_vs_weighted_20 | 3 | 10.80% | 15.95% | 18.25% | 11.55% |
| theme_breadth_rank_20 | 3 | 12.62% | 16.53% | 17.70% | 3.55% |
| theme_ma_position_rank_10 | 3 | 10.36% | 14.16% | 15.84% | 8.68% |
| theme_vs_weighted_10 | 3 | 7.24% | 12.96% | 15.16% | 9.62% |
| theme_breadth_rank_10 | 3 | 9.31% | 14.42% | 14.48% | 5.10% |
| theme_strength_rank_5 | 3 | 8.51% | 11.90% | 13.74% | 8.11% |
| theme_breadth_rank_5 | 3 | 6.66% | 11.11% | 11.63% | 5.51% |
| stock_vs_theme_ret_20 | 3 | 5.47% | 11.17% | 9.39% | 10.25% |
| theme_ma_position_rank_5 | 3 | 6.75% | 9.84% | 11.01% | 8.26% |
| theme_strength_rank_3 | 3 | 6.29% | 9.58% | 10.31% | 7.99% |
| theme_median_ret_5 | 3 | -8.75% | -7.58% | 6.22% | -4.47% |

## Plain Meaning

- This review checks whether group rotation can be useful as input material for the single main model.
- It is not a new stock-picking module.
- If accepted later, useful fields must be integrated into the existing main model feature contract.
- The formal daily report must remain one report, not a comparison between branches.
