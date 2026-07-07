# Target Sensitivity Review

- Generated: 2026-07-07 08:50:23
- Data latest date: 2026-07-06
- Scope: label-only target sensitivity; no model training; no stock candidates.
- Formal output: unchanged by this review.

## 白話結論

目前主目標已回到 10 日內 +3%；-3% 只作風險旁支標籤。此審查保留硬風險目標作比較，但不建議把 -3% 再改回自動失敗。

- Review status: `current_target_label_viable_but_model_failed`
- Recommended next step: `review_data_enrichment`
- Current target: `old_touch_3pct_10d`
- Best non-old target by this review: `risk_adjusted_3pct_before_minus3pct_10d`

## Current Target Snapshot

- Holdout success rate: 70.64%
- Holdout adverse-first rate: 0.00%
- Holdout realized rule return: -0.13%
- Split success-rate max gap: 19.01%

## Best Alternative Snapshot

- Target: `risk_adjusted_3pct_before_minus3pct_10d`
- Description: 硬風險比較：10 日內先收盤 +3%，且不能先最低價 -3%。
- Holdout success rate: 38.94%
- Holdout adverse-first rate: 59.88%
- Holdout realized rule return: -0.63%
- Split success-rate max gap: 3.16%

## Candidate Ranking

| target | holdout success | realized return | split gap | decision score |
|---|---:|---:|---:|---:|
| risk_adjusted_3pct_before_minus3pct_10d | 38.94% | -0.63% | 3.16% | 4.905 |
| risk_adjusted_2pct_before_minus3pct_10d | 42.20% | -0.87% | 1.76% | 4.895 |
| risk_adjusted_3pct_before_minus3pct_5d | 36.47% | -0.63% | 7.01% | 4.867 |
| risk_adjusted_3pct_before_minus5pct_10d | 51.40% | -0.76% | 7.21% | 4.852 |
| old_touch_3pct_10d | 70.64% | -0.13% | 19.01% | 1.297 |

## Boundary

- This does not choose stocks.
- This does not train or promote a model.
- This does not use old report scores or artificial labels.
- This review compares target definitions only; it is not a probability report.

## Outputs

- `validation_layer\target_sensitivity_summary.csv`
- `validation_layer\target_sensitivity_monthly.csv`
- `decision_layer\target_sensitivity_decision.json`
