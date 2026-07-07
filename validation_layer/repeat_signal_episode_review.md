# Repeat Signal Episode Review

- Generated: 2026-07-07 08:50:43
- Data latest date: 2026-07-06
- Selected score gate: 2.1172019958496096
- Review type: label-only repeat signal episode review.
- Formal output: unchanged by this review.
- This does not choose stocks.
- This does not train or promote a model.
- research_score is a ranking score, not a probability.
- 10 trading days is a validation window, not a wave boundary.

## Decision

- Status: `allow_reentry_after_reset`
- Recommended next step: `plan_reentry_label_contract`
- Reason: Returned-after-leaving-Top10 repeats passed the holdout lift, return, and concentration checks.

## Holdout Scenario Summary

| 情境 | 已結案筆數 | 成功率 | 成功率差 | 平均最高收盤報酬 | 報酬差 | 贏同日市場率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_repeat_events | 545 | 79.08% | 8.54% | 15.94% | 4.10% | 51.74% |
| within_10_not_success | 184 | 73.91% | -0.78% | 16.26% | 3.09% | 46.20% |
| within_10_after_success | 317 | 83.91% | 15.38% | 15.74% | 4.61% | 54.57% |
| after_10_reappeared | 44 | 65.91% | -1.80% | 15.97% | 4.62% | 54.55% |
| returned_after_leaving_top10 | 286 | 83.22% | 14.77% | 17.64% | 6.49% | 61.54% |

## Interpretation

- `within_10_after_success` means the previous formal signal had already hit +3% before the repeat high-score day.
- `within_10_not_success` means the previous formal signal was still not successful when the repeat high-score day appeared.
- `after_10_reappeared` checks whether day-count alone is enough evidence.
- `returned_after_leaving_top10` checks whether leaving raw Top10 and coming back is a better reset signal.
