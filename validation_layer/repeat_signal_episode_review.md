# Repeat Signal Episode Review

- Generated: 2026-07-04 10:25:34
- Data latest date: 2026-07-03
- Selected score gate: 1.9442562103271483
- Review type: label-only repeat signal episode review.
- Formal output: unchanged by this review.
- This does not choose stocks.
- This does not train or promote a model.
- research_score is a ranking score, not a probability.
- 10 trading days is a validation window, not a wave boundary.

## Decision

- Status: `keep_tracking_only`
- Recommended next step: `keep_formal_tracking_only`
- Reason: Repeat high-score events do not yet justify opening a new formal buy point; keep them as tracking evidence.

## Holdout Scenario Summary

| 情境 | 已結案筆數 | 成功率 | 成功率差 | 平均最高收盤報酬 | 報酬差 | 贏同日市場率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_repeat_events | 471 | 77.07% | 6.15% | 16.18% | 4.22% | 51.17% |
| within_10_not_success | 43 | 81.40% | 5.96% | 17.60% | 4.15% | 48.84% |
| within_10_after_success | 104 | 82.69% | 16.12% | 20.81% | 9.85% | 70.19% |
| after_10_reappeared | 324 | 74.69% | 2.97% | 14.51% | 2.43% | 45.37% |
| returned_after_leaving_top10 | 378 | 76.19% | 5.12% | 15.17% | 3.23% | 47.62% |

## Interpretation

- `within_10_after_success` means the previous formal signal had already hit +3% before the repeat high-score day.
- `within_10_not_success` means the previous formal signal was still not successful when the repeat high-score day appeared.
- `after_10_reappeared` checks whether day-count alone is enough evidence.
- `returned_after_leaving_top10` checks whether leaving raw Top10 and coming back is a better reset signal.
