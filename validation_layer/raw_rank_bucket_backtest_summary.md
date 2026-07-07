# Raw Rank Bucket Backtest

- Generated: 2026-07-07 08:50:47
- Latest replayed date: 2026-07-06
- Selected gate: 2.1172019958496096
- Purpose:制度審查；不重訓模型、不改正式候選。
- research_score is a ranking score, not a probability.
- 10日+3% success rate in this report is historical bucket performance, not individual stock probability.

## Decision

- Status: `restrict_formal_candidates_to_raw_top10`
- Basis: `holdout`
- Recommended policy: `raw_top10_only_do_not_fill_from_raw_11_plus`
- Reason: raw_11_plus underperformed raw_top4_10 or failed to keep positive market lift.

## 2026-07-03 Bucket Check

- 7769 鴻勁: raw rank 9, bucket `raw_top4_10`
- 8046 南電: raw rank 13, bucket `raw_11_plus`
- 6669 緯穎: raw rank 15, bucket `raw_11_plus`

## Bucket Summary

| raw_rank_bucket | split | completed_signals | tracking_signals | success_rate | avg_10d_high_close_return | success_lift | return_lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_top3 | overall | 41 | 0 | 75.61% | 15.72% | 17.04% | 6.94% |
| raw_top3 | train | 14 | 0 | 85.71% | 16.83% | 35.30% | 11.31% |
| raw_top3 | development | 9 | 0 | 88.89% | 22.27% | 23.84% | 12.43% |
| raw_top3 | holdout | 16 | 0 | 68.75% | 12.61% | -0.62% | 1.06% |
| raw_top4_10 | overall | 94 | 2 | 70.21% | 15.29% | 14.34% | 6.65% |
| raw_top4_10 | train | 30 | 0 | 73.33% | 9.68% | 19.71% | 3.26% |
| raw_top4_10 | development | 25 | 0 | 92.00% | 21.93% | 33.61% | 13.51% |
| raw_top4_10 | holdout | 32 | 0 | 65.62% | 17.67% | -2.61% | 5.86% |
| raw_11_plus | overall | 168 | 1 | 65.48% | 14.06% | 2.60% | 3.40% |
| raw_11_plus | train | 16 | 0 | 68.75% | 10.34% | 5.10% | 2.84% |
| raw_11_plus | development | 17 | 0 | 88.24% | 16.97% | 18.13% | 5.02% |
| raw_11_plus | holdout | 117 | 0 | 71.79% | 15.36% | 0.40% | 3.41% |
