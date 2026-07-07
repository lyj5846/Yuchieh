# Raw Rank Bucket Selection Policy Decision

- Generated: 2026-07-07 08:50:47
- Status: `restrict_formal_candidates_to_raw_top10`
- Basis: `holdout`
- Recommended policy: `raw_top10_only_do_not_fill_from_raw_11_plus`
- Reason: raw_11_plus underperformed raw_top4_10 or failed to keep positive market lift.
- Formal output is unchanged by this review.
- This does not train or promote a model.
- research_score remains a ranking score, not a probability.

- success_drop_vs_raw_top4_10: -6.17%
- return_drop_vs_raw_top4_10: 2.31%
- raw_11_plus_success_lift: 0.40%
- raw_11_plus_return_lift: 3.41%