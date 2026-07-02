# Current Model Plan

- Generated: 2026-07-02 23:40:00
- Data latest date: 2026-06-30
- Status: waiting_for_user_confirmation
- Confirmation required: true

## Problem

The clean main model has learned some return-ranking signal, but the current formal target is too broad: `next-day open buy, any close within 10 trading days reaches +3%`.

After score-weight repair, holdout return lift is positive while success lift is still negative versus the same-day market baseline. That means the next step is not another model tweak. The target itself must be reviewed.

## Recommended Next Step

- Experiment id: `risk_adjusted_10d_target_plan`
- Hypothesis: A trade target that requires +3% to occur before a meaningful adverse move will be closer to usable stock selection than the current "eventually touches +3%" label.
- Why now: Weighting passed development stability, but holdout still loses on success lift. Continuing to tune weights would repeat the old loop.

This is recommended because it keeps the 10-day trading idea, but removes misleading cases where a stock only reaches +3% after first becoming a poor trade.

## Candidate Experiments

### risk_adjusted_10d_target_plan

- Hypothesis: Success should mean the trade reaches +3% before a -3% adverse low from the next-day open.
- Target labels: `risk_adjusted_10d_success, max_adverse_return, first_event_day, realized_10d_trade_return`
- Allowed inputs: `stock_daily_all, market_daily, theme_group`
- Expected outputs: `target_redefinition_contract.md, target_redefinition_baseline.csv, target_redefinition_decision.md`

### fixed_horizon_return_target_plan

- Hypothesis: A fixed 5-day or 10-day close return target may be cleaner than "any day touches +3%".
- Target labels: `fixed_5d_close_return, fixed_10d_close_return, same_day_relative_return`
- Allowed inputs: `stock_daily_all, market_daily, theme_group`
- Expected outputs: `fixed_horizon_target_review.md, fixed_horizon_baseline.csv`

### relative_market_outperformance_target_plan

- Hypothesis: The model should learn whether a stock beats same-day market opportunity, not just whether it reaches an absolute return.
- Target labels: `same_day_outperformance, top30_same_day_return, excess_10d_high_close_return`
- Allowed inputs: `stock_daily_all, market_daily, theme_group`
- Expected outputs: `relative_target_review.md, relative_target_baseline.csv`

## Boundaries

- This plan does not train a model.
- This plan does not select stocks.
- This plan does not update formal output.
- Raw scores remain research ranking scores, not probabilities.
- The next implementation should only build and compare label contracts.
