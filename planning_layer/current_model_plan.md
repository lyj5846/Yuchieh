# Current Model Plan

- Generated: 2026-07-02 23:48:05
- Data latest date: 2026-06-30
- Status: waiting_for_user_confirmation
- Confirmation required: true

## Problem

The target review found the old +3% touch target is too broad. The next step is retraining the single deep-learning main model on the risk-adjusted success label.

## Recommended Next Step

- Experiment id: `risk_adjusted_main_model_training_plan`
- Hypothesis: The single integrated deep-learning route should train on risk_adjusted_10d_success so it learns clean +3% trades, not old +3% touches that first hit -3% adverse risk.
- Why now: The target redefinition review found enough samples and showed many old successes fail the adverse-first rule, so the next step is retraining the existing main model against the cleaner target.

This is recommended because the target review already passed. The main question is now whether the existing integrated deep-learning route can learn the cleaner risk-adjusted label without creating another model branch.

## Candidate Experiments

### risk_adjusted_main_model_training_plan

- Hypothesis: The single integrated deep-learning route should train on risk_adjusted_10d_success so it learns clean +3% trades, not old +3% touches that first hit -3% adverse risk.
- Why now: The target redefinition review found enough samples and showed many old successes fail the adverse-first rule, so the next step is retraining the existing main model against the cleaner target.
- Target labels: `risk_adjusted_10d_success, old_target_success_comparison, failure_risk, same_day_relative_advantage, episode_start`
- Allowed inputs: `stock_daily_all, market_daily, theme_group`
- Expected outputs: `main_model_training_spec.md, main_model_validation_summary.csv, main_model_decision.md`

### validation_harness_first_plan

- Hypothesis: A stronger validation harness may prevent another cycle of attractive research results that cannot become formal output.
- Why now: The project now has contracts, but planning can still fail if acceptance checks are not executable before training.
- Target labels: `10_day_success, same_day_relative_advantage, failure_risk`
- Allowed inputs: `stock_daily_all, market_daily, theme_group`
- Expected outputs: `validation_harness_spec.md, validation_harness_checks.csv`

### episode_and_risk_focus_plan

- Hypothesis: Focusing on repeated signals and failure separation may reduce duplicate wave recommendations before the full model route is trained.
- Why now: Previous research suggested repeated signals and weak failure separation caused confusion, but this should be integrated rather than exposed as another branch.
- Target labels: `episode_start, failure_risk, 10_day_success`
- Allowed inputs: `stock_daily_all, market_daily, theme_group`
- Expected outputs: `episode_risk_plan_summary.md, episode_risk_validation.csv`

## Boundaries

- This plan does not select stocks.
- This plan does not train a model.
- This plan does not update formal output.
- Raw model scores must not be called success rates unless calibration passes.
