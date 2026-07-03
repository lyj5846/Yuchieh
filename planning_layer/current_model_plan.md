# Current Model Plan

- Generated: 2026-07-03 09:21:00
- Data latest date: 2026-06-30
- Status: waiting_for_user_confirmation
- Confirmation required: true

## Problem

The hard -3% risk-adjusted target failed because it rejected many valid +3% rebound successes. The next step is feature-screened retraining of the single deep-learning main model with +3% touch as primary success and -3% drawdown as a risk side label.

## Recommended Next Step

- Experiment id: `drawdown_side_label_main_model_training_plan`
- Hypothesis: The single integrated deep-learning route should retrain with the +3% touch rule as the primary target while learning -3% drawdown as a risk side label, so valid rebound successes are not incorrectly treated as failures.
- Why now: The drawdown side-label review showed that many valid +3% successes first touched -3%; the previous hard-risk target was too strict for the user's trading question. The next step is retraining the same main model with corrected labels, not a new model branch.

This is recommended because the target review showed that the hard -3% rule rejected valid +3% rebound successes. The main question is whether the existing integrated deep-learning route improves after treating drawdown as risk context instead of automatic failure.

## Candidate Experiments

### drawdown_side_label_main_model_training_plan

- Hypothesis: The single integrated deep-learning route should retrain with the +3% touch rule as the primary target while learning -3% drawdown as a risk side label, so valid rebound successes are not incorrectly treated as failures.
- Why now: The drawdown side-label review showed that many valid +3% successes first touched -3%; the previous hard-risk target was too strict for the user's trading question. The next step is retraining the same main model with corrected labels, not a new model branch.
- Target labels: `target_success_10d_plus3_touch, drawdown_minus3_side_label, clean_success_label, painful_success_label, failure_risk, same_day_relative_advantage, episode_start`
- Allowed inputs: `stock_daily_all, market_daily, theme_group`
- Expected outputs: `main_model_training_spec.md, main_model_feature_screen.csv, main_model_validation_summary.csv, main_model_decision.md`

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
