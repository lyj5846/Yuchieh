# Current Model Plan

- Generated: 2026-07-03 00:10:11
- Data latest date: 2026-06-30
- Status: waiting_for_user_confirmation
- Confirmation required: true

## Problem

The data learnability review found stable signal in the three CSV inputs, but the full-feature risk-adjusted model failed holdout. The next step is feature-screened retraining of the single deep-learning main model.

## Recommended Next Step

- Experiment id: `risk_adjusted_main_model_training_plan`
- Hypothesis: The single integrated deep-learning route should first screen for train/development-stable features, then retrain on risk_adjusted_10d_success so weak or unstable features do not dilute the useful signal.
- Why now: The data learnability review found stable success, risk-filter, and return-ranking clues, while the full-feature retrain still failed holdout. The next step is feature-screened retraining, not a new model branch.

This is recommended because the data learnability review found stable clues, while the full-feature model diluted them. The main question is whether screened features can improve the existing integrated deep-learning route without creating another model branch.

## Candidate Experiments

### risk_adjusted_main_model_training_plan

- Hypothesis: The single integrated deep-learning route should first screen for train/development-stable features, then retrain on risk_adjusted_10d_success so weak or unstable features do not dilute the useful signal.
- Why now: The data learnability review found stable success, risk-filter, and return-ranking clues, while the full-feature retrain still failed holdout. The next step is feature-screened retraining, not a new model branch.
- Target labels: `risk_adjusted_10d_success, old_target_success_comparison, failure_risk, same_day_relative_advantage, episode_start`
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
