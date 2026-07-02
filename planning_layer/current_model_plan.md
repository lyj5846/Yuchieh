# Current Model Plan

- Generated: 2026-07-02 18:08:26
- Data latest date: 2026-06-30
- Status: waiting_for_user_confirmation
- Confirmation required: true

## Problem

The project has a clean architecture and formal entrypoint, but no single trained main model has been accepted under the new contract.

## Recommended Next Step

- Experiment id: `single_main_model_training_plan`
- Hypothesis: A single integrated model route can reduce branch confusion by learning success, risk, relative advantage, and episode start inside one training plan.
- Why now: The architecture is clean, but the main route has not yet been trained through the new planning and validation contract.

This is recommended because it turns the current clean architecture into one trainable main route. The other candidates are useful support work, but they would delay the main question: whether one integrated model can pass validation.

## Candidate Experiments

### single_main_model_training_plan

- Hypothesis: A single integrated model route can reduce branch confusion by learning success, risk, relative advantage, and episode start inside one training plan.
- Why now: The architecture is clean, but the main route has not yet been trained through the new planning and validation contract.
- Target labels: `10_day_success, failure_risk, same_day_relative_advantage, episode_start`
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
