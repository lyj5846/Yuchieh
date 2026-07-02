# stock_ai_probability_v2

This is the clean Taiwan-stock AI selection line.

Allowed inputs:

- `stock_daily_all.csv`
- `market_daily.csv`
- `inputs/user_theme_group_v2_2026-06-16.csv`

Do not import or reuse outputs from any old model project.

Formal execution rule:

- Run only `python scripts/run_main_pipeline.py` for formal output.
- Run `python scripts/run_planning_pipeline.py` only to create the next model experiment plan.
- Research scripts write to `research_layer` only.
- Before a model passes calibration and holdout validation, the formal result remains:
  `目前無可信正式候選`
