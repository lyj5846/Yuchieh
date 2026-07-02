# Architecture Contract

This project has one formal path and one research path.

## Formal Path

The only formal execution command is:

```powershell
python scripts/run_main_pipeline.py
```

Formal output files are:

- `formal_layer/formal_status.md`
- `formal_layer/formal_candidates.csv`

Only `scripts/run_main_pipeline.py` may write those files.

## Fixed Flow

The formal path is:

```text
three CSV inputs
-> data checks
-> label contract
-> feature contract
-> main model contract
-> planning contract
-> validation contract
-> decision
-> formal output
```

No experiment script can bypass this path.

The planning path is separate from formal output. It can propose the next model experiment, but it cannot train, select stocks, or update formal files.

## Layer Roles

- `data_layer`: reads and checks the three allowed CSV files.
- `label_layer`: defines success, failure, same-day market comparison, episode grouping, and tracking-only status.
- `feature_layer`: defines features that are knowable on the signal day.
- `model_layer`: contains the single main model contract.
- `planning_layer`: produces the next model experiment plan and waits for user confirmation.
- `validation_layer`: checks baseline, benchmark, concentration, monthly stability, and calibration.
- `decision_layer`: decides whether the main model can become formal.
- `formal_layer`: contains only formal status and formal candidates.
- `research_layer`: stores experiments that are not formal.

## Model Contract

The main model may learn:

- 10 trading day success.
- Failure risk.
- Same-day relative advantage.
- Episode starting point.

The final formal surface must not expose multiple competing branches. If a model is not calibrated, its value is a research ranking score, not a success rate.

## Promotion Rule

A model can replace the current formal benchmark only if holdout validation passes all checks:

- It beats the same-day market baseline.
- It is not weaker than the current benchmark by more than the agreed tolerance.
- Higher score bands perform better than lower bands.
- Higher risk bands fail more often than lower risk bands.
- It does not depend on a small group of stocks, dates, or industries.
- It outputs at most Top 3 formal candidates per day.

If the model does not pass, the formal status remains on the current benchmark and no research candidates are copied into the formal file.
