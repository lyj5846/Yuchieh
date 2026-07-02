# GitHub Workflow Contract

GitHub is for engineering control, not for prediction answers.

## Rules

- GitHub issues track one model repair at a time.
- Pull requests must show the model decision, diagnosis, and verification commands.
- Research scores must not be described as calibrated success rates.
- Formal output can only be changed through `scripts/run_main_pipeline.py`.
- Remote CI runs repository checks that do not need private Windows CSV paths.
- Local quality gate runs the full data-bound checks on this machine.

## Required Local Gate

Run this before any commit or pull request:

```powershell
python scripts/run_local_quality_gate.py
```

The local gate verifies:

- planning contract
- model label contract
- model failure diagnosis contract
- architecture contract
- clean project validation
- GitHub connection status

## Remote GitHub Gate

The GitHub Actions workflow runs:

- Python syntax checks
- GitHub governance checks
- repository static contract checks
- planning contract

It does not run data-bound checks that require local Windows CSV paths or large local model output files.

## Current Connection Check

Run:

```powershell
python scripts/check_github_connection.py
```

This writes `validation_layer/github_connection_report.md` and states whether local Git, GitHub CLI, and repository initialization are ready.
