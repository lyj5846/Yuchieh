# [model] redefine_return_target: review the 10-day +3% target

## Problem

The clean main model has been rebuilt and score weighting has been repaired, but it still cannot be promoted.

The current diagnosis points to `redefine_return_target`, not another weight tweak.

## Evidence

- Data latest date: 2026-06-30
- Main model status: `not_promoted`
- Holdout success rate: 61.90%
- Same-day market baseline: 69.93%
- Holdout success lift: -8.02%
- Holdout return lift: +3.10%
- Development monthly stability: 3/3
- Selected weight stability passed: True
- Same-day return ranking feature stability: 9/10

Diagnosis summary:

> Weighting passed development monthly stability, and return lift is positive, but holdout success rate still loses to the same-day market baseline. The next step should review the formal trading target instead of continuing patch-style score weighting.

## Goal

Review the current formal target:

`Buy at next trading day open; success if any close within the next 10 trading days reaches +3%.`

Decide whether this target is too broad or conflicts with actual stock-selection quality, then propose one replacement target contract.

## Constraints

- Use only the three approved CSV inputs.
- Do not use old models, old reports, old scores, or manual conclusions.
- Do not describe research ranking scores as probabilities.
- Do not add another parallel model branch.
- Do not restore formal candidates before the new target contract is validated.

## Required Checks

Before closing this issue:

```powershell
python scripts/run_local_quality_gate.py
```

Formal output must remain unchanged unless intentionally updated through:

```powershell
python scripts/run_main_pipeline.py
```
