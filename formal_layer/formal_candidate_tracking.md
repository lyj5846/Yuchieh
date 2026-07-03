# Formal Candidate Tracking Replay

- Generated: 2026-07-03 15:26:33
- As-of date: 2026-06-30
- Scope: last 10 signal dates from the formal main-model score file.
- No-lookahead rule: candidates are selected by signal-day research score and the selected gate; outcomes are checked only after selection.
- Buy assumption: next trading day open.
- Success rule: within the next 10 trading days, any close reaches buy open +3%.
- Drawdown: -3% low is tracked as risk context, not automatic failure.
- research_score is not a calibrated probability.

## Status Counts

- not_started: 3
- success: 7
- tracking: 8

## Files

- `formal_layer\formal_candidate_tracking.csv`
